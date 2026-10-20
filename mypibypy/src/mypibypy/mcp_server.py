"""MCP server: the deterministic pipeline as tools for Claude Desktop / Claude Code.

Run with:  python -m mypibypy mcp     (stdio; see README for registering it)

This is the agent layer turned around. There, mypibypy drives and asks a model
for a decision in two narrow places. Here an MCP client -- Claude, usually --
drives, and mypibypy is the toolbox. That moves the fixed decision structure out
of the code, so the guarantees that matter have to live in what each tool will
accept and what it will hand back. Three of them, all structural:

* **No tool accepts a sequence, a target, or a path.** Targets are named by id
  from the loci registry (`list_targets`), which the lab maintains; a client
  cannot point the pipeline at a region, an accession or a file of its own.
  `FORBIDDEN_MCP_PARAMETERS` plus `test_mcp_server.py` enforce it on the live
  tool schemas, so a tool added later that takes one fails the build.
* **No tool returns a base.** Results carry pair names, coordinates, GC, Tm,
  flags and screen verdicts. The oligos themselves go only into the order
  workbook, written locally under a name the server chooses. Lab sequences are
  unpublished; they do not travel to a model provider in a tool result.
* **A screen that did not run says so.** Every pair reports `NOT SCREENED` with
  its reason, and placeholder systems or loci are announced in every result.

Every design call lands in the same run record the agent loop writes
(`agent.record`), so an MCP-driven design is as traceable as a loop-driven one.

The server is thin on purpose: `pipeline` does the work, `cli.design_warnings`
writes the provenance note, `agent.actions.clamp` bounds the numbers. Nothing
here is a second design path.
"""
from __future__ import annotations

import os
import re
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from . import loci as loci_mod
# Not `from . import systems`: the package re-exports the function `systems`
# under that name, which shadows the module.
from .systems import ENV_VAR as SYSTEMS_ENV_VAR
from .systems import LOCAL_FILENAME as SYSTEMS_LOCAL_FILENAME
from .systems import is_example as systems_are_example
from .systems import loaded_from as systems_loaded_from
from .systems import systems
from .agent import record as rec
from .agent.actions import (RELAXING_ACTIONS, Action, Decision, DesignParameters,
                            ParameterRefused, apply_action, clamp)
from .agent.loop import Evaluation, Situation, classify
from .agent.states import (BudgetExceeded, Event, Machine, State, fire_logged)
from .agent.tools import (FORBIDDEN_TOOL_PARAMETERS, WindowStats,
                          screen_evidence)
from .cli import design_warnings, example_warnings
from .design import DEFAULT_MIN_SPACING_NT, PADS_DIRECT
from .expression import ENV_EXPRESSION, resolve_expression
from .fold import not_evaluated as fold_not_evaluated
from .loci import load_fusions, load_loci, load_rna_targets, resolve_loci_path
from .pipeline import design_fusion, design_locus, design_rna
from .specificity import SpecificityNotConfigured, resolve_backend
from .workbook import write_workbook

SERVER_NAME = "mypibypy"

#: A transcriptome index for RNA targets. Separate from the genome index on
#: purpose -- see `pipeline.RNA_SCREEN_POLICY` for why RNA is not screened
#: against the genome.
ENV_TRANSCRIPTOME_INDEX = "MYPIBYPY_TRANSCRIPTOME_INDEX"
#: Where order workbooks are written. Default: `designs/` beside the package,
#: which is gitignored.
ENV_OUT_DIR = "MYPIBYPY_MCP_OUT"

#: The same budgets the agent loop defaults to: the first design plus three
#: revisions, and at most three relaxations among them.
MAX_TURNS = 4
MAX_RELAXATIONS = 3

#: What a client may do at a decision point. The loop's closed `Action` set,
#: minus the actions that would do nothing here: `pipeline.design_*` takes a pair
#: count, a spacing and a channel, so relax_tm / relax_gc / widen_region /
#: accept_single_arm_hits would change no result and only spend budget. An action
#: that cannot change the outcome is refused rather than offered.
MCP_ACTIONS = (Action.PROCEED, Action.FLAG_FOR_REVIEW, Action.FAIL,
               Action.REDUCE_PAIRS, Action.SWITCH_SYSTEM)

_STATUS = {State.SUCCEEDED: "succeeded", State.FAILED: "failed",
           State.FLAGGED: "flagged"}

#: Parameter names no MCP tool may take. The agent's list (anything a sequence
#: could be passed as) plus anything that would let a client choose the target
#: or a filesystem location -- the two things the chat path already keeps out of
#: model hands.
FORBIDDEN_MCP_PARAMETERS = FORBIDDEN_TOOL_PARAMETERS | frozenset({
    "region", "interval", "chrom", "start", "end", "build", "coordinates",
    "accession", "transcript", "junction", "breakpoint",
    "path", "file", "filename", "out", "output", "out_path", "directory", "dir",
    "loci", "index", "database", "db", "blast_db", "bowtie_index",
})

#: Anything that looks like oligo-length DNA. Used by the tests to prove tool
#: output carries no bases; 15 nt is shorter than any pad (25) or arm.
DNA_RUN = re.compile(r"[ACGTacgt]{15,}")


def _checkout_dir() -> Path:
    """The package's project directory for an editable install, else the cwd.

    An MCP client launches the server from wherever it likes -- often `/` -- so
    relative defaults would land somewhere arbitrary. The files that matter
    (`systems.local.json`, `loci.local.json`, `designs/`) live beside
    `pyproject.toml`, which is what `app.py` relies on too.
    """
    here = Path(__file__).resolve().parents[2]
    return here if (here / "pyproject.toml").is_file() else Path.cwd()


def configure_environment(home: Path | None = None) -> Path:
    """Point unset paths at the files beside the package. Returns that directory.

    Called by `main`, never at import, so importing this module (as the tests
    do) changes nothing. The `status` tool always reports which files were
    actually loaded, so this default can never be ambiguous.
    """
    home = home or _checkout_dir()
    for var, name in ((SYSTEMS_ENV_VAR, SYSTEMS_LOCAL_FILENAME),
                      (loci_mod.ENV_VAR, loci_mod.LOCAL_FILENAME)):
        if not os.environ.get(var) and (home / name).is_file():
            os.environ[var] = str(home / name)
    if not os.environ.get(rec.ENV_VAR):
        os.environ[rec.ENV_VAR] = str(home / rec.LOCAL_FILENAME)
    if not os.environ.get(ENV_OUT_DIR):
        os.environ[ENV_OUT_DIR] = str(home / "designs")
    return home


# --- target registry ---------------------------------------------------------

@dataclass(frozen=True)
class Target:
    """A registry entry addressable by id. The id is the only handle a client gets."""

    id: str
    kind: str        # locus | fusion | rna
    obj: object

    def describe(self) -> dict:
        o = self.obj
        if self.kind == "locus":
            return {"id": self.id, "kind": "locus", "gene": o.gene, "name": o.name,
                    "location": str(o.interval), "system": o.system,
                    "description": o.description}
        if self.kind == "fusion":
            return {"id": self.id, "kind": "fusion",
                    "gene": f"{o.five_prime_gene}::{o.three_prime_gene}",
                    "name": o.name, "location": "fusion junction (sequence in loci file)",
                    "system": o.system, "system_antisense": o.antisense_system,
                    "description": o.description}
        return {"id": self.id, "kind": "rna", "gene": o.gene, "name": o.name,
                "location": o.accession, "system": o.system,
                "description": o.description}


def _try_load(loader) -> tuple[list, str]:
    """A loader's entries, or none plus why. One missing kind must not hide the rest."""
    try:
        return list(loader()), ""
    except loci_mod.LociNotConfigured as exc:
        return [], str(exc)


def load_targets() -> tuple[dict[str, Target], list[str]]:
    """Every locus, fusion and RNA target in the loci file, keyed by id.

    Ids are `kind:gene:name` (`fusion:name` for a junction, which has two genes).
    A duplicated id is dropped and reported rather than resolved by order: a
    client asking for it could otherwise get whichever came first in the file.
    """
    found: list[Target] = []
    problems: list[str] = []
    entries, why = _try_load(load_loci)
    if why:
        problems.append(f"loci: {why}")
    found += [Target(f"locus:{L.gene}:{L.name}", "locus", L) for L in entries]
    entries, why = _try_load(load_fusions)
    if why:
        problems.append(f"fusions: {why}")
    found += [Target(f"fusion:{j.name}", "fusion", j) for j in entries]
    entries, why = _try_load(load_rna_targets)
    if why:
        problems.append(f"rna_targets: {why}")
    found += [Target(f"rna:{t.gene}:{t.name}", "rna", t) for t in entries]

    targets: dict[str, Target] = {}
    dupes = set()
    for t in found:
        if t.id in targets:
            dupes.add(t.id)
        targets[t.id] = t
    for d in sorted(dupes):
        del targets[d]
        problems.append(f"{d} appears more than once in the loci file and was "
                        f"withheld; give each entry a distinct gene/name")
    return targets, problems


# --- result rendering: names, numbers and verdicts, never bases ---------------

def pair_summary(index: int, pair, screen, access, folding=None) -> dict:
    """One ordered pair, as a client sees it. Deliberately has no sequence field."""
    cand = pair.candidate
    folding = folding or fold_not_evaluated("probe folding not checked for this pair")
    return {
        "pair": index,
        "p1l_name": pair.p1l_name,
        "p1r_name": pair.p1r_name,
        "footprint": str(cand.interval),
        "gc_left": round(cand.left.gc, 3), "gc_right": round(cand.right.gc, 3),
        "tm_left": round(cand.left.tm, 1), "tm_right": round(cand.right.tm, 1),
        "penalty": round(cand.penalty, 3),
        "qc_flags": list(cand.flags),
        "screened": screen.screened,
        "specificity": screen.verdict if screen.screened else "NOT SCREENED",
        "specificity_detail": screen.summary(),
        "offtarget_genes": list(screen.offtarget_genes[:10]) if screen.screened else [],
        "accessibility": (round(access.pair_nucleation_p, 3)
                          if access.evaluated else "not evaluated"),
        "accessibility_detail": access.summary(),
        "on_target_fraction_bound": (round(screen.on_target_fraction, 4)
                                     if screen.on_target_fraction == screen.on_target_fraction
                                     else None),
        "expression_weighting": ("applied" if screen.expression_evaluated
                                 else f"not evaluated: {screen.expression_reason}"),
        "probe_folding": (folding.limiting.label if folding.evaluated
                          else "not evaluated"),
        "probe_folding_detail": folding.summary(),
    }


def result_summary(res) -> dict:
    return {
        "target": res.locus.name,
        "gene": res.locus.gene,
        "system": res.system.name,
        "fluorophore": res.system.fluorophore,
        "repeat_hits_masked": len(res.rmsk),
        "n_pairs": res.n_pairs,
        "pairs": [pair_summary(i, p, s, a, res.folding_at(i - 1))
                  for i, (p, s, a) in enumerate(res.rows(), start=1)],
    }


# --- the session -------------------------------------------------------------

@dataclass
class StoredDesign:
    """One design and its run: where it is in the state machine decides what a
    client may do with it next."""

    design_id: str
    target_id: str
    results: list
    warnings: list[str]
    requested_pairs: int
    run_id: str
    target: Target | None = None
    params: DesignParameters | None = None
    screen: bool = True
    pad_assignment: str = PADS_DIRECT
    machine: Machine = field(default_factory=lambda: Machine(MAX_TURNS, MAX_RELAXATIONS))
    record: rec.RunRecord | None = None
    why: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def state(self) -> State:
        return self.machine.state


@dataclass
class Session:
    """Server state: resolved backends and the designs made so far.

    Designs are held in memory only. They carry sequence, and the workbook is
    the one place sequence is meant to leave this process.
    """

    genome_backend: object = None
    genome_reason: str = ""
    transcriptome_backend: object = None
    transcriptome_reason: str = ""
    expression: object = None
    expression_reason: str = ""
    designs: dict[str, StoredDesign] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_environment(cls) -> Session:
        s = cls()
        try:
            s.genome_backend = resolve_backend()
        except SpecificityNotConfigured as exc:
            s.genome_reason = str(exc)
        index = os.environ.get(ENV_TRANSCRIPTOME_INDEX)
        if index:
            try:
                s.transcriptome_backend = resolve_backend(bowtie_index=index)
            except SpecificityNotConfigured as exc:
                s.transcriptome_reason = str(exc)
        else:
            s.transcriptome_reason = (f"${ENV_TRANSCRIPTOME_INDEX} is not set; RNA "
                                      f"targets will not be screened")
        try:
            s.expression = resolve_expression()
            if s.expression is None:
                s.expression_reason = (f"${ENV_EXPRESSION} is not set; off-target "
                                       f"genes are not weighted by abundance")
        except (OSError, ValueError) as exc:
            s.expression_reason = f"could not load ${ENV_EXPRESSION}: {exc}"
        return s

    # -- tools ---------------------------------------------------------------

    def status(self) -> dict:
        reg = systems()
        try:
            loci_path = str(resolve_loci_path())
            loci_example = loci_mod.is_example()
        except loci_mod.LociNotConfigured as exc:
            loci_path, loci_example = f"not configured: {exc}", False
        return {
            "systems_file": str(systems_loaded_from()),
            "systems_are_synthetic_examples": systems_are_example(),
            "systems": [{"key": k, "fluorophore": s.fluorophore,
                         "example": s.example} for k, s in sorted(reg.items())],
            "loci_file": loci_path,
            "loci_are_shipped_example": loci_example,
            "genome_screen": (self.genome_backend.describe() if self.genome_backend
                              else f"NOT CONFIGURED -- {self.genome_reason}"),
            "transcriptome_screen": (
                self.transcriptome_backend.describe() if self.transcriptome_backend
                else f"NOT CONFIGURED -- {self.transcriptome_reason}"),
            "expression_weighting": (
                self.expression.describe() if self.expression is not None
                else f"NOT CONFIGURED -- {self.expression_reason}"),
            "workbook_directory": str(out_dir()),
            "run_record": str(rec.resolve_store_path()),
            "designs_this_session": sorted(self.designs),
            "design_states": {i: d.state.value for i, d in sorted(self.designs.items())},
        }

    def list_targets(self) -> dict:
        targets, problems = load_targets()
        out = {"targets": [t.describe() for t in targets.values()]}
        if problems:
            out["problems"] = problems
        warnings = _example_warnings()
        if warnings:
            out["warnings"] = warnings
        return out

    def design_target(self, target_id: str, n_pairs: int = 1,
                      min_spacing: int = DEFAULT_MIN_SPACING_NT,
                      system: str | None = None, screen: bool = True,
                      pad_assignment: str = PADS_DIRECT) -> dict:
        targets, _ = load_targets()
        if target_id not in targets:
            raise ValueError(f"unknown target {target_id!r}. Call list_targets for "
                             f"the ids in the loci file: {sorted(targets)}")
        target = targets[target_id]
        known = tuple(sorted(systems()))

        record = rec.RunRecord(target=target_id, model="mcp-client",
                               request=f"design_target via MCP ({target.kind})")
        machine = Machine(MAX_TURNS, MAX_RELAXATIONS)
        try:
            params, clamps = clamp(DesignParameters(
                n_pairs=n_pairs, min_spacing=min_spacing,
                system=(system or "").lower()), known)
        except ParameterRefused as exc:
            record.log(rec.REFUSAL, 0, str(exc))
            self._finish(record, "failed")
            raise
        for c in clamps:
            record.log(rec.PARAMETER_CLAMP, 0, str(c))
        record.log_parameters(0, params, note="as requested over MCP, after clamping")
        fire_logged(machine, record, Event.CLAMPED)

        design_id = f"d_{uuid.uuid4().hex[:8]}"
        stored = StoredDesign(
            design_id, target_id, [], [str(c) for c in clamps], params.n_pairs,
            record.run_id, target=target, params=params, screen=screen,
            pad_assignment=pad_assignment, machine=machine, record=record)
        self._redesign(stored)
        with self.lock:
            self.designs[design_id] = stored
        return self._payload(stored)

    def decide_design(self, design_id: str, action: str, reason: str,
                      primary_signal: str = "specificity") -> dict:
        """Resolve a design that stopped at a decision point.

        The client plays the model's role in the agent loop, under the same closed
        schema: one `Action`, a bounded reason, a signal. Legal only while the
        design is in the `decide` state; the machine refuses it otherwise.
        """
        stored = self._get(design_id)
        with stored.lock:
            m, record = stored.machine, stored.record
            if Event.APPLY not in m.legal():
                raise ValueError(
                    f"{design_id} is in state {m.state.value!r}, and decide_design "
                    f"only applies while a design is in 'decide'. "
                    f"{self._hint(stored)}")
            try:
                decision = Decision.from_payload(
                    {"action": action, "reason": reason,
                     "primary_signal": primary_signal}, source="mcp-client")
            except ParameterRefused as exc:
                record.log(rec.REFUSAL, m.turn, f"decision refused: {exc}")
                raise
            if decision.action not in MCP_ACTIONS:
                raise ParameterRefused(
                    f"{decision.action.value!r} would change nothing on this server "
                    f"(the pipeline takes a pair count, spacing and channel). "
                    f"Choose one of {[a.value for a in MCP_ACTIONS]}.")
            record.log(rec.MODEL_DECISION, m.turn,
                       f"{decision.action.value}: {decision.reason}",
                       action=decision.action.value,
                       primary_signal=decision.primary_signal,
                       prompted_because=stored.why, source="mcp-client")

            terminal = {Action.PROCEED: Event.PROCEED, Action.FAIL: Event.FAIL,
                        Action.FLAG_FOR_REVIEW: Event.FLAG}
            if decision.action in terminal:
                fire_logged(m, record, terminal[decision.action])
                if decision.action is Action.PROCEED:
                    stored.warnings.append(
                        f"{design_id} was accepted by the MCP client at a decision "
                        f"point ({stored.why}); client's reason: {decision.reason}")
                self._conclude(stored, decision.reason)
                return self._payload(stored)

            self._revise(stored, decision)
        return self._payload(stored)

    def write_order(self, design_ids: list[str]) -> dict:
        if not design_ids:
            raise ValueError("no design_ids given; design_target returns one per call")
        with self.lock:
            missing = [d for d in design_ids if d not in self.designs]
            if missing:
                raise ValueError(f"unknown design id(s) {missing}; this session has "
                                 f"{sorted(self.designs)}. Designs do not survive a "
                                 f"server restart.")
            chosen = [self.designs[d] for d in dict.fromkeys(design_ids)]
        # The gate. A workbook carries sequence, so only a design whose run ended
        # in SUCCEEDED -- a clear pass, or an explicit decision to proceed --
        # may be written. The other states each say what to do instead.
        blocked = {d.design_id: d.state.value for d in chosen
                   if d.state is not State.SUCCEEDED}
        if blocked:
            raise ValueError(
                "cannot write an order for designs that have not succeeded: "
                + "; ".join(f"{i} is {st!r} ({self._hint(self.designs[i])})"
                            for i, st in blocked.items()))
        results = [r for d in chosen for r in d.results]
        warnings = list(dict.fromkeys(w for d in chosen for w in d.warnings))
        note = (f"mypibypy MCP order of {len(results)} design(s): "
                f"{', '.join(d.target_id for d in chosen)}. Run records: "
                f"{', '.join(d.run_id for d in chosen)}.")
        for w in warnings:
            note += f" WARNING: {w}."
        directory = out_dir()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = directory / f"{stamp}_mcp_order_{chosen[0].design_id}.xlsx"
        write_workbook(str(path), results, note)
        for d in chosen:
            d.record.log(rec.TOOL_CALL, d.machine.turn,
                         f"order workbook written: {path.name}")
            rec.RunStore().append(d.record)
        out = {"workbook": str(path), "designs": [d.design_id for d in chosen],
               "primaries": sum(2 * r.n_pairs for r in results)}
        if warnings:
            out["warnings"] = warnings
        return out

    # -- the machine's side of the tools --------------------------------------

    def _get(self, design_id: str) -> StoredDesign:
        with self.lock:
            if design_id not in self.designs:
                raise ValueError(f"unknown design id {design_id!r}; this session has "
                                 f"{sorted(self.designs)}. Designs do not survive a "
                                 f"server restart.")
            return self.designs[design_id]

    def _redesign(self, stored: StoredDesign) -> None:
        """Run the pipeline for the design's current parameters, then evaluate."""
        m, record = stored.machine, stored.record
        try:
            stored.results = self._run(stored.target, stored.params, stored.screen,
                                       stored.pad_assignment)
        except Exception as exc:
            record.log(rec.ERROR, m.turn, f"design failed: {exc}")
            fire_logged(m, record, Event.EXHAUSTED)
            self._conclude(stored, f"design failed: {exc}")
            raise
        fire_logged(m, record, Event.DESIGNED)
        record.log(rec.TOOL_CALL, m.turn, f"designed {stored.target_id} as {stored.design_id}",
                   pairs=[p.p1l_name for r in stored.results for p in r.pairs],
                   screens=[s.summary() for r in stored.results for s in r.pair_screens])
        stored.warnings = (list(dict.fromkeys(
            stored.warnings + _example_warnings()
            + design_warnings(stored.results, stored.params.n_pairs))))

        ev = Evaluation(
            turn=m.turn, windows=WindowStats(),
            n_selected=sum(r.n_pairs for r in stored.results),
            screens=[screen_evidence(s, a) for r in stored.results
                     for _, s, a in r.rows()])
        situation, stored.why = classify(ev, m.relaxations_left)
        if situation is Situation.NEEDS_JUDGEMENT:
            fire_logged(m, record, Event.NEEDS_JUDGEMENT)
            return
        record.log(rec.DETERMINISTIC_DECISION, m.turn, f"{situation.value} -- {stored.why}")
        fire_logged(m, record, Event(situation.value))
        self._conclude(stored, stored.why)

    def _revise(self, stored: StoredDesign, decision: Decision) -> None:
        """Apply a non-terminal action and design again, all under the machine."""
        m, record = stored.machine, stored.record
        known = tuple(sorted(systems()))
        if stored.target.kind == "fusion":
            raise ParameterRefused(
                "a fusion takes exactly one pair and carries its channels in the "
                "loci file, so neither reduce_pairs nor switch_system applies; "
                "proceed, flag_for_review or fail")
        try:
            new, change = apply_action(decision.action, stored.params, known)
        except ParameterRefused:
            new, change = None, ""
        if new == stored.params:
            raise ValueError(
                f"{decision.action.value} would change nothing ({change or 'already at its limit'}); "
                f"choose another action. Nothing was charged to the budget.")

        try:
            fire_logged(m, record, Event.APPLY,
                        relaxing=decision.action in RELAXING_ACTIONS)
        except BudgetExceeded as exc:
            record.log(rec.BUDGET, m.turn, str(exc))
            fire_logged(m, record, Event.EXHAUSTED)
            self._conclude(stored, f"no acceptable design: {exc.what}")
            return
        if new is None:
            record.log(rec.REFUSAL, m.turn, "action refused")
            fire_logged(m, record, Event.REFUSED)
            self._conclude(stored, "the action could not be applied")
            return
        stored.params = new
        record.log(rec.TOOL_CALL, m.turn, f"applied {decision.action.value}: {change}")
        record.log_parameters(m.turn, new, f"after {decision.action.value}")
        try:
            fire_logged(m, record, Event.APPLIED)
        except BudgetExceeded as exc:
            record.log(rec.BUDGET, m.turn, str(exc))
            fire_logged(m, record, Event.EXHAUSTED)
            self._conclude(stored, f"no acceptable design: {exc.what}")
            return
        self._redesign(stored)

    def _conclude(self, stored: StoredDesign, message: str) -> None:
        """Persist the record once the run reaches a terminal state."""
        status = _STATUS.get(stored.state)
        if status is None:
            return
        stored.why = message
        stored.record.status = status
        stored.record.finished_at = rec._now()
        stored.record.log(rec.OUTCOME, stored.machine.turn, f"{status}: {message}")
        rec.RunStore().append(stored.record)

    def _hint(self, stored: StoredDesign) -> str:
        """What a client can do next, in words."""
        s = stored.state
        if s is State.SUCCEEDED:
            return "it can be written with write_order"
        if s is State.DECIDE:
            return (f"call decide_design with one of "
                    f"{[a.value for a in MCP_ACTIONS]} ({stored.why})")
        return ("the run is over and cannot be reopened; call design_target again "
                "for a fresh run")

    def _payload(self, stored: StoredDesign) -> dict:
        m = stored.machine
        out = {"design_id": stored.design_id, "target_id": stored.target_id,
               "kind": stored.target.kind, "run_id": stored.run_id,
               "state": m.state.value, "why": stored.why,
               "next": self._hint(stored),
               "relaxations_left": m.relaxations_left,
               "results": [result_summary(r) for r in stored.results]}
        if stored.warnings:
            out["warnings"] = stored.warnings
        return out

    # -- internals -------------------------------------------------------------

    def _run(self, target: Target, params: DesignParameters, screen: bool,
             pad_assignment: str) -> list:
        if target.kind == "fusion":
            if params.system:
                raise ParameterRefused(
                    "a fusion carries two channels (sense and antisense) in the "
                    "loci file; change them there rather than overriding one here")
            return design_fusion(target.obj, backend=self.genome_backend,
                                 screen=screen, pad_assignment=pad_assignment,
                                 n_pairs=params.n_pairs)
        obj = replace(target.obj, system=params.system) if params.system else target.obj
        if target.kind == "rna":
            return [design_rna(obj, backend=self.transcriptome_backend, screen=screen,
                               n_pairs=params.n_pairs, min_spacing=params.min_spacing,
                               expression=self.expression)]
        return [design_locus(obj, backend=self.genome_backend, screen=screen,
                             n_pairs=params.n_pairs, min_spacing=params.min_spacing)]

    @staticmethod
    def _finish(record: rec.RunRecord, status: str) -> None:
        record.status = status
        record.finished_at = rec._now()
        rec.RunStore().append(record)


def out_dir() -> Path:
    return Path(os.environ.get(ENV_OUT_DIR) or (_checkout_dir() / "designs")).expanduser()


def _example_warnings() -> list[str]:
    try:
        return example_warnings()
    except loci_mod.LociNotConfigured as exc:
        return [f"no loci file: {exc}"]


# --- MCP wiring --------------------------------------------------------------

INSTRUCTIONS = """\
mypibypy designs piFISH split-probe primaries (P1L/P1R) against targets the lab \
has registered in its loci file. Start with `status` and `list_targets`. Targets \
are addressed by id only; you cannot supply a region, accession or sequence, and \
results never include oligo sequences -- those go only into the order workbook \
that `write_order` saves locally. Every design is a run with a state, and \
`design_target` reports it. `succeeded` designs can be ordered. A design in \
`decide` stopped on a real tradeoff or an unscreened pair: resolve it with \
`decide_design`, which is refused in any other state. `write_order` refuses a \
design that has not succeeded. Report any "NOT SCREENED" specificity or \
placeholder-input warning to the user verbatim: it means no off-target screen \
was run, not that the design passed one.
"""


def build_server(session: Session | None = None):
    """The MCP server over a `Session`. Imports the SDK here so the rest of this
    module, and its tests, work without it installed."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    session = session or Session.from_environment()
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    @server.tool(annotations=read_only)
    def status() -> dict:
        """Which systems and loci files are loaded, whether they are placeholders,
        which specificity screens are configured, and where workbooks go."""
        return session.status()

    @server.tool(annotations=read_only)
    def list_targets() -> dict:
        """Every target in the lab's loci file: genomic loci, fusion junctions and
        RNA transcripts, each with the id that design_target takes."""
        return session.list_targets()

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                             openWorldHint=True))
    def design_target(target_id: str, n_pairs: int = 1,
                      min_spacing: int = DEFAULT_MIN_SPACING_NT,
                      system: str | None = None, screen: bool = True,
                      pad_assignment: Literal["direct", "as_ordered"] = PADS_DIRECT,
                      ) -> dict:
        """Design split-probe pairs for one registered target and screen each pair.

        n_pairs tiles that many non-overlapping pairs (1-50; one pair already
        works, more is headroom for weak signal; fusions take exactly 1).
        min_spacing is the gap in nt between tiled footprints. system overrides
        the target's channel by registry key (see status). pad_assignment applies
        to fusion antisense pairs only. Returns names, coordinates, GC/Tm, QC
        flags and screen verdicts per pair, plus a design_id for write_order.
        """
        return session.design_target(target_id, n_pairs=n_pairs,
                                     min_spacing=min_spacing, system=system,
                                     screen=screen, pad_assignment=pad_assignment)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                             idempotentHint=False, openWorldHint=False))
    def decide_design(design_id: str,
                      action: Literal["proceed", "flag_for_review", "fail",
                                      "reduce_pairs", "switch_system"],
                      reason: str,
                      primary_signal: Literal["specificity", "accessibility",
                                              "thermodynamics", "composition",
                                              "geometry", "capacity"] = "specificity",
                      ) -> dict:
        """Resolve a design whose state is 'decide' (design_target says why).

        proceed accepts it as it is, and your reason is recorded against the order;
        flag_for_review and fail end the run; reduce_pairs and switch_system
        revise it and design again. At most three relaxations and four designs per
        run. reason is one or two plain sentences -- never a sequence. Refused
        unless the design is in 'decide'.
        """
        return session.decide_design(design_id, action, reason, primary_signal)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                             idempotentHint=False, openWorldHint=False))
    def write_order(design_ids: list[str]) -> dict:
        """Write one order workbook for the given design_ids (from design_target
        in this session). Refused for any design whose state is not 'succeeded'. The server names the file and saves it in the
        workbook directory shown by status; the path is returned."""
        return session.write_order(design_ids)

    return server


def main() -> None:
    configure_environment()
    build_server().run("stdio")
