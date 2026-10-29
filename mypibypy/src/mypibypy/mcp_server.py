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
Designs survive a server restart: each one's description -- target id,
parameters, run id, warnings, state; never its oligos -- is saved to the design
store (`mcp_store`) as it changes, and a later server rebuilds one on first use
by running the deterministic pipeline again and checking the result against the
digest saved with it. The state it was left in holds, so a `flagged` design from
an earlier conversation still needs an explicit `proceed` before `write_order`.

The server is thin on purpose: `pipeline` does the work, `cli.design_warnings`
writes the provenance note, `agent.actions.clamp` bounds the numbers. Nothing
here is a second design path.
"""
from __future__ import annotations

import os
import re
import threading
import uuid
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from . import loci as loci_mod
from . import mcp_store
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
from .diagnosis import (NOT_PREVIEWABLE, PREVIEW_CAVEAT, diagnose, preview_change,
                        try_changes)
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
    if not os.environ.get(mcp_store.ENV_VAR):
        # Beside the run record, wherever that was just resolved to.
        os.environ[mcp_store.ENV_VAR] = str(mcp_store.resolve_store_path())
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
    } | ({"repacked": (f"greedy placement fit only {res.repacked_from}; the pairs "
                       f"were re-laid out to fit {res.n_pairs}, using some "
                       f"lower-ranked windows")}
         if res.repacked_from else {})


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
    #: The description last written to the design store, so an unchanged one is
    #: not written again.
    saved: dict | None = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def state(self) -> State:
        return self.machine.state

    def to_entry(self) -> dict:
        """What the design store keeps: the description, and a digest of the
        results, never the results. `Session._reload` is its inverse."""
        m = self.machine
        return {
            "design_id": self.design_id,
            "target_id": self.target_id,
            "kind": self.target.kind if self.target else "",
            "run_id": self.run_id,
            "params": self.params.to_dict(),
            "screen": self.screen,
            "pad_assignment": self.pad_assignment,
            "requested_pairs": self.requested_pairs,
            "warnings": list(self.warnings),
            "why": self.why,
            "machine": {"state": m.state.value, "turn": m.turn,
                        "relaxations": m.relaxations,
                        "max_turns": m.max_turns,
                        "max_relaxations": m.max_relaxations},
            "fingerprint": mcp_store.fingerprint(self.results),
        }


@dataclass
class Session:
    """Server state: resolved backends and the designs made so far.

    Designs are held in memory with their results, which carry sequence; the
    workbook is the one place sequence is meant to leave this process. What is
    saved to the design store is each design's description and state, so a
    later server can rebuild it by running the pipeline again (`_reload`).
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
        store = self._store()
        try:
            stored_count = len(store.ids())
        except ValueError as exc:
            stored_count = f"unreadable: {exc}"
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
            "design_store": str(store.path),
            "designs_stored": stored_count,
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
        self._save(stored)
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
            # A flagged design stopped for a human. The client, standing in for
            # one, may accept it as it is -- the machine's single exit from a
            # terminal state -- and nothing else: a revision is a new run.
            overriding = m.state is State.FLAGGED
            if overriding and action != Action.PROCEED.value:
                raise ValueError(
                    f"{design_id} is 'flagged': it stopped for a human "
                    f"({stored.why}). From here the only decision is 'proceed', "
                    f"which accepts the design as it is and records your reason "
                    f"against the order; for anything else call design_target "
                    f"again, which is a new run.")
            if not overriding and Event.APPLY not in m.legal():
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

            if overriding:
                fire_logged(m, record, Event.OVERRIDE)
                stored.warnings.append(
                    f"{design_id} stopped for a human ({stored.why}) and was "
                    f"accepted by the MCP client as it is; client's reason: "
                    f"{decision.reason}")
                self._conclude(stored, decision.reason)
                return self._payload(stored)

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
            self._save(stored)
        return self._payload(stored)

    def write_order(self, design_ids: list[str]) -> dict:
        if not design_ids:
            raise ValueError("no design_ids given; design_target returns one per call")
        # In memory, or rebuilt from the design store: a design from an earlier
        # conversation is ordered the same way as one from this one.
        found = {d: self._lookup(d) for d in dict.fromkeys(design_ids)}
        missing = [d for d, stored in found.items() if stored is None]
        if missing:
            raise ValueError(self._unknown(missing))
        chosen = list(found.values())
        # The gate. A workbook carries sequence, so only a design whose run ended
        # in SUCCEEDED -- a clear pass, or an explicit decision to proceed --
        # may be written. The other states each say what to do instead.
        blocked = [d for d in chosen if d.state is not State.SUCCEEDED]
        if blocked:
            raise ValueError(
                "cannot write an order for designs that have not succeeded: "
                + "; ".join(f"{d.design_id} is {d.state.value!r} ({self._hint(d)})"
                            for d in blocked))
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

    def diagnose_design(self, design_id: str) -> dict:
        """What a design came out as and, if it is short, what would help.

        Read-only: the facts and previews come from `diagnosis`, the layer the
        Streamlit chat reads too, and nothing here moves the run or its budget.
        """
        stored = self._get(design_id)
        with stored.lock:
            facts = diagnose(stored.results, stored.params)
            previews = try_changes(stored.results, stored.params)
            out = {"design_id": design_id, "state": stored.state.value,
                   "facts": [{"topic": f.topic, "text": f.text} for f in facts]}
        if previews:
            out["what_if"] = [_what_if(p) for p in previews]
            out["what_if_note"] = PREVIEW_CAVEAT + " " + _HOW_TO_APPLY
        return out

    def try_change(self, design_id: str, n_pairs: int | None = None,
                   min_spacing: int | None = None) -> dict:
        """Preview a pair count and/or spacing on a design without designing it.

        Read-only, like `diagnose_design`. The numbers are clamped exactly as
        `design_target` clamps them, so a preview never shows a value the
        design would not use.
        """
        stored = self._get(design_id)
        if n_pairs is None and min_spacing is None:
            raise ValueError("give n_pairs, min_spacing or both to preview")
        with stored.lock:
            params = stored.params
            new, clamps = clamp(replace(
                params,
                n_pairs=params.n_pairs if n_pairs is None else n_pairs,
                min_spacing=params.min_spacing if min_spacing is None else min_spacing))
            preview = preview_change(stored.results, params, new)
        if preview is None:
            raise ValueError(f"{design_id} cannot be previewed: {NOT_PREVIEWABLE}")
        out = {"design_id": design_id, "what_if": _what_if(preview),
               "note": PREVIEW_CAVEAT + " " + _HOW_TO_APPLY}
        if clamps:
            out["warnings"] = [str(c) for c in clamps]
        return out

    # -- the machine's side of the tools --------------------------------------

    def _get(self, design_id: str) -> StoredDesign:
        stored = self._lookup(design_id)
        if stored is None:
            raise ValueError(self._unknown([design_id]))
        return stored

    def _lookup(self, design_id: str) -> StoredDesign | None:
        """The design from memory, else rebuilt from the design store and kept
        in memory from then on; None if it is in neither place."""
        with self.lock:
            stored = self.designs.get(design_id)
        if stored is not None:
            return stored
        entry = self._store().get(design_id)
        if entry is None:
            return None
        # The pipeline runs outside the session lock: a screen can take seconds,
        # and `status` should answer meanwhile. If two clients reload the same
        # id at once the first one in wins and the other's copy is dropped.
        stored = self._reload(design_id, entry)
        with self.lock:
            return self.designs.setdefault(design_id, stored)

    def _unknown(self, design_ids: list[str]) -> str:
        with self.lock:
            known = sorted(self.designs)
        return (f"unknown design id(s) {design_ids}: not among this server's designs "
                f"{known} and not in the design store at {self._store().path}. "
                f"Designs are saved there as they are made and survive a server "
                f"restart, so an id in neither place was never made by this server; "
                f"call design_target for a new design.")

    def _reload(self, design_id: str, entry: dict) -> StoredDesign:
        """Rebuild a design saved by an earlier server.

        Same target by id, same parameters, the pipeline run again, and the
        result checked against the digest saved with the entry before the saved
        state is trusted. Nothing here moves the run: the machine is rebuilt at
        the state it was left in, and the gates apply to it exactly as before.
        Every refusal is a `ValueError`/`RuntimeError`, so it reaches the client
        with its reason.
        """
        try:
            target_id = entry["target_id"]
            saved_params, pos = dict(entry["params"]), dict(entry["machine"])
            state = State(pos["state"])
            turn, relaxations = int(pos["turn"]), int(pos["relaxations"])
            screen, pads = bool(entry["screen"]), str(entry["pad_assignment"])
            requested = int(entry["requested_pairs"])
            run_id, why = str(entry["run_id"]), str(entry["why"])
            warnings, digest = list(entry["warnings"]), str(entry["fingerprint"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{design_id}'s entry in the design store "
                             f"({self._store().path}) is not readable: {exc!r}; call "
                             f"design_target for a new design.") from exc
        targets, _ = load_targets()
        target = targets.get(target_id)
        if target is None:
            raise ValueError(
                f"{design_id} was designed for {target_id!r}, which is not in the "
                f"loci file now ({sorted(targets)}), so it cannot be rebuilt; call "
                f"design_target for a new design.")
        allowed = {f.name for f in fields(DesignParameters)}
        params = DesignParameters(**{k: v for k, v in saved_params.items() if k in allowed})
        try:
            checked, clamps = clamp(params, tuple(sorted(systems())))
        except ParameterRefused as exc:
            raise ValueError(f"{design_id} cannot be rebuilt as it was saved: {exc}") from exc
        if clamps or checked != params:
            raise ValueError(
                f"{design_id} cannot be rebuilt as it was saved: its parameters are "
                f"outside what this server accepts ({'; '.join(map(str, clamps))}); "
                f"call design_target for a new design.")
        machine = Machine(max_turns=int(pos.get("max_turns", MAX_TURNS)),
                          max_relaxations=int(pos.get("max_relaxations", MAX_RELAXATIONS)),
                          state=state, turn=turn, relaxations=relaxations)
        try:
            results = self._run(target, params, screen, pads)
        except Exception as exc:
            raise RuntimeError(f"{design_id} could not be regenerated from the loci "
                               f"file and the pipeline: {exc}") from exc
        if mcp_store.fingerprint(results) != digest:
            raise ValueError(
                f"{design_id} was rebuilt for {target_id!r} with its saved parameters, "
                f"but the pipeline did not give back the design that was recorded: "
                f"the target's entry in the loci file, the reference it reads or the "
                f"screen configuration has changed since. Its saved state "
                f"({machine.state.value!r}) no longer describes what the pipeline "
                f"makes, so nothing was loaded; call design_target for a new design, "
                f"which is evaluated afresh.")

        runs = rec.RunStore()
        record = runs.get(run_id)
        note = (f"{design_id} reloaded from the design store after a server restart; "
                f"results regenerated by running the pipeline again, digest matched")
        if record is None:
            # The run record is gone (moved, or the server was pointed elsewhere):
            # continue it under the same run id, saying so.
            record = rec.RunRecord(
                run_id=run_id, target=target_id, model="mcp-client",
                request=f"continuation of {design_id} over MCP; the original run "
                        f"record was not found in {runs.path}",
                status=_STATUS.get(machine.state, "running"),
                turns_used=machine.turn, relaxations_used=machine.relaxations)
            record.log_parameters(machine.turn, params, note="as reloaded from the design store")
            note += "; the original run record was not found, so this record continues it"
        record.log(rec.TOOL_CALL, machine.turn, note, design_store=str(self._store().path))
        runs.append(record)

        stored = StoredDesign(
            design_id, target_id, results, warnings, requested, run_id, target=target,
            params=params, screen=screen, pad_assignment=pads, machine=machine,
            record=record, why=why)
        stored.saved = {k: v for k, v in entry.items() if k != "saved_at"}
        return stored

    def _store(self) -> mcp_store.DesignStore:
        return mcp_store.DesignStore()

    def _save(self, stored: StoredDesign) -> None:
        """Write the design's description to the design store, if it changed.

        Called wherever a run moves: the end of `design_target`, after a revision
        in `decide_design`, and `_conclude`, which also sees the transitions that
        end in a raise. The store is append-only, so an unchanged description is
        not written again.
        """
        entry = stored.to_entry()
        if entry == stored.saved:
            return
        self._store().append(entry)
        stored.saved = entry

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
        # A design the client holds an id for is saved as it moves, including
        # here, where a revision whose pipeline call raised ends. One whose first
        # pipeline call raised was never handed out: `design_target` re-raises
        # before it registers it, so there is nothing a client could come back to.
        with self.lock:
            handed_out = self.designs.get(stored.design_id) is stored
        if handed_out:
            self._save(stored)

    def _hint(self, stored: StoredDesign) -> str:
        """What a client can do next, in words."""
        s = stored.state
        if s is State.SUCCEEDED:
            return "it can be written with write_order"
        if s is State.DECIDE:
            return (f"call decide_design with one of "
                    f"{[a.value for a in MCP_ACTIONS]} ({stored.why})")
        if s is State.FLAGGED:
            return (f"it stopped for a human ({stored.why}); decide_design "
                    f"'proceed' with your reason accepts it as it is and is "
                    f"recorded against the order, or call design_target again "
                    f"for a fresh run")
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


#: Previews do not revise a design: spacing is not an `Action`, and a preview
#: should not spend the run's budget. A change is used by designing afresh.
_HOW_TO_APPLY = ("To use a change, call design_target again with that n_pairs "
                 "and min_spacing; it is a new run.")


def _what_if(p) -> dict:
    return {"change": p.change, "n_pairs": p.params.n_pairs,
            "min_spacing": p.params.min_spacing, "pairs_placed": p.pairs_placed,
            "pairs_requested": p.pairs_requested, "text": p.text}


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
`decide` stopped on a real tradeoff: resolve it with `decide_design`. A design \
in `flagged` stopped for a human, usually because no specificity screen ran; \
`decide_design` with `proceed` and your reason accepts it as it is, and the \
acceptance is recorded against the order. `decide_design` is refused in any \
other state. `write_order` refuses a design that has not succeeded. Design ids \
survive a server restart: a design from an earlier conversation can be diagnosed, \
decided or ordered by its id, is rebuilt by running the pipeline again on first \
use, and keeps the state it was left in. To explain a design or troubleshoot one that \
placed too few pairs, call `diagnose_design`, and `try_change` to preview a \
pair count or spacing; quote their sentences rather than working the numbers \
out yourself. Report any "NOT SCREENED" specificity or \
placeholder-input warning to the user verbatim: it means no off-target screen \
was run, not that the design passed one.
"""


def build_server(session: Session | None = None):
    """The MCP server over a `Session`. Imports the SDK here so the rest of this
    module, and its tests, work without it installed."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    session = session or Session.from_environment()
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    def refusing(fn, *args, **kwargs):
        """Run a `Session` method so that a refusal reaches the client with its reason.

        The SDK treats any exception other than `ToolError` as a crash: the
        model is told only `Error executing tool <name>` and the text stays in
        the server log. Every refusal here is deliberate and carries the hint
        the client needs next -- "call list_targets for the ids", "only applies
        while a design is in 'decide'", "not in the design store" -- so it is
        re-raised as a `ToolError`. Verified over stdio:
        without this, `write_order` on an unknown id came back as the generic
        line and the hint was lost, so the client could not correct itself.
        A genuine bug (anything else) still crashes, generically.
        """
        try:
            return fn(*args, **kwargs)
        except (ValueError, RuntimeError, OSError) as exc:
            # ValueError covers ParameterRefused and FusionIsSinglePair;
            # RuntimeError covers IllegalTransition, LociNotConfigured and a
            # transcript fetch that failed; OSError a reference service down.
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=read_only)
    def status() -> dict:
        """Which systems and loci files are loaded, whether they are placeholders,
        which specificity screens are configured, where workbooks go, and where
        designs are kept between server runs."""
        return refusing(session.status)

    @server.tool(annotations=read_only)
    def list_targets() -> dict:
        """Every target in the lab's loci file: genomic loci, fusion junctions and
        RNA transcripts, each with the id that design_target takes."""
        return refusing(session.list_targets)

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
        return refusing(session.design_target, target_id, n_pairs=n_pairs,
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
        """Resolve a design whose state is 'decide' (design_target says why), or
        accept one whose state is 'flagged'.

        proceed accepts it as it is, and your reason is recorded against the order;
        flag_for_review and fail end the run; reduce_pairs and switch_system
        revise it and design again. From 'flagged' -- the design stopped for a
        human, usually because no specificity screen ran -- only proceed is
        allowed; a revision is a new design_target call. At most three relaxations
        and four designs per run. reason is one or two plain sentences -- never a
        sequence. Refused in any other state.
        """
        return refusing(session.decide_design, design_id, action, reason,
                        primary_signal)

    @server.tool(annotations=read_only)
    def diagnose_design(design_id: str) -> dict:
        """Explain a design: why it placed fewer pairs than asked, how its windows
        sit against the GC/Tm window, whether its pairs were screened, how folded
        its primaries are. Each fact is a sentence worked out by the software --
        quote it rather than recomputing it. When pairs are missing, what_if
        previews smaller spacings and fewer pairs. Changes nothing."""
        return refusing(session.diagnose_design, design_id)

    @server.tool(annotations=read_only)
    def try_change(design_id: str, n_pairs: int | None = None,
                   min_spacing: int | None = None) -> dict:
        """Preview how many pairs a design would place with a different n_pairs
        and/or min_spacing (nt), without designing it. Placement only: the screen
        and folding check run when you design it with design_target. Not for
        fusions. Changes nothing."""
        return refusing(session.try_change, design_id, n_pairs=n_pairs,
                        min_spacing=min_spacing)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                             idempotentHint=False, openWorldHint=False))
    def write_order(design_ids: list[str]) -> dict:
        """Write one order workbook for the given design_ids (from design_target,
        in this conversation or an earlier one: designs survive a server restart).
        Refused for any design whose state is not 'succeeded'. The server names
        the file and saves it in the workbook directory shown by status; the path
        is returned."""
        return refusing(session.write_order, design_ids)

    return server


def main() -> None:
    configure_environment()
    build_server().run("stdio")
