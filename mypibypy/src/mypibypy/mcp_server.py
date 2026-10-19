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
from .agent.actions import DesignParameters, ParameterRefused, clamp
from .agent.tools import FORBIDDEN_TOOL_PARAMETERS
from .cli import design_warnings, example_warnings
from .design import DEFAULT_MIN_SPACING_NT, PADS_DIRECT
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

def pair_summary(index: int, pair, screen, access) -> dict:
    """One ordered pair, as a client sees it. Deliberately has no sequence field."""
    cand = pair.candidate
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
    }


def result_summary(res) -> dict:
    return {
        "target": res.locus.name,
        "gene": res.locus.gene,
        "system": res.system.name,
        "fluorophore": res.system.fluorophore,
        "repeat_hits_masked": len(res.rmsk),
        "n_pairs": res.n_pairs,
        "pairs": [pair_summary(i, p, s, a)
                  for i, (p, s, a) in enumerate(res.rows(), start=1)],
    }


# --- the session -------------------------------------------------------------

@dataclass
class StoredDesign:
    design_id: str
    target_id: str
    results: list
    warnings: list[str]
    requested_pairs: int
    run_id: str


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
            "workbook_directory": str(out_dir()),
            "run_record": str(rec.resolve_store_path()),
            "designs_this_session": sorted(self.designs),
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

        try:
            results = self._run(target, params, screen, pad_assignment)
        except Exception as exc:
            record.log(rec.ERROR, 0, f"design failed: {exc}")
            self._finish(record, "failed")
            raise

        warnings = _example_warnings() + design_warnings(results, params.n_pairs)
        warnings += [str(c) for c in clamps]
        design_id = f"d_{uuid.uuid4().hex[:8]}"
        record.log(rec.TOOL_CALL, 0, f"designed {target_id} as {design_id}",
                   pairs=[p.p1l_name for r in results for p in r.pairs],
                   screens=[s.summary() for r in results for s in r.pair_screens])
        self._finish(record, "succeeded")
        with self.lock:
            self.designs[design_id] = StoredDesign(
                design_id, target_id, results, warnings, params.n_pairs,
                record.run_id)
        out = {"design_id": design_id, "target_id": target_id, "kind": target.kind,
               "run_id": record.run_id,
               "results": [result_summary(r) for r in results]}
        if warnings:
            out["warnings"] = warnings
        return out

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
        out = {"workbook": str(path), "designs": [d.design_id for d in chosen],
               "primaries": sum(2 * r.n_pairs for r in results)}
        if warnings:
            out["warnings"] = warnings
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
                               n_pairs=params.n_pairs, min_spacing=params.min_spacing)]
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
that `write_order` saves locally. Report any "NOT SCREENED" specificity or \
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
    def write_order(design_ids: list[str]) -> dict:
        """Write one order workbook for the given design_ids (from design_target
        in this session). The server names the file and saves it in the
        workbook directory shown by status; the path is returned."""
        return session.write_order(design_ids)

    return server


def main() -> None:
    configure_environment()
    build_server().run("stdio")
