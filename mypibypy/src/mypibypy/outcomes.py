"""An on-disk record linking each ordered probe pair to what it did at the scope.

Why this is so thin, on purpose: there are three designs. Three. Nothing can be
fitted to three outcomes, and a scoring function built on them would encode this
author's guesses while looking like evidence. So this module records and
tabulates, and that is all. It computes no quality score, fits no threshold and
makes no prediction. When there are tens of records, the data to calibrate
`crosshyb` and `specificity` thresholds against will be sitting here in a form
that can actually be regressed; until then the honest move is to write things
down.

What it links:

    design time                        ->  bench/imaging
    sequences, locus, system,              zFISHer puncta counts, SNR,
    QC metrics, crosshyb + specificity     and a human call

Imaging numbers come from zFISHer (``/Users/sstaller/scripts_ss/zFISHer``),
which writes ``reports/<layer>_puncta.csv`` with columns
``puncta_id, Z, Y, X, Nucleus_ID, Intensity, SNR, Source``. This module reads
that file and nothing else -- it does not import zfisher, does not write
anything into a zFISHer session, and treats those folders as read-only.

THE STORE CONTAINS REAL PROBE SEQUENCES, so it belongs outside version control.
The default filename is ``outcomes.local.jsonl``, which is gitignored, and the
format is one JSON object per line so that appending a record never rewrites
earlier ones.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOCAL_FILENAME = "outcomes.local.jsonl"
ENV_VAR = "MYPIBYPY_OUTCOMES"

#: The human call on a design. Deliberately coarse: with this little data,
#: a coarse label that someone will actually fill in beats a 1-5 scale nobody
#: applies consistently.
CALLS = ("signal", "weak", "no_signal", "ambiguous", "not_imaged")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DesignFacts:
    """What was known at design time, flattened for storage.

    Flat and primitive by intention. This file will be read years from now by
    something that is not this code, and a nested schema would make that worse.
    """

    design_id: str
    gene: str = ""
    locus_name: str = ""
    interval: str = ""
    system: str = ""
    fluorophore: str = ""
    p1l_name: str = ""
    p1r_name: str = ""
    p1l: str = ""
    p1r: str = ""
    left_pad: str = ""
    right_pad: str = ""
    gap_base: str = ""
    left_gc: float | None = None
    right_gc: float | None = None
    left_tm: float | None = None
    right_tm: float | None = None
    penalty: float | None = None
    qc_flags: list[str] = field(default_factory=list)
    rmsk_hits: int | None = None
    # Screens, stored as their own verdicts rather than re-derived later.
    specificity_status: str = ""
    specificity_verdict: str = ""
    specificity_summary: str = ""
    crosshyb_summary: str = ""
    crosshyb_worst_bp: int | None = None
    # Continuous numbers to calibrate against later. None when not computed.
    on_target_fraction_bound: float | None = None
    worst_offtarget_fraction_bound: float | None = None
    expression_weighted: bool = False
    probe_fold_min_dg: float | None = None
    probe_fold_worst_paired: float | None = None
    is_fusion: bool = False
    arms_meet_at_junction: bool | None = None
    arm_separation_nt: int | None = None
    notes: str = ""


@dataclass
class ImagingResult:
    """What the scope said, summarized from a zFISHer puncta table."""

    source: str = ""              # path to the zFISHer CSV this came from
    recorded_at: str = ""
    n_puncta: int | None = None
    n_nuclei: int | None = None
    puncta_per_nucleus: float | None = None
    median_snr: float | None = None
    median_intensity: float | None = None
    call: str = "not_imaged"
    notes: str = ""


@dataclass
class Outcome:
    design: DesignFacts
    imaging: ImagingResult = field(default_factory=ImagingResult)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_json(self) -> str:
        return json.dumps({"design": asdict(self.design),
                           "imaging": asdict(self.imaging),
                           "created_at": self.created_at,
                           "updated_at": self.updated_at}, sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> Outcome:
        design = DesignFacts(**d.get("design", {}))
        imaging = ImagingResult(**d.get("imaging", {}))
        return Outcome(design=design, imaging=imaging,
                       created_at=d.get("created_at", ""),
                       updated_at=d.get("updated_at", ""))


# --- Building a record from a design ---------------------------------------

def _number(x: float) -> float | None:
    """NaN means "not computed"; store it as None so JSON round-trips it."""
    return None if x is None or x != x else float(x)


def facts_from_result(result, design_id: str | None = None,
                      crosshyb_report=None) -> DesignFacts:
    """Flatten a `pipeline.DesignResult` into storable facts.

    `crosshyb_report` is optional because the cross-hybridization screen is a
    separate step; when it is passed, its summary is stored alongside.
    """
    cand, pair, locus = result.best, result.pair, result.locus
    spec = result.specificity
    worst = crosshyb_report.worst() if crosshyb_report is not None else None
    return DesignFacts(
        design_id=design_id or f"{locus.gene}_{locus.name}",
        gene=locus.gene, locus_name=locus.name, interval=str(cand.interval),
        system=result.system.name, fluorophore=result.system.fluorophore,
        p1l_name=pair.p1l_name, p1r_name=pair.p1r_name,
        p1l=pair.p1l, p1r=pair.p1r,
        left_pad=cand.left.seq, right_pad=cand.right.seq, gap_base=cand.gap_base,
        left_gc=cand.left.gc, right_gc=cand.right.gc,
        left_tm=cand.left.tm, right_tm=cand.right.tm,
        penalty=cand.penalty, qc_flags=list(cand.flags),
        rmsk_hits=len(result.rmsk),
        specificity_status=spec.status, specificity_verdict=spec.verdict,
        specificity_summary=spec.summary(),
        crosshyb_summary=crosshyb_report.summary() if crosshyb_report is not None else "",
        crosshyb_worst_bp=worst.duplex.length if worst is not None else None,
        on_target_fraction_bound=_number(spec.on_target_fraction),
        worst_offtarget_fraction_bound=_number(spec.worst_offtarget_fraction),
        expression_weighted=spec.expression_evaluated,
        probe_fold_min_dg=result.folding.min_dg,
        probe_fold_worst_paired=result.folding.worst_paired,
        is_fusion="junction" in str(cand.interval),
    )


# --- Reading zFISHer output ------------------------------------------------

def read_zfisher_puncta(path: str | Path, call: str = "not_imaged",
                        notes: str = "") -> ImagingResult:
    """Summarize one zFISHer puncta CSV.

    Expects zFISHer's own ``PUNCTA_CSV_COLUMNS``
    (``puncta_id, Z, Y, X, Nucleus_ID, Intensity, SNR, Source``) and tolerates
    extra or missing optional columns, because that file is produced by another
    project on its own release schedule. Reading is stdlib-only: mypibypy does
    not depend on pandas, and a summary of a few thousand rows does not need it.

    Only a handful of numbers are kept. A richer summary would be guessing at
    what will turn out to matter.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no zFISHer puncta table at {p}")
    if call not in CALLS:
        raise ValueError(f"call must be one of {CALLS}, got {call!r}")
    snrs: list[float] = []
    intensities: list[float] = []
    nuclei: set[str] = set()
    n = 0
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            for key, bucket in (("SNR", snrs), ("Intensity", intensities)):
                raw = (row.get(key) or "").strip()
                if raw:
                    try:
                        bucket.append(float(raw))
                    except ValueError:
                        pass
            nid = (row.get("Nucleus_ID") or "").strip()
            # zFISHer marks puncta outside any nucleus with 0 / -1 / blank.
            if nid and nid not in {"0", "-1", "nan"}:
                nuclei.add(nid)
    return ImagingResult(
        source=str(p), recorded_at=_now(), n_puncta=n,
        n_nuclei=len(nuclei) or None,
        puncta_per_nucleus=(n / len(nuclei)) if nuclei else None,
        median_snr=statistics.median(snrs) if snrs else None,
        median_intensity=statistics.median(intensities) if intensities else None,
        call=call, notes=notes,
    )


# --- The store -------------------------------------------------------------

def resolve_store_path(path: str | Path | None = None) -> Path:
    """Where outcomes are stored: explicit path, then $MYPIBYPY_OUTCOMES, then cwd."""
    if path:
        return Path(path).expanduser()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.cwd() / LOCAL_FILENAME


class OutcomeStore:
    """Append-only JSONL of design-to-result records.

    One JSON object per line: appending never rewrites what is already there,
    and a corrupt line costs one record rather than the file. There is no
    database here and should not be -- the whole point is that this survives
    being read by hand in five years.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = resolve_store_path(path)

    # -- reading ----------------------------------------------------------
    def all(self) -> list[Outcome]:
        if not self.path.is_file():
            return []
        out: list[Outcome] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                out.append(Outcome.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{self.path}: unreadable record: {exc}") from exc
        # Later records for the same design_id supersede earlier ones.
        latest: dict[str, Outcome] = {}
        for rec in out:
            latest[rec.design.design_id] = rec
        return list(latest.values())

    def get(self, design_id: str) -> Outcome | None:
        for rec in self.all():
            if rec.design.design_id == design_id:
                return rec
        return None

    def history(self, design_id: str) -> list[Outcome]:
        """Every version of one record, oldest first. Nothing is ever lost."""
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            rec = Outcome.from_dict(json.loads(line))
            if rec.design.design_id == design_id:
                out.append(rec)
        return out

    # -- writing ----------------------------------------------------------
    def _append(self, rec: Outcome) -> Outcome:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(rec.to_json() + "\n")
        return rec

    def add(self, design: DesignFacts, imaging: ImagingResult | None = None) -> Outcome:
        return self._append(Outcome(design=design,
                                    imaging=imaging or ImagingResult()))

    def record_design(self, result, design_id: str | None = None,
                      crosshyb_report=None) -> Outcome:
        """Record a `pipeline.DesignResult` at design time, before any imaging."""
        return self.add(facts_from_result(result, design_id, crosshyb_report))

    def attach_imaging(self, design_id: str, imaging: ImagingResult) -> Outcome:
        """Attach or replace the imaging result for an existing design."""
        existing = self.get(design_id)
        if existing is None:
            raise KeyError(
                f"no design {design_id!r} in {self.path}; record the design first "
                f"(known: {sorted(r.design.design_id for r in self.all())})")
        return self._append(Outcome(design=existing.design, imaging=imaging,
                                    created_at=existing.created_at,
                                    updated_at=_now()))

    def attach_zfisher_report(self, design_id: str, csv_path: str | Path,
                              call: str = "not_imaged", notes: str = "") -> Outcome:
        """Read a zFISHer puncta CSV and attach it to a recorded design."""
        return self.attach_imaging(
            design_id, read_zfisher_puncta(csv_path, call=call, notes=notes))

    # -- looking at it ----------------------------------------------------
    def table(self) -> list[dict]:
        """Flat rows, design next to result. For eyeballing and for export."""
        rows = []
        for rec in self.all():
            d, i = rec.design, rec.imaging
            rows.append({
                "design_id": d.design_id, "gene": d.gene, "system": d.system,
                "fluorophore": d.fluorophore,
                "specificity": d.specificity_verdict or d.specificity_status,
                "crosshyb_worst_bp": d.crosshyb_worst_bp,
                "qc_flags": len(d.qc_flags),
                "call": i.call, "n_puncta": i.n_puncta,
                "puncta_per_nucleus": i.puncta_per_nucleus,
                "median_snr": i.median_snr,
            })
        return sorted(rows, key=lambda r: r["design_id"])

    def summary(self) -> str:
        """A deliberately boring count. No scores, no thresholds, no inference."""
        rows = self.table()
        if not rows:
            return f"{self.path}: no records yet"
        by_call: dict[str, int] = {}
        for r in rows:
            by_call[r["call"]] = by_call.get(r["call"], 0) + 1
        imaged = sum(n for c, n in by_call.items() if c != "not_imaged")
        parts = ", ".join(f"{c}={n}" for c, n in sorted(by_call.items()))
        text = f"{len(rows)} design(s) recorded, {imaged} imaged ({parts})"
        if imaged < 10:
            text += (
                f"\n{imaged} imaged design(s) is far too few to calibrate any "
                f"threshold against. This store exists so that the comparison "
                f"becomes possible later; it does not score anything now.")
        return text

    def write_csv(self, path: str | Path) -> Path:
        """Export `table()` as CSV. Contains sequences' metadata, not sequences."""
        rows = self.table()
        out = Path(path)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(rows[0]) if rows else ["design_id"])
            writer.writeheader()
            writer.writerows(rows)
        return out
