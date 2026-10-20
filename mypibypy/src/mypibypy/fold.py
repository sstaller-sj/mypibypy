"""Does the probe oligo fold on itself?

A different question from `accessibility`, which asks whether the *target site*
on a transcript is free. This one asks whether the synthesized P1L / P1R is
free: a primary whose pad folds back onto its own arm cannot hybridize to the
target, and one whose arm is tied up cannot present half of P2's landing site.

`crosshyb.best_duplex` already flags self-complementary runs, which catches
the gross cases. This adds the folding engine's view: the minimum-free-energy
structure of the whole oligo, with loops priced properly, and how much of the
pad and of the arm it leaves paired.

HOW IT IS RUN. ViennaRNA's ``RNAfold`` with DNA parameters (``-P DNA``, the
Mathews 2004 set; older releases fall back to the ``dna_mathews2004.par`` file
shipped beside the binary), at the primary-hyb salt and temperature. RNAfold
has no formamide term, so formamide enters the same way it does everywhere else
in this package -- as an effective temperature 0.65 C higher per % -- which is a
first-order correction derived for duplexes, applied here to hairpins.

WHAT IS REPORTED. Per oligo: the mean probability that a pad base, and that an
arm base, is paired -- from the partition function, over the whole ensemble --
with the MFE dG and structure alongside as context. The pair takes the worse
oligo.

WHY THE ENSEMBLE AND NOT THE MFE STRUCTURE. Counting paired bases in the MFE
structure is the obvious metric and it misleads on exactly the oligos that
matter. A 47-nt primary with a weak -2.2 kcal/mol MFE at hyb conditions had
that structure at 2% of the ensemble (diversity 15.5): the MFE called 60% of
its pad paired when no single structure is populated. A designed 10-bp hairpin,
by contrast, is 85% MFE and the two metrics agree. The same reasoning is why
`accessibility` scores ensemble unpaired probabilities rather than one fold. It is a **reported attribute and never a reject**, and the cutoffs that
put a word next to a number are not thresholds: nothing ties a piFISH failure
to a measured probe fold, so `FOLDING_IS_UNCALIBRATED` travels with the number.

Absent RNAfold the result is ``NOT_EVALUATED`` with a reason, exactly as the
screens degrade.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import thermo

EVALUATED = "EVALUATED"
NOT_EVALUATED = "NOT_EVALUATED"

FOLDING_IS_UNCALIBRATED = (
    "probe-folding cutoffs are UNCALIBRATED: no piFISH failure has been tied to "
    "a measured probe fold, so treat these as a reported attribute, not a "
    "pass/fail criterion"
)

#: Only used to put a word next to a number in reports. NOT thresholds.
#:
#: Anchored to background, because at 6x SSC almost everything pairs a little:
#: over 300 random 47-mers at primary-hyb conditions the mean per-base pairing
#: probability was median 0.39, p90 0.48, p99 0.58 -- transient pairs a target
#: displaces easily. So "open" runs to 0.50, about the random p90, and "folded"
#: starts at 0.65; a designed 10-bp hairpin's pad sits near 0.80. Re-derive
#: these if the hyb conditions change.
_BANDS = ((0.50, "open"), (0.65, "partly folded"))


def describe_folding(paired_fraction: float) -> str:
    """A word for a paired fraction, for reports. Explicitly not a cutoff."""
    for ceiling, label in _BANDS:
        if paired_fraction < ceiling:
            return label
    return "folded"


def rnafold_path() -> str | None:
    """Where RNAfold is, honouring ``$MYPIBYPY_RNAFOLD`` first."""
    override = os.environ.get("MYPIBYPY_RNAFOLD")
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("RNAfold")


def _dna_param_args(exe: str) -> list[list[str]]:
    """Ways to ask for DNA parameters, newest first."""
    shipped = Path(exe).resolve().parent.parent / "share" / "ViennaRNA" / "dna_mathews2004.par"
    options = [["-P", "DNA"]]
    if shipped.is_file():
        options.append(["-P", str(shipped)])
    return options


@dataclass(frozen=True)
class OligoFold:
    """Folding of one synthesized primary: ensemble pairing, plus its MFE."""

    name: str
    seq: str
    mfe_dg: float           # kcal/mol; 0.0 means no structure at all
    structure: str          # MFE dot-bracket, context only
    pad: tuple[int, int]    # 0-based half-open range of the target-binding pad
    arm: tuple[int, int]    # 0-based half-open range of the constant arm
    #: P(base i is paired) over the ensemble. Empty -> fall back to the MFE
    #: structure (hand-built results, or output parsed without dot plots).
    paired_prob: tuple[float, ...] = ()
    mfe_frequency: float = float("nan")

    def _paired(self, span: tuple[int, int]) -> float:
        a, b = span
        if b <= a:
            return 0.0
        if self.paired_prob:
            return sum(self.paired_prob[a:b]) / (b - a)
        return sum(1 for c in self.structure[a:b] if c != ".") / (b - a)

    @property
    def pad_paired(self) -> float:
        return self._paired(self.pad)

    @property
    def arm_paired(self) -> float:
        return self._paired(self.arm)

    @property
    def worst_paired(self) -> float:
        return max(self.pad_paired, self.arm_paired)

    @property
    def label(self) -> str:
        return describe_folding(self.worst_paired)

    def __str__(self) -> str:
        return (f"{self.name}: pad {self.pad_paired:.0%} / arm {self.arm_paired:.0%} "
                f"paired over the ensemble ({self.label}); MFE {self.mfe_dg:+.1f} "
                f"kcal/mol")


@dataclass
class PairFolding:
    """Self-folding of both primaries, or an explicit statement it was not run."""

    status: str
    reason: str = ""
    p1l: OligoFold | None = None
    p1r: OligoFold | None = None
    tool: str = ""
    temp_c: float = float("nan")      # effective, formamide folded in
    salt_m: float = float("nan")
    caveat: str = FOLDING_IS_UNCALIBRATED

    @property
    def evaluated(self) -> bool:
        return self.status == EVALUATED

    @property
    def limiting(self) -> OligoFold | None:
        """The more folded of the two primaries."""
        if not self.evaluated:
            return None
        return max((self.p1l, self.p1r), key=lambda o: (o.worst_paired, -o.mfe_dg))

    @property
    def worst_paired(self) -> float | None:
        lim = self.limiting
        return lim.worst_paired if lim else None

    @property
    def min_dg(self) -> float | None:
        if not self.evaluated:
            return None
        return min(self.p1l.mfe_dg, self.p1r.mfe_dg)

    def summary(self) -> str:
        if not self.evaluated:
            return f"NOT EVALUATED -- {self.reason}"
        lim = self.limiting
        return (f"{lim.label.upper()} -- {self.p1l}; {self.p1r}. {self.tool} at "
                f"{self.temp_c:.1f} C effective, {self.salt_m:g} M Na+. {self.caveat}")

    def to_dict(self) -> dict:
        d = {"status": self.status, "evaluated": self.evaluated, "reason": self.reason,
             "tool": self.tool, "temp_c": self.temp_c, "salt_m": self.salt_m}
        for key in ("p1l", "p1r"):
            o = getattr(self, key)
            d[key] = None if o is None else {
                "name": o.name, "mfe_dg": o.mfe_dg, "structure": o.structure,
                "pad_paired": o.pad_paired, "arm_paired": o.arm_paired, "label": o.label,
                "mfe_frequency": o.mfe_frequency}
        d["summary"] = self.summary()
        return d


def not_evaluated(reason: str) -> PairFolding:
    return PairFolding(status=NOT_EVALUATED, reason=reason)


def parse_rnafold(text: str) -> list[tuple[str, float]]:
    """(MFE structure, MFE) per record from RNAfold stdout.

    With ``-p`` each record also carries ensemble, centroid and MEA lines in
    other brackets; only the first round-bracketed line per record is the MFE.
    """
    out, want = [], False
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith(">"):
            want = True
            continue
        if not want or not line or line[0] not in ".()" or not line.endswith(")"):
            continue
        struct, _, energy = line.rpartition(" (")
        try:
            out.append((struct.strip(), float(energy.strip(" ()"))))
            want = False
        except ValueError:
            continue
    return out


def parse_mfe_frequencies(text: str) -> list[float]:
    """"frequency of mfe structure in ensemble" per record, from ``-p`` stdout."""
    out = []
    for line in text.splitlines():
        if "frequency of mfe structure in ensemble" in line:
            try:
                out.append(float(line.split("ensemble")[1].split(";")[0]))
            except (IndexError, ValueError):
                out.append(float("nan"))
    return out


def parse_dot_plot(text: str, length: int) -> tuple[float, ...]:
    """Per-base pairing probability from an RNAfold ``*_dp.ps`` file.

    The file lists each pair as ``i j sqrt(p) ubox`` (1-based); a base's
    pairing probability is the sum of p over its partners.
    """
    probs = [0.0] * length
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[3] == "ubox":
            try:
                i, j, sq = int(parts[0]), int(parts[1]), float(parts[2])
            except ValueError:
                continue
            if 1 <= i <= length and 1 <= j <= length:
                probs[i - 1] += sq * sq
                probs[j - 1] += sq * sq
    return tuple(min(1.0, p) for p in probs)


@dataclass(frozen=True)
class FoldResult:
    structure: str
    mfe_dg: float
    paired_prob: tuple[float, ...]
    mfe_frequency: float


def fold_oligos(seqs: list[str], cond: thermo.Conditions = thermo.HYB_PRIMARY,
                exe: str | None = None) -> tuple[list[FoldResult], str]:
    """Fold each sequence (MFE and partition function), and describe the call.

    Raises RuntimeError when RNAfold cannot be run with DNA parameters.
    """
    exe = exe or rnafold_path()
    if not exe:
        raise RuntimeError("RNAfold not found")
    temp = cond.temp_c + thermo.FORMAMIDE_C_PER_PCT * cond.formamide_pct
    stdin = "".join(f">o{i}\n{s.upper()}\n" for i, s in enumerate(seqs))
    errors = []
    for params in _dna_param_args(exe):
        # -p writes one dot plot per record into the working directory, named
        # after the FASTA header -- hence a private directory per call.
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [exe, "-p", "--noPS", "-T", f"{temp:.2f}",
                   "--salt", f"{cond.na_molar:.4f}", *params]
            proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                                  check=False, cwd=tmp)
            mfes = parse_rnafold(proc.stdout)
            if proc.returncode != 0 or len(mfes) != len(seqs):
                errors.append(proc.stderr.strip().splitlines()[-1:]
                              or [f"exit {proc.returncode}"])
                continue
            freqs = parse_mfe_frequencies(proc.stdout)
            results = []
            for i, (seq, (struct, dg)) in enumerate(zip(seqs, mfes)):
                dp = Path(tmp) / f"o{i}_dp.ps"
                probs = parse_dot_plot(dp.read_text(), len(seq)) if dp.is_file() else ()
                results.append(FoldResult(struct, dg, probs,
                                          freqs[i] if i < len(freqs) else float("nan")))
        return results, f"RNAfold -p {' '.join(params)} (DNA, Mathews 2004)"
    raise RuntimeError(f"RNAfold could not fold with DNA parameters: {errors}")


def pair_folding(pair, cond: thermo.Conditions = thermo.HYB_PRIMARY) -> PairFolding:
    """Fold P1L and P1R of a `design.ProbePair`. Never raises for a missing tool."""
    exe = rnafold_path()
    if not exe:
        return not_evaluated(
            "RNAfold not found. Install ViennaRNA (conda install -c conda-forge "
            "-c bioconda viennarna) or set $MYPIBYPY_RNAFOLD. No probe-folding "
            "check was run")
    sysm = pair.system
    n_al, n_ar = len(sysm.arm_left), len(sysm.arm_right)
    try:
        (fl, fr), tool = fold_oligos([pair.p1l, pair.p1r], cond, exe)
    except (RuntimeError, OSError, ValueError) as exc:
        return not_evaluated(str(exc))
    p1l = OligoFold(pair.p1l_name, pair.p1l, fl.mfe_dg, fl.structure,
                    pad=(n_al, len(pair.p1l)), arm=(0, n_al),
                    paired_prob=fl.paired_prob, mfe_frequency=fl.mfe_frequency)
    p1r = OligoFold(pair.p1r_name, pair.p1r, fr.mfe_dg, fr.structure,
                    pad=(0, len(pair.p1r) - n_ar), arm=(len(pair.p1r) - n_ar, len(pair.p1r)),
                    paired_prob=fr.paired_prob, mfe_frequency=fr.mfe_frequency)
    return PairFolding(status=EVALUATED, p1l=p1l, p1r=p1r, tool=tool,
                       temp_c=cond.temp_c + thermo.FORMAMIDE_C_PER_PCT * cond.formamide_pct,
                       salt_m=cond.na_molar)
