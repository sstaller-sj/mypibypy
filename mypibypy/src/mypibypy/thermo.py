"""Buffer-aware duplex thermodynamics.

`sequence.nn_tm` is a *fixed-reference* Tm (0.33 M Na+, no formamide) used to rank
candidate windows against each other; it is deliberately left alone so rankings
stay comparable with earlier designs. This module answers a different question:

    at the bench, in this buffer, at this temperature -- does the duplex hold?

Everything here is reported as a **margin**: Tm minus the step temperature.
A positive margin means the duplex is expected to survive that step; negative
means it should come apart. That framing matters for off-target screening, where
"the aligner found it" is not the same as "it survives the wash".

Formamide correction: linear, 0.65 C per % (v/v). The widely used range is
0.6-0.75 C/% for DNA duplexes (McConaughy et al. 1969, Blake & Delcourt 1996);
0.65 is mid-range. It is a first-order correction, not an exact model.

Salt: SantaLucia's log[Na+] entropy correction. 1x SSC is taken as 0.195 M Na+
(0.15 M NaCl contributes 0.15; 0.015 M trisodium citrate contributes 0.045).

MISMATCHES, TWO WAYS. When the off-target sequence is known, `site_thermo`
walks the actual mismatched duplex with published nearest-neighbour parameters
for internal mismatches (Allawi & SantaLucia 1997-1998, Peyret et al. 1999) and
terminal mismatches (Bommarito et al. 2000), taken from Biopython's tables rather
than retyped. Position and identity then matter as they do at the bench: a
terminal G-T barely costs anything, a central C-C costs a great deal. When only
a mismatch *count* is known, `tm_mismatched` falls back to a flat penalty per
mismatch -- see there for why that is an approximation.

FRACTION BOUND. `fraction_bound` turns the same dH/dS into the equilibrium
fraction of a site occupied by probe in excess, theta = K*C / (1 + K*C). It puts
on-target and off-target sites on one continuous scale. Two conventions to know:

* `tm_buffer` uses Ct/4, as `sequence.nn_tm` does, so at the reported Tm (margin
  0) a probe-in-excess site is ~80% occupied, not 50%. Margins and fractions are
  therefore consistent but not interchangeable.
* The washes carry no probe, so an equilibrium at `probe_nM` there is a ranking
  device, not a model of dissociation kinetics. The same was already true of the
  Tm margins, which use the same concentration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from Bio.SeqUtils import MeltingTemp as _mt

from .sequence import _NN_DH, _NN_DS, _R, gc_fraction

#: Na+ molarity contributed by 1x SSC.
SSC_NA = 0.195

#: Formamide depression, degrees C per % (v/v).
FORMAMIDE_C_PER_PCT = 0.65

#: Default conservative penalty per mismatch, in degrees C. See `tm_mismatched`.
DEFAULT_MISMATCH_PENALTY_C = 8.0


@dataclass(frozen=True)
class Conditions:
    """One hybridization or wash step."""

    name: str
    temp_c: float
    formamide_pct: float
    ssc_x: float
    probe_nM: float = 10.0

    @property
    def na_molar(self) -> float:
        return self.ssc_x * SSC_NA

    def __str__(self) -> str:
        return (f"{self.name} ({self.temp_c:g} C, {self.formamide_pct:g}% formamide, "
                f"{self.ssc_x:g}x SSC)")


# --- The lab's piFISH conditions -------------------------------------------
# Primary (P1) hybridization. Matches the published pi-FISH rainbow protocol
# (Tao et al., Nat Commun 2023): solution A, 25% formamide / 6x SSC, 40 C.
HYB_PRIMARY = Conditions("primary hyb", temp_c=40.0, formamide_pct=25.0, ssc_x=6.0, probe_nM=10.0)

# Wash series, also from the published protocol (elution buffers A/B/C, 40 C).
# NOTE: transcribed from the paper, NOT confirmed against this lab's bench
# protocol. Override with your own if they differ.
WASH_A = Conditions("wash A", temp_c=40.0, formamide_pct=20.0, ssc_x=2.0, probe_nM=10.0)
WASH_B = Conditions("wash B", temp_c=40.0, formamide_pct=10.0, ssc_x=1.0, probe_nM=10.0)
WASH_C = Conditions("wash C", temp_c=40.0, formamide_pct=0.0, ssc_x=0.1, probe_nM=10.0)

WASH_SERIES = (WASH_A, WASH_B, WASH_C)

#: The step a probe must survive to stay bound: the most stringent wash.
#: Stringency here is dominated by the formamide/salt combination, so the
#: limiting step is computed rather than assumed -- see `most_stringent`.
ALL_STEPS = (HYB_PRIMARY, *WASH_SERIES)


def nn_thermo(seq: str) -> tuple[float, float]:
    """(dH kcal/mol, dS cal/mol/K) at 1 M Na+ for a perfect DNA/DNA duplex."""
    s = seq.upper()
    if len(s) < 2 or any(b not in "ACGT" for b in s):
        return float("nan"), float("nan")
    dh = ds = 0.0
    for end in (s[0], s[-1]):
        if end in "GC":
            dh += 0.1
            ds += -2.8
        else:
            dh += 2.3
            ds += 4.1
    for i in range(len(s) - 1):
        dh += _NN_DH[s[i : i + 2]]
        ds += _NN_DS[s[i : i + 2]]
    return dh, ds


def tm_buffer(seq: str, cond: Conditions) -> float:
    """Melting temperature (C) of a perfect duplex under `cond`.

    Probe in excess over target (the FISH case), so Ct/4 is used as in
    `sequence.nn_tm`.
    """
    dh, ds = nn_thermo(seq)
    if math.isnan(dh):
        return float("nan")
    ds += 0.368 * (len(seq) - 1) * math.log(cond.na_molar)
    tm_k = (dh * 1000.0) / (ds + _R * math.log(cond.probe_nM * 1e-9 / 4.0))
    return tm_k - 273.15 - FORMAMIDE_C_PER_PCT * cond.formamide_pct


def margin(seq: str, cond: Conditions) -> float:
    """Tm - step temperature. Positive = expected to stay hybridized."""
    return tm_buffer(seq, cond) - cond.temp_c


def tm_mismatched(seq: str, n_mismatches: int, cond: Conditions,
                  penalty_c: float = DEFAULT_MISMATCH_PENALTY_C) -> float:
    """Approximate Tm of a duplex carrying `n_mismatches`.

    APPROXIMATION. A proper treatment needs the mismatch identities, their
    positions and internal-loop NN parameters, which a seed-and-extend aligner
    does not give us. Published single-internal-mismatch penalties for 20-25mers
    span roughly 4-12 C depending on identity and position; `penalty_c` defaults
    to 8 C, mid-range and deliberately on the destabilizing side so that
    off-targets are not dismissed too readily.

    Use this to *rank* and to *triage* off-target sites, never as a measured Tm.
    Terminal mismatches are far less destabilizing than internal ones, so a site
    whose mismatches sit at the very ends will be under-estimated here.
    """
    if n_mismatches < 0:
        raise ValueError("n_mismatches must be >= 0")
    return tm_buffer(seq, cond) - penalty_c * n_mismatches


# --- Mismatched duplexes from sequence -------------------------------------
#
# The duplex is written the way Biopython writes it: `top` 5'->3' is the site
# actually present, `bottom` 3'->5' beneath it is the probe. The probe is the
# reverse complement of the designed footprint, so read 3'->5' it is simply the
# complement of the footprint -- position i of `bottom` pairs with position i of
# `top`, and a mismatch is wherever the site differs from the footprint.

#: Internal single mismatches: Allawi & SantaLucia 1997-1998, Peyret et al. 1999.
_IMM = _mt.DNA_IMM1
#: Terminal mismatches: Bommarito et al. 2000.
_TMM = _mt.DNA_TMM1

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

#: Which model produced a site's Tm. Recorded per site so a report can say so.
MODEL_NN_MISMATCH = "nearest-neighbour mismatch"
MODEL_PERFECT = "perfect duplex"
MODEL_COUNT_PENALTY = "flat per-mismatch penalty"


def _nn_step(top2: str, bottom2: str) -> tuple[float, float] | None:
    """dH/dS for one stacked step, matched or carrying one mismatch."""
    if top2.translate(_COMPLEMENT) == bottom2:
        return _NN_DH[top2], _NN_DS[top2]
    key = f"{top2}/{bottom2}"
    for k in (key, key[::-1]):
        if k in _IMM:
            return _IMM[k]
    return None


def site_thermo(seq: str, site: str) -> tuple[float, float, int] | None:
    """(dH kcal/mol, dS cal/mol/K at 1 M Na+, paired length) for probe on `site`.

    `seq` is the footprint the probe was designed against, `site` the sequence
    actually present, both in the same orientation and of equal length. Returns
    None when the duplex is not covered by the parameters -- two mismatches on
    adjacent steps, a non-ACGT base -- so the caller can fall back to the
    count model and say it did.

    Frayed ends are trimmed: where the outermost two or more positions are all
    mismatched, the bases beyond the last terminal mismatch cannot pair and are
    dropped, leaving one terminal mismatch to be priced with the terminal table.
    """
    top, ref = site.upper(), seq.upper()
    if len(top) != len(ref) or len(top) < 2 or set(top + ref) - set("ACGT"):
        return None
    if top == ref:
        dh, ds = nn_thermo(top)
        return dh, ds, len(top)
    bottom = ref.translate(_COMPLEMENT)
    ok = [a == b for a, b in zip(top, ref)]

    # Fray: keep at most one unpaired base at each end.
    lo, hi = 0, len(top)
    while hi - lo > 2 and not ok[lo] and not ok[lo + 1]:
        lo += 1
    while hi - lo > 2 and not ok[hi - 1] and not ok[hi - 2]:
        hi -= 1
    top, bottom, ok = top[lo:hi], bottom[lo:hi], ok[lo:hi]

    dh = ds = 0.0
    if not ok[0]:
        k = f"{bottom[:2][::-1]}/{top[:2][::-1]}"
        if k not in _TMM:
            return None
        dh, ds = dh + _TMM[k][0], ds + _TMM[k][1]
        top, bottom, ok = top[1:], bottom[1:], ok[1:]
    if not ok[-1]:
        k = f"{top[-2:]}/{bottom[-2:]}"
        if k not in _TMM:
            return None
        dh, ds = dh + _TMM[k][0], ds + _TMM[k][1]
        top, bottom, ok = top[:-1], bottom[:-1], ok[:-1]
    if len(top) < 2 or not ok[0] or not ok[-1]:
        return None

    # Initiation on the terminal base pairs that actually form, as in nn_thermo.
    for end in (top[0], top[-1]):
        if end in "GC":
            dh, ds = dh + 0.1, ds - 2.8
        else:
            dh, ds = dh + 2.3, ds + 4.1
    for i in range(len(top) - 1):
        step = _nn_step(top[i : i + 2], bottom[i : i + 2])
        if step is None:
            return None
        dh, ds = dh + step[0], ds + step[1]
    return dh, ds, len(top)


def _salted_ds(ds: float, n_bp: int, cond: Conditions) -> float:
    return ds + 0.368 * (n_bp - 1) * math.log(cond.na_molar)


def tm_site(seq: str, site: str, cond: Conditions) -> float:
    """Tm (C) of the probe for `seq` on the sequence `site`, under `cond`.

    NaN when `site_thermo` cannot price the duplex -- callers fall back to
    `tm_mismatched` on that signal, never to the perfect-match Tm.
    """
    t = site_thermo(seq, site)
    if t is None:
        return float("nan")
    dh, ds, n = t
    tm_k = (dh * 1000.0) / (_salted_ds(ds, n, cond) + _R * math.log(cond.probe_nM * 1e-9 / 4.0))
    return tm_k - 273.15 - FORMAMIDE_C_PER_PCT * cond.formamide_pct


def fraction_bound(seq: str, cond: Conditions, site: str | None = None) -> float:
    """Equilibrium fraction of a site occupied by probe in excess, 0..1.

    theta = K*C / (1 + K*C) with K = exp(-dG/RT) from the nearest-neighbour
    dH/dS at the step's salt. Formamide enters as the same linear correction the
    Tm uses -- evaluating at T + 0.65 C per % -- so the two never disagree about
    which of two sites is the more stable. NaN when the duplex cannot be priced.
    """
    t = site_thermo(seq, site if site is not None else seq)
    if t is None:
        return float("nan")
    dh, ds, n = t
    temp_k = cond.temp_c + FORMAMIDE_C_PER_PCT * cond.formamide_pct + 273.15
    dg = dh * 1000.0 - temp_k * _salted_ds(ds, n, cond)          # cal/mol
    log_kc = -dg / (_R * temp_k) + math.log(cond.probe_nM * 1e-9)
    # 1 / (1 + 1/KC), computed without overflow at either extreme.
    if log_kc > 0:
        return 1.0 / (1.0 + math.exp(-log_kc))
    e = math.exp(log_kc)
    return e / (1.0 + e)


def min_fraction_bound(seq: str, site: str | None = None,
                       steps: tuple[Conditions, ...] = ALL_STEPS) -> float:
    """Fraction bound at the limiting step: the occupancy that survives them all."""
    vals = [fraction_bound(seq, c, site) for c in steps]
    return float("nan") if any(math.isnan(v) for v in vals) else min(vals)


def most_stringent(seq: str, steps: tuple[Conditions, ...] = ALL_STEPS) -> Conditions:
    """The step in `steps` giving this sequence its smallest margin."""
    return min(steps, key=lambda c: margin(seq, c))


def _step_tm(seq: str, n_mismatches: int, cond: Conditions, penalty_c: float,
             site: str | None) -> tuple[float, str]:
    """(Tm, model) for one step: from the site's sequence when it can be priced."""
    if site is not None:
        tm = tm_site(seq, site, cond)
        if not math.isnan(tm):
            return tm, (MODEL_PERFECT if site.upper() == seq.upper()
                        else MODEL_NN_MISMATCH)
    if n_mismatches == 0:
        return tm_buffer(seq, cond), MODEL_PERFECT
    return tm_mismatched(seq, n_mismatches, cond, penalty_c), MODEL_COUNT_PENALTY


def survives(seq: str, n_mismatches: int = 0, steps: tuple[Conditions, ...] = ALL_STEPS,
             penalty_c: float = DEFAULT_MISMATCH_PENALTY_C,
             site: str | None = None) -> bool:
    """Would a duplex with this many mismatches still be bound after every step?

    Pass `site` -- the sequence actually present -- to price the mismatches by
    position and identity instead of by count.
    """
    return all(
        _step_tm(seq, n_mismatches, c, penalty_c, site)[0] > c.temp_c for c in steps
    )


def profile(seq: str, n_mismatches: int = 0,
            steps: tuple[Conditions, ...] = ALL_STEPS,
            penalty_c: float = DEFAULT_MISMATCH_PENALTY_C,
            site: str | None = None) -> dict:
    """Per-step Tm, margin and fraction bound, plus the limiting step.

    With `site` the mismatches are priced from sequence (`site_thermo`); without
    it, or where the parameters do not cover the duplex, by the flat count
    penalty. `model` says which, and `approximate` is True only for the latter.
    Fraction bound is NaN under the count model: a flat Tm penalty has no
    dH/dS to turn into an occupancy.
    """
    rows, models = [], set()
    for c in steps:
        tm, model = _step_tm(seq, n_mismatches, c, penalty_c, site)
        models.add(model)
        frac = (fraction_bound(seq, c, site) if model != MODEL_COUNT_PENALTY
                else float("nan"))
        rows.append({"step": c.name, "conditions": str(c), "tm_c": tm,
                     "temp_c": c.temp_c, "margin_c": tm - c.temp_c,
                     "fraction_bound": frac})
    worst = min(rows, key=lambda r: r["margin_c"]) if rows else None
    fracs = [r["fraction_bound"] for r in rows]
    model = (MODEL_COUNT_PENALTY if MODEL_COUNT_PENALTY in models
             else next(iter(models), MODEL_PERFECT))
    return {
        "seq": seq,
        "length": len(seq),
        "gc": gc_fraction(seq),
        "n_mismatches": n_mismatches,
        "steps": rows,
        "limiting_step": worst["step"] if worst else None,
        "min_margin_c": worst["margin_c"] if worst else float("nan"),
        "min_fraction_bound": (min(fracs) if fracs and not any(math.isnan(f) for f in fracs)
                               else float("nan")),
        "survives": all(r["margin_c"] > 0 for r in rows),
        "mismatch_penalty_c": penalty_c,
        "model": model,
        "approximate": model == MODEL_COUNT_PENALTY,
    }


def at_temperature(cond: Conditions, temp_c: float) -> Conditions:
    """Same buffer, different temperature."""
    return replace(cond, temp_c=temp_c)
