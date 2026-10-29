"""Presentation helpers shared by the Streamlit app.

These live here rather than in `app.py` so they can be imported and tested: a
Streamlit script executes its page body on import, so anything defined inside it
is awkward to reach from a test. The app keeps the layout; this keeps the logic
that turns a `DesignResult` into rows and bytes.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

from .expression import EXPRESSION_IS_UNCALIBRATED
from .workbook import write_workbook


def candidate_rows(result) -> list[dict]:
    """One row per ordered pair: composition, thermodynamics, and both screens.

    `accessibility`, `specificity`, `self-folding` and the two P(bound) columns
    render the *absence* of a result as words rather than as a blank or a zero
    -- "not evaluated", "NOT SCREENED", "not computed" -- because a table cell
    is exactly where a missing screen would otherwise look like a passing one.
    """
    rows = []
    for i, (pair, screen, access) in enumerate(result.rows(), start=1):
        cand = pair.candidate
        rows.append({
            "#": i,
            "footprint": str(cand.interval),
            "GC L": round(cand.left.gc, 3),
            "GC R": round(cand.right.gc, 3),
            "Tm L": round(cand.left.tm, 1),
            "Tm R": round(cand.right.tm, 1),
            "accessibility": (round(access.pair_nucleation_p, 3)
                              if access.evaluated else "not evaluated"),
            "specificity": screen.verdict if screen.screened else "NOT SCREENED",
            "on-target P(bound)": on_target_cell(screen),
            "worst off-target P(bound)": worst_offtarget_cell(screen),
            "self-folding": folding_cell(result.folding_at(i - 1)),
            "off-target genes": ", ".join(screen.offtarget_genes[:4]) or "-",
            "QC flags": ", ".join(cand.flags) or "none",
            "P1L": pair.p1l_name,
            "P1R": pair.p1r_name,
        })
    return rows


def workbook_bytes(results, note: str) -> bytes:
    """Render the order workbook into memory.

    Via a temporary directory, never into the repository: the workbook carries
    probe sequences, and the browser download is the only place they should go.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "order.xlsx"
        write_workbook(str(path), results, note)
        return path.read_bytes()


def no_change_reply(intent: str, run_number: int) -> str:
    """The chat's answer when a design request changes nothing the design uses.

    "Too few passed, loosen the Tm" came back as a change with every field at
    its sentinel and a note promising to loosen the Tm; the app then designed
    the identical run again under that promise. The pipeline is deterministic,
    so nothing is designed and the reply says what would change it instead.
    """
    return (f"The model read that as *{intent}*, but it changes none of the "
            f"settings the design uses (channel, pairs, minimum spacing, the "
            f"specificity screen, GC/Tm enforcement), so run #{run_number} stands: "
            f"the pipeline is deterministic and designing again would give the same "
            f"pairs. The GC/Tm window itself cannot be widened from the chat; it "
            f"only rejects windows when enforcement is on. Say what to change, e.g. "
            f"*6 pairs*, *20 nt spacing*, *screen against hg38* or *enforce the "
            f"GC/Tm window*.")


def unscreened_pairs(result) -> int:
    """How many ordered pairs have no specificity result. Drives the UI warning."""
    return sum(1 for screen in result.pair_screens if not screen.screened)


def unscreened_reason(result) -> str:
    """Why the first unscreened pair was not screened, or "" if all were."""
    for screen in result.pair_screens:
        if not screen.screened:
            return screen.reason or "no reason recorded"
    return ""


def folding_cell(fold) -> str:
    """The limiting primary's folding word, or "not evaluated"."""
    return fold.limiting.label if fold.evaluated else "not evaluated"


def folding_details(result) -> list[str]:
    """One line per ordered pair: the full self-folding summary, caveat included."""
    return [f"Pair {i + 1}: {result.folding_at(i).summary()}"
            for i in range(result.n_pairs)]


def _fraction(x: float) -> str:
    # Two significant figures, not rounding to a fixed place: an occupancy of
    # 3e-5 rounded to three decimals would print as 0.0 and read as "none".
    return f"{x:.2g}"


def on_target_cell(screen) -> str:
    """On-target equilibrium fraction bound, or why there is none."""
    if not screen.screened:
        return "NOT SCREENED"
    if math.isnan(screen.on_target_fraction):
        return "not computed"
    return _fraction(screen.on_target_fraction)



def worst_offtarget_cell(screen) -> str:
    """Highest off-target fraction bound among signal-capable sites.

    NaN has two meanings here and they must not be conflated: a screen that
    found no signal-capable site has nothing to price, while one that found
    sites but priced none did not compute the number.
    """
    if not screen.screened:
        return "NOT SCREENED"
    worst = screen.worst_offtarget_fraction
    if not math.isnan(worst):
        return _fraction(worst)
    if not screen.signal_capable_sites:
        return "no off-target site"
    return "not computed"


def expression_status(result) -> tuple[bool, str]:
    """(weighted?, sentence) for expression weighting across a result's pairs.

    Only "weighted" when every ordered pair was; otherwise the sentence says
    weighting was NOT EVALUATED and why, so a partial weighting cannot pass for
    a complete one.
    """
    screens = [screen for _, screen, _ in result.rows()]
    missing = [s for s in screens if not s.expression_evaluated]
    if not missing:
        source = screens[0].expression.source
        return True, (f"Off-target genes weighted by expression from `{source}`. "
                      f"Note: {EXPRESSION_IS_UNCALIBRATED}.")
    reasons = sorted({s.expression_reason for s in missing})
    scope = ("" if len(missing) == len(screens)
             else f" for {len(missing)} of {len(screens)} pair(s)")
    return False, (f"Expression weighting NOT EVALUATED{scope} — "
                   f"{'; '.join(reasons)}.")
