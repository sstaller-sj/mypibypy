"""What a design came out as, and what would change it. No model.

One layer for every place that troubleshoots a design: the Streamlit chat (which
hands these to a small local model), and -- to come -- the MCP server (whose host
model reads them from a tool) and the results panel (which shows them as they
are). Working the
answer out here, once, is the point: a local model asked to compare numbers or
infer a reason got them wrong, and each fix made in the chat's prompt helped the
chat alone. A wrong fact is now a bug in this module, fixed and tested here.

* `diagnose` states what happened: placement, the GC/Tm window, screening,
  folding, QC flags. Sentences a reader -- or a model -- can repeat as they are.
* `try_changes` previews what the changes the pipeline actually takes would do.
  The pipeline varies only the pair count, the spacing and the channel, and the
  channel picks the arms, not where pads sit; so a shortfall is answered by
  spacing and pair count. Loosening Tm or GC changes nothing unless the window is
  enforced, and the pipeline does not enforce it, so it is not offered.

The preview re-runs only placement (`design.tile`: greedy, re-packed if greedy
falls short) over the windows the pipeline tiled from -- milliseconds, no
fetch, no screen. For a genomic target those are the region's scored windows.
An RNA target is tiled in `design_rna`'s own order and costs (its accessibility
shortlist first), which the result keeps as `tiling_pool` / `tiling_costs`;
previewed from plain re-scored windows instead, one said "5 of 6" where
designing placed 3. No preview is shown unless it first reproduces the run it
starts from.

No sequence leaves here: every fact is a sentence of labels and numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .agent.actions import DesignParameters
from .agent.tools import window_stats
from .design import FOOTPRINT, tile

#: What a fact is about. Closed, so a caller can pick facts by topic.
TOPICS = ("placement", "windows", "screen", "folding", "qc")


@dataclass(frozen=True)
class Fact:
    topic: str
    text: str


@dataclass(frozen=True)
class WhatIf:
    """One previewed change: the parameters it would design with, and the result."""

    change: str                 # the parameter changed: "min_spacing" | "n_pairs"
    params: DesignParameters
    pairs_placed: int
    pairs_requested: int
    text: str


#: Said with every preview, so no reader takes one for a screened design.
PREVIEW_CAVEAT = ("Previews re-run placement only: the specificity screen and "
                  "folding check run when the change is designed.")


# --- facts --------------------------------------------------------------------

def free_gaps(result, spacing: int) -> list[tuple[str, int, bool]]:
    """(where, nt free, fits another footprint) between and around placed pairs."""
    need_edge, need_mid = FOOTPRINT + spacing, FOOTPRINT + 2 * spacing
    offsets = sorted(p.candidate.offset for p in result.pairs)
    edges = [0] + [o + FOOTPRINT for o in offsets]
    starts = offsets + [len(result.region_seq)]
    out = []
    for i, (lo, hi) in enumerate(zip(edges, starts)):
        free = hi - lo
        interior = 0 < i < len(offsets)
        where = ("before pair 1" if i == 0 else
                 f"after pair {len(offsets)}" if i == len(offsets) else
                 f"between pairs {i} and {i + 1}")
        out.append((where, free, free >= (need_mid if interior else need_edge)))
    return out


def placement_facts(result, params: DesignParameters) -> list[str]:
    """Placed vs requested for one result and, if short, the real reason --
    as short sentences, each one quotable on its own.

    The tiler is greedy by rank, so a shortfall is usually not "the target is
    too short" but "the best-ranked windows left gaps too small for another
    footprint". Both numbers are stated so that can be told apart.

    Short on purpose: said as one long sentence, the deciding part ("needs
    151 nt") sat in the middle, and the model quoted the start and the end.
    """
    placed, wanted, spacing = result.n_pairs, params.n_pairs, params.min_spacing
    length = len(result.region_seq)
    out = [f"Placed {placed} of {wanted} requested."]
    if result.repacked_from:
        out.append(f"Placing the best-ranked windows first fit only "
                   f"{result.repacked_from}, so the pairs were re-laid out across "
                   f"the target to fit {placed}; that uses some lower-ranked "
                   f"windows (see each pair's penalty).")
    if placed >= wanted:
        return out
    fit = max(0, (length + spacing) // (FOOTPRINT + spacing))
    need_mid, need_edge = FOOTPRINT + 2 * spacing, FOOTPRINT + spacing
    if fit < wanted:
        return out + [f"Each pair needs a {FOOTPRINT} nt footprint and {spacing} nt "
                      f"minimum spacing, and the target is {length} nt, which "
                      f"holds at most {fit} such pairs even packed perfectly."]
    # Said with what each gap needs: told only "105 nt free", a user (and the
    # model) read that as room for a 51 nt footprint, forgetting the spacing
    # owed on BOTH sides of a pair placed between two others.
    found = free_gaps(result, spacing)
    gaps = "; ".join(
        f"{w}: {n} nt free, needs {need_mid if w.startswith('between') else need_edge}"
        for w, n, _ in found)
    roomy = [w for w, _, fits in found if fits]
    out.append(f"At {spacing} nt minimum spacing, a new pair between two placed "
               f"ones needs {need_mid} nt free ({spacing} + {FOOTPRINT} + "
               f"{spacing}), and one before the first or after the last needs "
               f"{need_edge} nt ({FOOTPRINT} + {spacing}).")
    out.append(f"The gaps left are: {gaps}.")
    if result.packing_solved:
        # The layout was solved, not just filled greedily, so this is a bound.
        out.append(f"No layout of the target's usable windows fits more than "
                   f"{placed} at this spacing: the {length} nt target could hold "
                   f"{fit} only if every position had a usable window, and some "
                   f"do not (repeat-masked, or not a valid 51 nt window).")
        return out
    out.append("So no gap is big enough for another pair." if not roomy else
               f"{' and '.join(roomy).capitalize()} {'has' if len(roomy) == 1 else 'have'} "
               f"room, but no candidate window lies there: it may be "
               f"repeat-masked or not a valid 51 nt window.")
    out.append(f"The target ({length} nt) could hold {fit} if packed perfectly, "
               f"but pairs are placed greedily, best-ranked first. Windows are "
               f"not rejected for Tm or GC when the window is not enforced.")
    return out


def placement_reason(result, params: DesignParameters) -> str:
    """`placement_facts` as one paragraph."""
    return " ".join(placement_facts(result, params))


def tm_vs_window(cand, params: DesignParameters) -> str:
    """Both pads against the Tm window, in words, with direction and amount."""
    notes = []
    for side, pad in (("left", cand.left), ("right", cand.right)):
        if pad.tm < params.tm_min:
            notes.append(f"{side} pad {pad.tm:.1f} C is {params.tm_min - pad.tm:.1f} "
                         f"below {params.tm_min:g}")
        elif pad.tm > params.tm_max:
            notes.append(f"{side} pad {pad.tm:.1f} C is {pad.tm - params.tm_max:.1f} "
                         f"above {params.tm_max:g}")
    return "; ".join(notes) or (f"both pads within {params.tm_min:g}-"
                                f"{params.tm_max:g} C")


def windows_text(results, params: DesignParameters) -> tuple[str, str]:
    """(how the kept windows fare against the GC/Tm window, limiting constraint).

    Said in words that cannot be read as a rejection when the window is not
    enforced: told that only in a prompt, a model explained "0/5 pass" as
    though nothing had been designed.
    """
    stats = window_stats([c for r in results for c in r.candidates], params)
    text = str(stats) if params.enforce_window else (
        f"{stats.total} top-ranked windows were kept for reporting; "
        f"{stats.failed_tm} sit outside the Tm range and {stats.failed_gc} "
        f"outside the GC range, but NOTHING was rejected for it (the window "
        f"is not enforced; windows are ranked by distance from it)")
    return text, stats.limiting_constraint or "none"


def screening(results) -> tuple[int, int, float | None]:
    """(pairs screened, pairs not screened, worst off-target P(bound) or None)."""
    from .ui import unscreened_pairs

    placed = sum(r.n_pairs for r in results)
    not_screened = sum(unscreened_pairs(r) for r in results)
    priced = [s.worst_offtarget_fraction for r in results for _, s, _ in r.rows()
              if s.screened and not math.isnan(s.worst_offtarget_fraction)]
    return (placed - not_screened, not_screened,
            float(f"{max(priced):.2g}") if priced else None)


def most_folded(results) -> str:
    """The most folded primary's label, or "not evaluated"."""
    folds = [r.folding_at(i) for r in results for i in range(r.n_pairs)]
    evaluated = [f for f in folds if f.evaluated]
    worst = max(evaluated, key=lambda f: f.worst_paired, default=None)
    return worst.limiting.label if worst else "not evaluated"


def flagged_pairs(results) -> int:
    return sum(1 for r in results for p in r.pairs if p.candidate.flags)


def diagnose(results, params: DesignParameters) -> list[Fact]:
    """What the design came out as, one plain sentence per fact.

    `params` are the parameters the results were designed with.
    """
    from .ui import unscreened_reason

    facts = [Fact("placement", line) for r in results
             for line in placement_facts(r, params)]
    facts.append(Fact("windows", windows_text(results, params)[0] + "."))

    screened, unscreened, worst = screening(results)
    if not unscreened:
        text = f"All {screened} placed pairs were screened for off-targets"
        text += (f"; the worst off-target P(bound) is {worst:g}." if worst is not None
                 else "; no off-target site was priced.")
    else:
        reason = next((unscreened_reason(r) for r in results if unscreened_reason(r)),
                      "no reason recorded").rstrip(". ")
        text = (f"{unscreened} of {screened + unscreened} placed pairs were NOT "
                f"screened for off-targets ({reason}).")
        if worst is not None:
            text += f" The worst off-target P(bound) among those screened is {worst:g}."
    facts.append(Fact("screen", text))

    folded = most_folded(results)
    facts.append(Fact("folding", "Probe self-folding was not evaluated."
                      if folded == "not evaluated"
                      else f"The most folded primary is {folded}."))
    flagged = flagged_pairs(results)
    if flagged:
        facts.append(Fact("qc", f"{flagged} placed pair{'s' if flagged != 1 else ''} "
                                f"carry QC flags; see the results table."))
    return facts


# --- what if --------------------------------------------------------------------

def _spacings_to_try(spacing: int) -> list[int]:
    """Smaller spacings, a few steps down to touching footprints."""
    return sorted({spacing // 2, spacing // 4, 0} - {spacing}, reverse=True)


#: Why a design cannot be previewed, for a caller that asked for one.
NOT_PREVIEWABLE = ("a fusion is one pair per strand, so there is nothing to "
                   "tile; and a preview is shown only when it reproduces the "
                   "design it starts from")


def _previewable(results, params: DesignParameters, windows):
    """(the result, its scored windows), or None if a preview would not be the
    pipeline's own answer.

    `windows` is the result's scored windows, best first
    (`probe_map.scored_windows`); pass a cached list to skip re-scoring.
    """
    if len(results) != 1:
        return None
    result = results[0]
    from .probe_map import _is_genomic, scored_windows

    if result.tiling_pool:
        windows = result.tiling_pool
    elif not _is_genomic(result):
        return None
    elif windows is None:
        windows = scored_windows(result)
    # Shown only if it reproduces the run it starts from.
    if not windows or _placed(result, windows, params.n_pairs,
                              params.min_spacing) != result.n_pairs:
        return None
    return result, windows


def _placed(result, windows, n_pairs: int, spacing: int) -> int:
    """How many pairs the pipeline's tiler places, at the pipeline's costs."""
    costs = result.tiling_costs
    cost = (lambda c: costs.get(c.offset, c.penalty)) if costs else None
    return len(tile(windows, n_pairs, spacing, cost)[0])


def preview_change(results, params: DesignParameters, new: DesignParameters,
                   windows=None) -> WhatIf | None:
    """What designing with `new` instead of `params` would place. None if it
    cannot be previewed (see `NOT_PREVIEWABLE`). Only the pair count and the
    spacing move placement; the caller bounds `new` (`actions.clamp`)."""
    found = _previewable(results, params, windows)
    if found is None:
        return None
    result, windows = found
    got = _placed(result, windows, new.n_pairs, new.min_spacing)
    parts = []
    if new.n_pairs != params.n_pairs:
        parts.append(f"pairs {params.n_pairs} → {new.n_pairs}")
    if new.min_spacing != params.min_spacing:
        parts.append(f"minimum spacing {params.min_spacing} → {new.min_spacing} nt")
    changed = [name for name in ("n_pairs", "min_spacing")
               if getattr(new, name) != getattr(params, name)]
    said = ", ".join(parts) or "no change"
    return WhatIf("+".join(changed) or "none", new, got, new.n_pairs,
                  f"{said[0].upper()}{said[1:]}: {got} of {new.n_pairs} placed.")


def try_changes(results, params: DesignParameters, windows=None) -> list[WhatIf]:
    """Preview the changes that would place more of the requested pairs.

    Empty when nothing is short, or when the design cannot be previewed (see
    `NOT_PREVIEWABLE`).
    """
    found = _previewable(results, params, windows)
    if found is None:
        return []
    result, windows = found
    placed, wanted = result.n_pairs, params.n_pairs
    if placed >= wanted:
        return []

    out = []
    for spacing in _spacings_to_try(params.min_spacing):
        got = _placed(result, windows, wanted, spacing)
        out.append(WhatIf(
            "min_spacing", replace(params, min_spacing=spacing), got, wanted,
            f"Minimum spacing {params.min_spacing} → {spacing} nt: {got} of "
            f"{wanted} placed."))
    if placed:
        out.append(WhatIf(
            "n_pairs", replace(params, n_pairs=placed), placed, placed,
            f"Request {placed} pair{'s' if placed != 1 else ''} instead of "
            f"{wanted}: {placed} of {placed} placed, the same pairs as now."))
    return out
