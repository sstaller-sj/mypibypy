"""The probe map: where the pairs sit on the target, and how one pair is built.

Two zoom levels, both drawn from a finished `DesignResult` -- nothing here
designs, scores for selection, or changes a result:

* **Region map** (`map_data` -> `region_charts`): the target as a ruler, the
  score of every 51-nt window along it, the best-ranked windows stacked into
  lanes where they overlap, and the ordered pairs drawn as left pad | gap |
  right pad. A previous run's pairs can be laid underneath so a tweak shows as
  a visible shift.
* **Pair close-up** (`pair_closeup_svg`): the 51-nt footprint base by base, with
  P1L and P1R bound antiparallel beneath it, their arms meeting at the gap where
  P2 lands, and each probe base shaded by how often it is paired in the probe's
  own folding ensemble when that was computed.

The window scores are a re-scan of the result's own sequence with the same
`design.scan` the pipeline uses. They are the base GC/Tm score only: for an RNA
target the pipeline also re-ranks by accessibility, so the map's "best windows"
can differ from the order the pairs were picked in. The map says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape

from .design import FOOTPRINT, GAP_LEN, LEFT_LEN, RIGHT_LEN, scan
from .sequence import GenomicInterval, mask_repeats

#: How many of the best-ranked windows are stacked into lanes. Every window is
#: still in the score track; stacking all ~1000 windows of a 1 kb region would
#: be a solid block.
TOP_WINDOWS = 40

#: Above this many nt the per-window score track is skipped -- it is a re-scan
#: of the whole target, and a long transcript would make every redraw slow.
MAX_SCORED_NT = 20_000

_COMPLEMENT = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")


@dataclass
class MapData:
    axis_title: str                 # "chr7 (hg38)", "NM_000546.6 (nt)", ...
    origin: int                     # coordinate of offset 0
    length: int                     # nt
    pairs: list[dict] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)   # top-ranked, with lanes
    scores: list[dict] = field(default_factory=list)    # every window
    note: str = ""                  # why something is missing or approximate


def _is_genomic(result) -> bool:
    iv = result.locus.interval
    return iv.chrom != "transcript" and iv.length == len(result.region_seq)


def scored_windows(result) -> list:
    """Every valid 51-nt window of the result's own sequence, best first.

    Genomic targets are re-masked with the result's own RepeatMasker hits, so
    a window the pipeline could not place a pad in is not shown as available.
    """
    seq = result.region_seq
    if not seq or len(seq) > MAX_SCORED_NT or len(seq) < FOOTPRINT:
        return []
    if _is_genomic(result):
        iv = result.locus.interval
        return scan(mask_repeats(seq, iv, result.rmsk), iv, top=len(seq))
    plain = seq.upper().replace("U", "T")
    return scan(plain, GenomicInterval("map", "map", 1, len(plain)), top=len(plain))


def assign_lanes(spans: list[tuple[int, int]], gap: int = 1) -> list[int]:
    """Greedy pileup: the lowest lane whose last span ends before this one starts.

    `spans` are (start, end) half-open; the returned lanes are parallel to it.
    """
    order = sorted(range(len(spans)), key=lambda i: spans[i])
    lane_ends: list[int] = []
    lanes = [0] * len(spans)
    for i in order:
        start, end = spans[i]
        for lane, last in enumerate(lane_ends):
            if last + gap <= start:
                lane_ends[lane] = end
                lanes[i] = lane
                break
        else:
            lanes[i] = len(lane_ends)
            lane_ends.append(end)
    return lanes


def _pair_rows(result) -> list[dict]:
    rows = []
    for i, (pair, screen, _access) in enumerate(result.rows(), start=1):
        cand = pair.candidate
        rows.append(dict(
            pair=i, offset=cand.offset,
            left_tm=round(cand.left.tm, 1), right_tm=round(cand.right.tm, 1),
            left_gc=round(cand.left.gc, 2), right_gc=round(cand.right.gc, 2),
            screen=screen.verdict if screen.screened else "NOT SCREENED",
            flags=", ".join(cand.flags) or "none",
        ))
    return rows


def map_data(result, windows: list | None = None) -> MapData:
    """Everything the region map draws, in target coordinates.

    `windows` is `scored_windows(result)`, passed in when it was computed once
    already (it is the slow part); computed here otherwise.
    """
    if windows is None:
        windows = scored_windows(result)
    if _is_genomic(result):
        iv = result.locus.interval
        title, origin = f"{iv.chrom} ({iv.build})", iv.start
    elif result.locus.interval.chrom == "transcript":
        title, origin = f"{result.locus.interval.build} (nt)", 1
    else:
        title, origin = "position (nt)", 1

    data = MapData(axis_title=title, origin=origin, length=len(result.region_seq))
    for row in _pair_rows(result):
        start = origin + row["offset"]
        data.pairs.extend([
            dict(row, part="left pad", start=start, end=start + LEFT_LEN),
            dict(row, part="gap", start=start + LEFT_LEN,
                 end=start + LEFT_LEN + GAP_LEN),
            dict(row, part="right pad", start=start + LEFT_LEN + GAP_LEN,
                 end=start + FOOTPRINT),
        ])

    top = windows[:TOP_WINDOWS]
    lanes = assign_lanes([(c.offset, c.offset + FOOTPRINT) for c in top])
    ordered = {c.offset for c in (p.candidate for p in result.pairs)}
    for rank, (cand, lane) in enumerate(zip(top, lanes), start=1):
        data.windows.append(dict(
            rank=rank, lane=lane, start=origin + cand.offset,
            end=origin + cand.offset + FOOTPRINT,
            penalty=round(cand.penalty, 3),
            gc=round(cand.gc_mean, 2), tm_delta=round(cand.tm_delta, 1),
            ordered="ordered" if cand.offset in ordered else "not ordered",
        ))
    data.scores = [dict(pos=origin + c.offset + LEFT_LEN,
                        penalty=round(c.penalty, 3))
                   for c in sorted(windows, key=lambda c: c.offset)]

    notes = []
    if not windows:
        notes.append("Window scores not drawn: the target is longer than "
                     f"{MAX_SCORED_NT:,} nt or has no valid window.")
    elif result.locus.interval.chrom == "transcript" or not _is_genomic(result):
        notes.append("Window scores are the base GC/Tm score; RNA designs are "
                     "also re-ranked by accessibility, so pair order can differ.")
    data.note = " ".join(notes)
    return data


def previous_pairs(data: MapData, previous: MapData | None) -> list[dict]:
    """The previous run's pairs, for drawing faintly beneath the current ones."""
    if previous is None or previous.axis_title != data.axis_title:
        return []
    return [dict(p, part="previous") for p in previous.pairs
            if p["part"] != "gap"]


# --- region chart (Altair) ---------------------------------------------------

#: Width reserved for every track's y-axis, so the three charts' x positions
#: line up when they are drawn one above the other.
_Y_AXIS_PX = 96


def region_charts(data: MapData, previous: MapData | None = None,
                  selected_pair: int | None = None) -> dict:
    """The region map as separate Altair charts on one shared x domain.

    Separate rather than one `vconcat`: Streamlit can only stretch a single
    chart to its column, and a concatenated chart overflowed the design panel.
    Each track reserves the same y-axis width, so they still line up.

    Returns {"pairs": chart, "windows": chart | None, "scores": chart | None}.
    Click a pair in "pairs" to select it (selection name: "pair").

    Altair ships with Streamlit, so it is imported here rather than declared:
    nothing outside the app draws a chart.
    """
    import altair as alt
    import pandas as pd

    x_scale = alt.Scale(domain=[data.origin, data.origin + max(data.length, 1)],
                        nice=False)

    def y_axis(**kwargs):
        return alt.Axis(minExtent=_Y_AXIS_PX, maxExtent=_Y_AXIS_PX, **kwargs)

    def finish(chart):
        # Fit the column's width only. Fitting both made each track's height
        # the TOTAL height, so a legend and an axis left the 30-px pairs track
        # no room at all and it rendered as nothing.
        return (chart.properties(autosize=alt.AutoSizeParams(
                    type="fit-x", contains="padding"))
                .configure_view(strokeWidth=0))

    pick = alt.selection_point(name="pair", fields=["pair"], on="click",
                               clear=False)
    prev = previous_pairs(data, previous)
    pairs = pd.DataFrame(data.pairs + prev)
    pairs["track"] = pairs["part"].map(
        lambda p: "previous run" if p == "previous" else "ordered pairs")
    pairs["is_selected"] = (pairs["pair"] == (selected_pair or 0)) \
        & (pairs["part"] != "previous")
    domain = ["left pad", "gap", "right pad"] + (["previous"] if prev else [])
    part_colors = alt.Scale(domain=domain,
                            range=["#1D9E75", "#BA7517", "#0F6E56", "#B4B2A9"])
    charts = {"pairs": finish(
        alt.Chart(pairs)
        .mark_rect(cornerRadius=2, height={"band": 0.7})
        .encode(
            x=alt.X("start:Q", scale=x_scale,
                    axis=alt.Axis(format="d", title=data.axis_title,
                                  labelFlush=True)),
            x2="end:Q",
            y=alt.Y("track:N", title=None, axis=y_axis(),
                    sort=["ordered pairs", "previous run"]),
            color=alt.Color("part:N", scale=part_colors, title=None,
                            legend=alt.Legend(orient="top")),
            stroke=alt.condition("datum.is_selected", alt.value("#D85A30"),
                                 alt.value("transparent")),
            strokeWidth=alt.condition("datum.is_selected", alt.value(2.5),
                                      alt.value(0)),
            tooltip=[alt.Tooltip("pair:Q", title="pair"), "part:N",
                     alt.Tooltip("start:Q", format="d"),
                     alt.Tooltip("left_tm:Q", title="Tm L"),
                     alt.Tooltip("right_tm:Q", title="Tm R"),
                     alt.Tooltip("left_gc:Q", title="GC L"),
                     alt.Tooltip("right_gc:Q", title="GC R"),
                     "screen:N", "flags:N"],
        )
        .add_params(pick)
        .properties(height=30 * pairs["track"].nunique())
    ), "windows": None, "scores": None}

    if data.windows:
        windows = pd.DataFrame(data.windows)
        charts["windows"] = finish(
            alt.Chart(windows)
            .mark_rect(height=5, cornerRadius=1)
            .encode(
                x=alt.X("start:Q", scale=x_scale, axis=None), x2="end:Q",
                y=alt.Y("lane:O", title=f"best {len(windows)} windows",
                        axis=y_axis(labels=False, ticks=False, domain=False)),
                color=alt.Color("ordered:N", title=None,
                                scale=alt.Scale(domain=["ordered", "not ordered"],
                                                range=["#1D9E75", "#B4B2A9"]),
                                legend=None),
                tooltip=["rank:Q", alt.Tooltip("start:Q", format="d"),
                         "penalty:Q", alt.Tooltip("gc:Q", title="mean GC"),
                         alt.Tooltip("tm_delta:Q", title="ΔTm L/R")],
            )
            .properties(height=int(min(7 * (windows["lane"].max() + 1), 140)))
        )
    if data.scores:
        charts["scores"] = finish(
            alt.Chart(pd.DataFrame(data.scores))
            .mark_line(strokeWidth=1, color="#888780")
            .encode(
                x=alt.X("pos:Q", scale=x_scale, axis=None),
                y=alt.Y("penalty:Q", title="penalty",
                        scale=alt.Scale(reverse=True, zero=False),
                        axis=y_axis(tickCount=3)),
                tooltip=[alt.Tooltip("pos:Q", format="d", title="window centre"),
                         "penalty:Q"],
            )
            .properties(height=56)
        )
    return charts


# --- pair close-up (SVG) ------------------------------------------------------

_CELL = 12          # px per base
_ROW = 20           # px per row


def _shade(prob: float | None) -> str:
    """Background for a probe base: stronger red the more it is paired."""
    if prob is None:
        return "none"
    return f"rgba(226,75,74,{min(max(prob, 0.0), 1.0) * 0.75:.2f})"


def pair_closeup_svg(result, index: int, ink: str = "#2C2C2A") -> str:
    """Pair `index` (0-based) drawn base by base. Returns an `<svg>` string.

    Layout, left to right in target coordinates (5' -> 3' of the target):

        target   5' [left pad 25][gap][right pad 25] 3'
        P1L/P1R     [  P1L pad  ]     [  P1R pad  ]       antiparallel
        arms              [arm L]     [arm R]             meet at the gap
        P2                [=== lands on both arms ===]

    `P1L = arm_left + rc(left)`, so its arm hangs off the end next to the gap;
    `P1R = rc(right) + arm_right`, so its arm does too. Probe bases are the
    probe's own sequence, not recomputed complements.

    `ink` is the text colour. The SVG is shown as an image, which does not
    inherit the page's colour, so the app passes one that suits its theme.
    """
    pair = result.pairs[index]
    cand = pair.candidate
    target = (cand.left.seq + cand.gap_base + cand.right.seq).upper()
    arm_l = pair.system.arm_left.upper()
    arm_r = pair.system.arm_right.upper()
    p1l, p1r = pair.p1l.upper(), pair.p1r.upper()
    fold = result.folding_at(index)
    prob_l = fold.p1l.paired_prob if fold.evaluated and fold.p1l else ()
    prob_r = fold.p1r.paired_prob if fold.evaluated and fold.p1r else ()

    # Columns: 0..50 are the footprint. Arms may extend past either edge.
    gap_col = LEFT_LEN
    first = min(0, gap_col - len(arm_l))
    last = max(FOOTPRINT, gap_col + GAP_LEN + len(arm_r))
    margin = 64
    width = margin * 2 + (last - first) * _CELL
    x = lambda col: margin + (col - first) * _CELL      # noqa: E731

    cells: list[str] = []

    def base(col: int, row: int, letter: str, fill: str = "none",
             cls: str = "b") -> None:
        y = 24 + row * _ROW
        if fill != "none":
            cells.append(f'<rect x="{x(col)}" y="{y}" width="{_CELL}" '
                         f'height="{_ROW - 4}" fill="{fill}"/>')
        cells.append(f'<text class="{cls}" x="{x(col) + _CELL / 2}" '
                     f'y="{y + 12}">{escape(letter)}</text>')

    def prob(seq_probs, j):
        return seq_probs[j] if j < len(seq_probs) else None

    # Row 0: target, with pad/gap bands behind it.
    for col, letter in enumerate(target):
        band = ("rgba(186,117,23,0.35)" if col == gap_col
                else "rgba(29,158,117,0.22)")
        base(col, 0, letter, band)
    # Row 1: P1L pad under the left pad, P1R pad under the right pad.
    # P1L[len(arm_l) + k] pairs with left[24 - k]; P1R[k] pairs with right[24 - k].
    for k in range(LEFT_LEN):
        j = len(arm_l) + k
        if j < len(p1l):
            base(LEFT_LEN - 1 - k, 1, p1l[j], _shade(prob(prob_l, j)))
    for k in range(RIGHT_LEN):
        if k < len(p1r):
            base(FOOTPRINT - 1 - k, 1, p1r[k], _shade(prob(prob_r, k)))
    # Row 2: the arms. P1L's arm is its 5' end, adjacent to the gap; P1R's arm
    # is its 3' end, also adjacent to the gap.
    for j in range(len(arm_l)):
        base(gap_col - len(arm_l) + j, 2, p1l[j] if j < len(p1l) else arm_l[j],
             _shade(prob(prob_l, j)))
    for j in range(len(arm_r)):
        pj = RIGHT_LEN + j
        base(gap_col + GAP_LEN + j, 2, p1r[pj] if pj < len(p1r) else arm_r[j],
             _shade(prob(prob_r, pj)))

    # Orange outlines mark the arms (amplifier sequence, not target-binding).
    arm_y = 24 + 2 * _ROW - 1
    for c0, n in ((gap_col - len(arm_l), len(arm_l)), (gap_col + GAP_LEN, len(arm_r))):
        if n:
            cells.insert(0, f'<rect x="{x(c0)}" y="{arm_y}" width="{n * _CELL}" '
                            f'height="{_ROW - 2}" rx="2" fill="none" '
                            f'stroke="#D85A30" stroke-width="1"/>')

    y_p2 = 24 + 3 * _ROW + 4
    p2_from, p2_to = x(gap_col - len(arm_l)), x(gap_col + GAP_LEN + len(arm_r))
    end_y = lambda row: 24 + row * _ROW + 12          # noqa: E731
    shading = ("probe bases shaded by P(paired) in their own fold"
               if prob_l or prob_r else "self-folding not evaluated: no shading")
    height = y_p2 + 3 * _ROW + 8
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'aria-label="Pair {index + 1} close-up">',
        "<style>"
        f".b{{font:12px ui-monospace,Menlo,monospace;text-anchor:middle;fill:{ink}}}"
        f".l{{font:11px sans-serif;fill:{ink};opacity:.75}}"
        f".e{{font:11px sans-serif;fill:{ink};opacity:.6}}"
        "</style>",
        f'<text class="l" x="{x(0)}" y="14">left pad 25 nt</text>',
        f'<text class="l" x="{x(gap_col) - 4}" y="14">gap</text>',
        f'<text class="l" x="{x(gap_col + GAP_LEN) + 18}" y="14">right pad 25 nt</text>',
        f'<text class="e" x="{x(first) - 8}" y="{end_y(0)}" text-anchor="end">target 5\'</text>',
        f'<text class="e" x="{x(last) + 8}" y="{end_y(0)}">3\'</text>',
        f'<text class="e" x="{x(first) - 8}" y="{end_y(1)}" text-anchor="end">P1L 3\'</text>',
        f'<text class="e" x="{x(last) + 8}" y="{end_y(1)}">5\' P1R</text>',
        f'<text class="e" x="{x(gap_col - len(arm_l)) - 8}" y="{end_y(2)}" '
        f'text-anchor="end">arm L 5\'</text>',
        f'<text class="e" x="{x(gap_col + GAP_LEN + len(arm_r)) + 8}" '
        f'y="{end_y(2)}">3\' arm R</text>',
        *cells,
        f'<rect x="{p2_from}" y="{y_p2}" width="{p2_to - p2_from}" height="14" '
        f'rx="3" fill="rgba(216,90,48,0.25)" stroke="#D85A30" stroke-width="0.5"/>',
        f'<text class="l" x="{(p2_from + p2_to) / 2}" y="{y_p2 + 11}" '
        f'text-anchor="middle">P2 lands only when both arms are present</text>',
        f'<text class="e" x="{(p2_from + p2_to) / 2}" y="{y_p2 + 34}" '
        f'text-anchor="middle">P2 → P3 → P4 ({escape(pair.system.fluorophore)})'
        f' · {shading}</text>',
        "</svg>",
    ]
    return "".join(parts)
