"""The probe map's data and the pair close-up.

The chart itself was checked by running the app. What is guarded here is that
the map draws the pairs where the design put them, that the close-up shows the
probe's own bases in the right columns, and that a missing fold is said rather
than drawn as unshaded "fine".
"""
from __future__ import annotations

import random

import pytest

from mypibypy.design import FOOTPRINT, LEFT_LEN, build_tiled_pairs, scan, select_tiled
from mypibypy.loci import Locus
from mypibypy.pipeline import DesignResult
from mypibypy.probe_map import (
    TOP_WINDOWS,
    assign_lanes,
    map_data,
    pair_closeup_svg,
    previous_pairs,
    scored_windows,
)
from mypibypy.sequence import GenomicInterval
from mypibypy.specificity import not_screened
from mypibypy.systems import systems


@pytest.fixture(scope="module")
def system():
    reg = systems()
    assert reg, "no amplifier systems configured"
    return list(reg.values())[0]


def make_result(system, n_pairs=3, start=5000, seed=77):
    rng = random.Random(seed)
    seq = "".join(rng.choice("ACGT") for _ in range(800))
    iv = GenomicInterval("hg38", "chr1", start, start + len(seq) - 1)
    cands = scan(seq, iv, top=len(seq))
    pairs = build_tiled_pairs("t", "GENE", system, select_tiled(cands, n_pairs, 20))
    screens = [not_screened("test") for _ in pairs]
    return DesignResult(Locus("t", "GENE", "test", iv, system.name), system, seq,
                        cands[:5], pairs[0], [], screens[0], pairs=pairs,
                        pair_screens=screens)


def test_lanes_never_overlap():
    spans = [(0, 51), (10, 61), (52, 103), (60, 111), (200, 251)]
    lanes = assign_lanes(spans)
    for i, a in enumerate(spans):
        for j, b in enumerate(spans):
            if i < j and lanes[i] == lanes[j]:
                assert a[1] < b[0] or b[1] < a[0]
    assert lanes[0] == lanes[2] == lanes[4] == 0


def test_pairs_are_drawn_at_their_genomic_footprint(system):
    result = make_result(system)
    data = map_data(result)
    assert data.axis_title == "chr1 (hg38)"
    for pair in result.pairs:
        cand = pair.candidate
        drawn = {p["part"]: p for p in data.pairs
                 if p["offset"] == cand.offset}
        assert drawn["left pad"]["start"] == cand.interval.start
        assert drawn["gap"]["start"] == cand.interval.start + LEFT_LEN
        assert drawn["right pad"]["end"] == cand.interval.start + FOOTPRINT


def test_windows_are_the_best_ranked_and_mark_ordered_ones(system):
    result = make_result(system)
    data = map_data(result)
    assert len(data.windows) == TOP_WINDOWS
    assert [w["rank"] for w in data.windows] == list(range(1, TOP_WINDOWS + 1))
    assert data.windows[0]["penalty"] <= data.windows[-1]["penalty"]
    # Greedy tiling always takes the best window first.
    assert data.windows[0]["ordered"] == "ordered"
    assert len(data.scores) == len(scored_windows(result))


def test_previous_pairs_only_on_the_same_axis(system):
    now, before = make_result(system), make_result(system, n_pairs=2)
    assert previous_pairs(map_data(now), map_data(before))
    elsewhere = make_result(system, start=90_000)
    elsewhere.locus = Locus("t", "GENE", "test",
                            GenomicInterval("hg38", "chr2", 1, 800), system.name)
    assert previous_pairs(map_data(now), map_data(elsewhere)) == []


def test_closeup_shows_the_probes_own_bases(system):
    result = make_result(system)
    pair = result.pairs[0]
    svg = pair_closeup_svg(result, 0)
    letters = "".join(t.split(">")[-1] for t in svg.split("</text>")
                      if 'class="b' in t)
    target = pair.candidate.left.seq + pair.candidate.gap_base + pair.candidate.right.seq
    assert letters.startswith(target.upper())
    # Every base of both primaries appears exactly once in the drawing.
    probe_letters = letters[len(target):]
    assert sorted(probe_letters) == sorted(pair.p1l.upper() + pair.p1r.upper())


def test_closeup_says_when_folding_was_not_evaluated(system):
    svg = pair_closeup_svg(make_result(system), 0)
    assert "self-folding not evaluated" in svg


def test_region_charts_build(system):
    pytest.importorskip("altair")
    from mypibypy.probe_map import region_charts
    result = make_result(system)
    charts = region_charts(map_data(result), map_data(make_result(system, 2)), 2)
    spec = charts["pairs"].to_dict()
    assert spec["params"][0]["name"] == "pair"
    assert charts["windows"] is not None and charts["scores"] is not None
