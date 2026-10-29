"""The view of the design the chat model is shown each turn.

Guarded here: it can never carry a sequence, it mirrors every design setting,
its run summary matches the run, and per-pair detail appears only for the pairs
the message names.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import fields

import pytest

from mypibypy.agent import DesignParameters
from mypibypy.agent.context import (
    ChatContext,
    PairDetail,
    RunSummary,
    Settings,
    build_chat_context,
    pairs_named,
)
from mypibypy.design import build_tiled_pairs, scan, select_tiled
from mypibypy.loci import Locus
from mypibypy.pipeline import DesignResult
from mypibypy.sequence import GenomicInterval
from mypibypy.session import DesignSession
from mypibypy.specificity import not_screened
from mypibypy.systems import systems

#: Longer than any label could plausibly need; any probe pad is 25 nt.
SEQUENCE_SHAPED = re.compile(r"[ACGTUacgtu]{12,}")


@pytest.fixture(scope="module")
def system():
    reg = systems()
    assert reg, "no amplifier systems configured"
    return list(reg.values())[0]


@pytest.fixture
def run(system):
    rng = random.Random(3)
    seq = "".join(rng.choice("ACGT") for _ in range(700))
    iv = GenomicInterval("hg38", "chr1", 1000, 1699)
    cands = scan(seq, iv, top=len(seq))
    pairs = build_tiled_pairs("t", "GENE", system, select_tiled(cands, 3, 20))
    screens = [not_screened("test") for _ in pairs]
    result = DesignResult(Locus("t", "GENE", "test", iv, system.name), system,
                          seq, cands[:5], pairs[0], [], screens[0], pairs=pairs,
                          pair_screens=screens)
    s = DesignSession(params=DesignParameters(system=system.name, n_pairs=6))
    return s.record(source="chat", results=[result], note="",
                    target_label="hg38 chr1:1,000-1,699")


def test_settings_mirror_every_design_parameter():
    assert set(Settings.model_fields) == {f.name for f in fields(DesignParameters)}


def test_schema_has_no_field_that_could_hold_a_sequence():
    for model in (ChatContext, Settings, RunSummary, PairDetail):
        for name in model.model_fields:
            # `accept_single_arm_hits` is a yes/no setting, not an arm sequence.
            assert not re.search(r"seq|bases?$|oligo|probe|^arm|_arm$", name), name


def test_extra_fields_are_refused():
    with pytest.raises(Exception):
        ChatContext(target=None, settings=Settings.from_params(DesignParameters()),
                    sequence="ACGT")


def test_rendered_context_carries_no_sequence(run):
    ctx = build_chat_context("why is pair 2 worse than pair 1?", run.params,
                             run.target_label, run)
    text = ctx.render()
    assert not SEQUENCE_SHAPED.search(text)
    for pair in run.results[0].pairs:
        assert pair.candidate.left.seq not in text
        assert pair.p1l not in text and pair.p1r not in text


def test_summary_matches_the_run(run):
    summary = build_chat_context("", run.params, run.target_label, run).last_run
    result = run.results[0]
    assert summary.run_number == run.number
    assert (summary.pairs_requested, summary.pairs_placed) == (6, result.n_pairs)
    assert summary.pairs_not_screened == result.n_pairs
    assert summary.pairs_screened == 0
    assert summary.worst_offtarget_p_bound is None
    assert summary.most_folded == "not evaluated"


def test_pair_detail_only_for_named_pairs(run):
    assert build_chat_context("make it 4 pairs", run.params, run.target_label,
                              run).pair_details == []
    ctx = build_chat_context("why is pair 3 worse than pair 1?", run.params,
                             run.target_label, run)
    assert [d.pair for d in ctx.pair_details] == [3, 1]
    assert ctx.pair_details[0].specificity == "NOT SCREENED"
    # A pair that does not exist is left out, not invented.
    assert build_chat_context("pair 9", run.params, run.target_label,
                              run).pair_details == []


def test_no_run_means_no_summary():
    ctx = build_chat_context("hello", DesignParameters(system="cy3"))
    data = json.loads(ctx.render())
    assert "target" not in data and "last_run" not in data and data["pair_details"] == []


@pytest.mark.parametrize("message,expected", [
    ("why is pair 3 worse than pair 2?", [3, 2]),
    ("compare pairs 2 and 4", [2, 4]),
    ("#1 vs #3", [1, 3]),
    ("can you make a set of 4", []),
    ("12 pairs at 20 nt spacing", []),
])
def test_pairs_named(message, expected):
    assert pairs_named(message) == expected


def test_comparison_states_direction_from_the_numbers(run):
    ctx = build_chat_context("why is pair 3 worse than pair 2?", run.params,
                             run.target_label, run)
    three, two = ctx.pair_details
    lines = {line.split(" is ")[0]: line for line in ctx.pair_comparison}
    tm = lines.get(f"pair 3 left-pad Tm {three.tm_left:g}")
    if tm is not None:
        assert tm.split(" than ")[0].endswith(
            "lower" if three.tm_left < two.tm_left else "higher")
    penalty = next(l for l in ctx.pair_comparison if "ranking penalty" in l)
    direction = penalty.split(" than ")[0].rsplit(" ", 1)[-1]
    assert direction == ("lower" if three.penalty < two.penalty else "higher")
    assert build_chat_context("pair 2", run.params, run.target_label,
                              run).pair_comparison == []


# --- candidate windows and the shortfall reason -----------------------------

def test_windows_table_lists_kept_windows_with_status(run):
    ctx = build_chat_context("", run.params, run.target_label, run)
    result = run.results[0]
    assert len(ctx.windows) == len(result.candidates) == 5
    placed = {p.candidate.offset + 1 for p in result.pairs}
    for row, cand in zip(ctx.windows, result.candidates):
        assert row.target_position == f"{cand.offset + 1}-{cand.offset + 51}"
        assert row.tm_left == round(cand.left.tm, 1)
        assert row.penalty == round(cand.penalty, 3)
        if cand.offset + 1 in placed:
            assert row.status.startswith("placed as pair ")
        else:
            assert row.status.startswith("not placed")
    assert "not kept" in ctx.windows_note


def test_windows_absent_without_a_run():
    ctx = build_chat_context("hi", DesignParameters(system="cy3"))
    assert ctx.windows == [] and ctx.windows_note == ""


def test_tm_vs_window_says_direction_and_amount(run):
    params = DesignParameters(system=run.params.system, tm_min=0.0, tm_max=1.0)
    rows = build_chat_context("", params, run.target_label, run).windows
    assert all("above 1" in r.tm_vs_window for r in rows)
    params = DesignParameters(system=run.params.system, tm_min=0.0, tm_max=200.0)
    rows = build_chat_context("", params, run.target_label, run).windows
    assert all(r.tm_vs_window == "both pads within 0-200 C" for r in rows)


def test_shortfall_reason_blames_greedy_placement_when_room_existed(run):
    ctx = build_chat_context("", run.params, run.target_label, run)
    assert run.results[0].n_pairs == 3 and ctx.last_run.pairs_requested == 6
    assert ctx.facts[0] == "Placed 3 of 6 requested."
    placement = " ".join(ctx.facts)
    assert "if packed perfectly" in placement
    assert "free" in placement and "too short" not in placement


def test_facts_and_previews_come_from_the_diagnosis_layer(run):
    from mypibypy.diagnosis import diagnose, try_changes

    ctx = build_chat_context("", run.params, run.target_label, run)
    assert ctx.facts == [f.text for f in diagnose(run.results, run.params)]
    # The fixture's result was tiled at other settings than its run records, so
    # no preview reproduces it and none is offered; test_diagnosis.py has the
    # case where one is.
    assert ctx.what_if == [w.text for w in try_changes(run.results, run.params)] == []


def test_the_fallback_reply_is_the_facts(run):
    from mypibypy.agent.context import FACTS_FALLBACK, facts_reply

    ctx = build_chat_context("", run.params, run.target_label, run)
    reply = facts_reply(ctx)
    assert reply.startswith(FACTS_FALLBACK)
    assert all(line in reply for line in ctx.facts + ctx.what_if)


def test_no_run_means_no_facts():
    from mypibypy.agent.context import facts_reply

    ctx = build_chat_context("", DesignParameters(system="cy3"), "chr1:1-2")
    assert ctx.facts == [] and ctx.what_if == []
    assert "no design" in facts_reply(ctx)


def test_shortfall_reason_blames_length_when_it_cannot_fit(run):
    from mypibypy.agent.context import placement_reason

    params = DesignParameters(system=run.params.system, n_pairs=50, min_spacing=20)
    reason = placement_reason(run.results[0], params)
    assert "700 nt" in reason and "at most 10" in reason


def test_no_shortfall_when_all_placed(run):
    from mypibypy.agent.context import placement_reason

    params = DesignParameters(system=run.params.system, n_pairs=3, min_spacing=20)
    assert placement_reason(run.results[0], params) == "Placed 3 of 3 requested."


def test_unenforced_window_is_not_described_as_rejection(run):
    text = next(f for f in build_chat_context("", run.params, run.target_label,
                                              run).facts if "top-ranked windows were kept" in f)
    assert "NOTHING was rejected" in text
    assert "windows pass" not in text


def test_windows_carry_no_sequence(run):
    rendered = build_chat_context("pair 1", run.params, run.target_label, run).render()
    assert not SEQUENCE_SHAPED.search(rendered)
