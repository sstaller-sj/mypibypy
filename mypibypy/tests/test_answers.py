"""The chat's software-written answers: one template per topic, no model."""
from __future__ import annotations

import re

import pytest

from mypibypy.agent.actions import DesignParameters
from mypibypy.agent.answers import (
    NEEDS_DESIGN,
    Answer,
    Asked,
    after_design,
    answer,
    starters,
)
from mypibypy.agent.battery import _design, synthetic_sequence
from mypibypy.agent.context import ScreenStatus
from mypibypy.agent.intent import extract_target
from mypibypy.agent.topics import ALL_KEYS, NONE, OTHER, TOPIC_KEYS, TOPICS, guide
from mypibypy.session import DesignSession
from mypibypy.systems import systems

KEYS = tuple(sorted(systems()))
DNA_RUN = re.compile(r"[ACGTUacgtu]{15,}")
SCREEN = ScreenStatus(on=False, against="off",
                      available=["hg38 (genome, human GRCh38)",
                                 "gencode-v50 (transcriptome, GENCODE v50)"])


def designed(n_pairs: int, min_spacing: int = 50):
    """(session, run) for the battery's synthetic 600 nt target, unscreened."""
    seq = synthetic_sequence()
    s = DesignSession(params=DesignParameters(system=KEYS[0], n_pairs=n_pairs,
                                              min_spacing=min_spacing))
    s.target = extract_target(f"design for {seq}")
    run, _ = _design(s, s.params, seq)
    return s, run


@pytest.fixture(scope="module")
def six():
    return designed(6)


@pytest.fixture(scope="module")
def short():
    """Asked for more pairs than the target holds at this spacing."""
    return designed(12)


def ask(topic: str, message: str, made, **over) -> Answer:
    s, run = made
    return answer(topic, Asked(message=message, params=s.params, run=run,
                               screen=SCREEN, target=str(s.target),
                               system_keys=KEYS, **over))


class TestEveryTopicIsAnswered:
    def test_the_topic_list_and_the_templates_match(self):
        assert len(TOPIC_KEYS) == len(set(TOPIC_KEYS)) == 22
        assert ALL_KEYS == (*TOPIC_KEYS, OTHER, NONE)
        assert NEEDS_DESIGN <= set(TOPIC_KEYS)

    @pytest.mark.parametrize("topic", TOPICS, ids=lambda t: t.key)
    def test_each_topic_gives_text_and_follow_ups(self, topic, six):
        got = ask(topic.key, topic.example, six)
        assert isinstance(got, Answer) and len(got.text) > 40
        assert 1 <= len(got.follow_ups) <= 3
        assert all(f and len(f) < 60 for f in got.follow_ups)

    @pytest.mark.parametrize("topic", TOPICS, ids=lambda t: t.key)
    def test_no_answer_carries_a_sequence(self, topic, six):
        got = ask(topic.key, topic.example, six)
        assert not DNA_RUN.findall(got.text + " ".join(got.follow_ups))

    def test_the_models_own_topics_are_left_to_the_caller(self, six):
        assert ask(OTHER, "anything", six) is None
        assert ask(NONE, "anything", six) is None
        assert ask("horoscope", "anything", six) is None

    def test_the_guide_lists_every_topic_once(self):
        text = guide()
        for key in ALL_KEYS:
            assert text.count(f'- "{key}":') == 1

    @pytest.mark.parametrize("topic", sorted(NEEDS_DESIGN))
    def test_a_design_question_with_no_design_says_so(self, topic):
        params = DesignParameters(system=KEYS[0])
        none = answer(topic, Asked(message="why?", params=params))
        assert "no target yet" in none.text
        held = answer(topic, Asked(message="why?", params=params,
                                   target="hg38 chr7:5,530,000-5,531,000"))
        assert "Nothing has been designed" in held.text
        assert "design 3 pairs" in held.follow_ups


class TestAboutTheDesign:
    def test_placement_quotes_the_diagnosis_and_offers_what_would_place_more(self, short):
        s, run = short
        got = ask("placement", "why so few?", short)
        placed = sum(r.n_pairs for r in run.results)
        assert placed < 12
        assert got.text.startswith(f"Placed {placed} of 12 requested.")
        assert "What would place more" in got.text
        assert any(f.startswith(("space them", "make it")) for f in got.follow_ups)

    def test_an_unscreened_design_is_never_described_as_passing(self, six):
        got = ask("screen", "were they checked?", six)
        assert "NOT screened" in got.text and "pass" not in got.text
        assert "screen against gencode-v50" in got.follow_ups, \
            "a pasted sequence is designed as a transcript"
        summary = ask("summary", "is it ok?", six)
        assert "NOT SCREENED" in summary.text and "single-copy" in summary.text

    def test_with_no_index_configured_it_says_how_to_add_one(self, six):
        s, run = six
        got = answer("screen", Asked(message="?", params=s.params, run=run,
                                     target=str(s.target)))
        assert "indexes.local.json" in got.text
        assert not any(f.startswith("screen against") for f in got.follow_ups)

    def test_folding_carries_the_uncalibrated_caveat(self, six):
        assert "UNCALIBRATED" in ask("folding", "is folding bad?", six).text

    def test_qc_flags_explains_only_the_flags_present(self, six):
        s, run = six
        flags = [f for r in run.results for p in r.pairs for f in p.candidate.flags]
        got = ask("qc_flags", "what are the flags?", six)
        if flags:
            assert flags[0] in got.text
            assert ("homopolymerN" in got.text) == any("homopolymer" in f for f in flags)
        else:
            assert "No placed pair carries a QC flag" in got.text


class TestAboutAPair:
    def test_the_value_asked_about_leads_and_is_the_designs_own(self, six):
        s, run = six
        cand = run.results[0].pairs[1].candidate
        got = ask("pair_detail", "what's the Tm of pair 2?", six)
        first = got.text.splitlines()[0]
        assert first.startswith("Pair 2's Tm is")
        assert f"{round(cand.left.tm, 1):g}" in first
        assert f"{round(cand.right.tm, 1):g}" in first
        assert "The rest of pair 2" in got.text

    def test_a_pair_that_does_not_exist_is_said_not_invented(self, six):
        got = ask("pair_detail", "tell me about pair 9", six)
        assert "There is no pair 9" in got.text and "6 pairs" in got.text

    def test_with_no_pair_named_it_asks_which(self, six):
        got = ask("pair_detail", "what's its Tm?", six)
        assert got.text.startswith("Which pair?")
        assert got.follow_ups[0] == "tell me about pair 1"

    def test_a_comparison_leads_with_which_ranks_better(self, six):
        s, run = six
        from mypibypy.agent.context import _details

        a, b = _details(run, [2, 1])
        better = a if a.penalty < b.penalty else b
        got = ask("compare_pairs", "why is pair 2 worse than pair 1?", six)
        assert got.text.startswith(f"Pair {better.pair} ranks better")
        assert "lower is better" in got.text

    def test_best_and_worst_are_by_the_pipelines_own_ranking(self, six):
        s, run = six
        from mypibypy.agent.context import _details

        ranked = sorted(_details(run, list(range(1, 7))), key=lambda d: d.penalty)
        got = ask("best_worst", "which is weakest?", six)
        assert f"Best ranked: pair {ranked[0].pair} " in got.text
        assert f"Weakest: pair {ranked[-1].pair} " in got.text
        assert "No pair was screened" in got.text


class TestWhatIf:
    def test_a_preview_is_the_tilers_own_answer_and_says_it_is_a_preview(self, six):
        from mypibypy.diagnosis import PREVIEW_CAVEAT

        got = ask("what_if", "would 3 fit?", six, n_pairs=3)
        assert "Pairs 6 → 3: 3 of 3 placed." in got.text
        assert PREVIEW_CAVEAT in got.text
        assert got.follow_ups[0] == "make it 3 pairs"

    def test_a_preview_that_falls_short_does_not_offer_to_design_it(self, six):
        got = ask("what_if", "would 40 fit?", six, n_pairs=40)
        assert "short of what it asks for" in got.text
        assert not any(f.startswith("make it") for f in got.follow_ups)

    def test_a_number_out_of_bounds_is_clamped_and_said(self, six):
        got = ask("what_if", "would 500 fit?", six, n_pairs=500)
        assert "→ 50" in got.text and "permitted range" in got.text

    def test_with_no_number_it_answers_how_to_get_more(self, six):
        assert ask("what_if", "what if?", six).text == \
            ask("get_more", "how do I get more?", six).text

    def test_capacity_at_this_spacing_and_with_footprints_touching(self, six):
        got = ask("capacity", "how many fit?", six)
        assert "At 50 nt minimum spacing" in got.text
        assert "0 nt spacing" in got.text

    def test_get_more_on_a_short_design_lists_the_previews(self, short):
        got = ask("get_more", "how do I get more?", short)
        assert got.text.startswith("The design is short")
        assert "Minimum spacing 50 →" in got.text


class TestMethodAndTool:
    def test_the_glossary_answers_the_term_asked_about(self, six):
        got = ask("glossary", "what does the penalty number mean?", six)
        assert got.text.startswith("**Penalty**") and "P(bound)" not in got.text
        both = ask("glossary", "what are Tm and P(bound)?", six)
        assert "**Tm**" in both.text and "**P(bound)**" in both.text

    def test_the_glossary_lists_its_terms_when_none_is_named(self, six):
        got = ask("glossary", "what does that mean?", six)
        assert "Which one?" in got.text and "Penalty" in got.text

    def test_setup_names_what_is_configured(self, six):
        got = ask("how_setup", "how do I screen mouse?", six)
        assert "hg38 (genome, human GRCh38)" in got.text
        assert KEYS[0] in got.text and "indexes.local.json" in got.text

    def test_out_of_scope_says_so_and_what_it_can_do(self, six):
        got = ask("out_of_scope", "which antibody?", six)
        assert got.text.startswith("That is outside what I can answer")
        assert "change the design" in got.text

    def test_the_target_is_asked_for_only_when_there_is_none(self, six):
        assert "Give me a target" not in ask("greeting", "hi", six).text
        fresh = answer("greeting", Asked(message="hi",
                                         params=DesignParameters(system=KEYS[0])))
        assert "Give me a target" in fresh.text


class TestFollowUps:
    def test_after_an_unscreened_design_the_screen_is_offered_first(self, six):
        s, run = six
        got = after_design(Asked(message="", params=s.params, run=run,
                                 screen=SCREEN, target=str(s.target)))
        assert got[0] == "screen against gencode-v50"
        assert "is this design ok?" in got

    def test_after_a_short_design_it_offers_why(self, short):
        s, run = short
        got = after_design(Asked(message="", params=s.params, run=run,
                                 target=str(s.target)))
        assert got[0] == "why did I get fewer pairs?"

    def test_starters_follow_what_the_session_holds(self, six):
        s, run = six
        params = DesignParameters(system=KEYS[0])
        assert "how does this work?" in starters(Asked(message="", params=params))
        assert "design 3 pairs" in starters(Asked(message="", params=params,
                                                  target="NM_000546.6"))
        assert "is this design ok?" in starters(Asked(message="", params=s.params,
                                                      run=run, target=str(s.target)))
