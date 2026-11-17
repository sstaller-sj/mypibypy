"""The chat battery: scripted turns through the chat path, scored, no network.

A `ScriptedClient` plays the model, so the scoring, the session bookkeeping
(a change is designed again, as the app does) and the report are tested here;
`scripts/benchmark_chat.py` points the same code at a real model.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mypibypy.agent import ScriptedClient
from mypibypy.agent.battery import (
    INDEX_KEYS,
    ModelfileSamplerClient,
    run_battery,
    run_loop,
    synthetic_sequence,
    turns,
)
from mypibypy.agent.client import SAMPLING
from mypibypy.agent.intent import UNCHANGED, UNCHANGED_INT
from mypibypy.systems import systems

KEYS = tuple(sorted(systems()))
DNA_RUN = re.compile(r"[ACGT]{15,}")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_chat.py"


def expected_replies(keys=KEYS) -> list[dict]:
    """One payload per turn that satisfies the turn's expectations."""
    out = []
    for turn in turns(keys):
        payload = {"intent": "change", "system": UNCHANGED, "n_pairs": UNCHANGED_INT,
                   "min_spacing": UNCHANGED_INT, "enforce_window": UNCHANGED,
                   "screen": UNCHANGED, "topic": "none",
                   "note": "The design has 4 pairs."}
        for key, want in turn.expect.items():
            payload[key] = sorted(want)[0] if isinstance(want, (set, frozenset)) else want
        out.append(payload)
    return out


@pytest.fixture(scope="module")
def perfect():
    client = ScriptedClient(structured_replies=expected_replies())
    return run_battery(client, system_keys=KEYS, label="scripted")


class TestScoring:
    def test_a_model_that_answers_as_expected_passes_every_turn(self, perfect):
        assert perfect.total == len(turns(KEYS)) == 31
        assert perfect.passed == perfect.total, [t.lines() for t in perfect.turns
                                                 if not t.ok]
        assert not perfect.unreachable

    def test_a_wrong_field_fails_that_turn_only(self):
        replies = expected_replies()
        replies[3]["n_pairs"] = 7                       # "make it 6 pairs"
        report = run_battery(ScriptedClient(structured_replies=replies),
                             system_keys=KEYS)
        assert report.passed == report.total - 1
        failed = [t for t in report.turns if not t.ok]
        assert len(failed) == 1 and "make it 6 pairs" in failed[0].message
        assert any("WRONG(7)" in line for line in failed[0].lines())

    def test_a_transport_error_is_reported_not_scored(self):
        report = run_battery(ScriptedClient(error="could not reach Ollama"),
                             system_keys=KEYS)
        assert report.unreachable and report.passed == 0
        assert all("could not reach" in t.error for t in report.turns)

    def test_a_reply_outside_the_schema_is_refused(self):
        replies = expected_replies()
        replies[0]["intent"] = "adjust"                 # the old vocabulary
        report = run_battery(ScriptedClient(structured_replies=replies),
                             system_keys=KEYS)
        assert "not one of" in report.turns[0].refused
        assert not report.turns[0].ok


class TestItRunsTheChatPath:
    def test_a_change_is_designed_again_as_the_app_would(self, perfect):
        six = next(t for t in perfect.turns if t.message == "make it 6 pairs")
        assert six.applied == ["pairs → 6"]
        assert six.designed.endswith("s") and "pair(s)" in six.designed
        assert perfect.designs >= 2

    def test_the_question_names_the_pairs_actually_placed(self, perfect):
        why = next(t for t in perfect.turns if t.message.startswith("why did I only get"))
        assert str(perfect.placed) in why.message or "6" in why.message

    def test_a_request_that_changes_nothing_designs_nothing(self, perfect):
        loosen = next(t for t in perfect.turns if "loosen the Tm" in t.message)
        assert loosen.applied == [] and loosen.designed == ""

    def test_a_question_with_a_topic_is_answered_by_the_software(self, perfect):
        """As the app: the model's part is the topic, and the reply is written
        from the design. Nothing of the model's is in it, so it is not put
        through the reply check."""
        answered = [t for t in perfect.turns if t.answer]
        assert len(answered) == perfect.answered_by_software >= 24
        assert all(t.reply_problems is None for t in answered)
        why = next(t for t in answered if t.message.startswith("why did I only get"))
        assert why.answer.startswith("Placed ")
        assert any("answered by the software" in line for line in why.lines())

    def test_a_question_is_never_applied_even_with_a_number_in_it(self, perfect):
        fit = next(t for t in perfect.turns if t.message == "would 10 pairs fit?")
        assert fit.ok and fit.applied == [] and fit.designed == ""
        assert "10" in fit.answer and "placed" in fit.answer

    def test_a_follow_up_is_answered_about_the_pair_it_names(self, perfect):
        three = next(t for t in perfect.turns if t.message == "and pair 3?")
        assert "Pair 3" in three.answer

    def test_a_model_written_reply_still_goes_through_the_reply_check(self):
        """Topic "other" is the one answer the model writes itself."""
        replies = expected_replies()
        index = next(i for i, t in enumerate(turns(KEYS))
                     if t.message == "is it ready to order?")
        replies[index].update(topic="other", note="The design has 99 pairs.")
        report = run_battery(ScriptedClient(structured_replies=replies),
                             system_keys=KEYS)
        turn = report.turns[index]
        assert turn.answer == "" and not turn.ok
        assert turn.reply_problems and "99" in turn.reply_problems[0]
        assert report.reply_check_flags == 1

    def test_the_screen_request_is_applied_without_screening(self, perfect):
        screen = next(t for t in perfect.turns if t.message == "screen against hg38")
        assert any("hg38" in a and "nothing is screened" in a for a in screen.applied)

    def test_turns_that_name_another_target_are_scored_only(self, perfect):
        first = perfect.turns[0]
        assert first.ok and first.applied == [] and first.designed == ""


class TestNothingIdentifyingIsPrinted:
    def test_the_report_carries_no_sequence(self, perfect):
        text = "\n".join(perfect.lines()) + json.dumps(perfect.to_dict())
        assert not DNA_RUN.findall(text)
        assert synthetic_sequence()[:30] not in text

    def test_a_reply_that_did_carry_bases_is_scrubbed(self):
        replies = expected_replies()
        replies[2]["note"] = "Try " + "ACGT" * 10 + " instead."
        report = run_battery(ScriptedClient(structured_replies=replies),
                             system_keys=KEYS)
        text = "\n".join(report.lines())
        assert "ACGTACGT" not in text and "[bases removed]" in text


class TestTheLoopDemo:
    def test_an_unscreened_design_is_flagged_without_a_model(self):
        client = ScriptedClient([{"action": "proceed", "reason": "fine",
                                  "primary_signal": "specificity"}])
        demo = run_loop(client, system_keys=KEYS)
        assert demo.status == "flagged" and demo.model_calls == 0
        assert client.calls == []
        assert any("[deterministic_decision] flag_for_review" in e for e in demo.events)
        assert not DNA_RUN.findall("\n".join(demo.lines()))


class TestTheSamplerSwitch:
    def test_the_modelfile_client_drops_the_pinned_sampler(self):
        body = ModelfileSamplerClient().request_body("s", "e", {"type": "object"})
        options = body["options"]
        assert "top_k" not in options and "presence_penalty" not in options
        assert options["temperature"] == 0.0 and "num_ctx" in options
        assert "modelfile sampler" in ModelfileSamplerClient().name

    def test_the_pinned_client_keeps_every_sampler_key(self):
        from mypibypy.agent.client import OllamaDecisionClient
        options = OllamaDecisionClient().request_body("s", "e")["options"]
        assert set(SAMPLING) <= set(options)


def test_the_script_describes_itself():
    done = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "--sampler" in done.stdout and "--loop" in done.stdout
    assert INDEX_KEYS[0] not in done.stderr
