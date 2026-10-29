"""The reply check: the chat model's answer against what it was shown.

The refused cases are the chat's real failures, as recorded in the comments of
`agent/context.py`; the accepted ones are the same answers said correctly.
"""
import pytest

from mypibypy.agent.reply_check import (
    check_reply,
    checked_note,
    numbers_in,
    retry_request,
)

GROUNDS = ('{"last_run": {"pairs_requested": 6, "pairs_placed": 3}, '
           '"facts": ["Placed 3 of 6 requested. Each pair needs a 51 nt footprint '
           'and 20 nt minimum spacing. The target (700 nt) could hold 9 ..."], '
           '"pair_details": [{"pair": 2, "tm_left": 73.8, "gc_left": 0.48}, '
           '{"pair": 3, "tm_left": 70.4, "gc_left": 0.52}], '
           '"worst_offtarget_p_bound": 2e-05, "target": "chr7:5,530,000-5,531,000"}')


def problems(note, enforce_window=False):
    return check_reply(note, GROUNDS, enforce_window=enforce_window)


class TestNumbers:
    def test_numbers_as_written(self):
        assert numbers_in("placed 3 of 6, Tm 70.4 C, at 5,530,000; P 2e-05") == \
            ["3", "6", "70.4", "5,530,000", "2e-05"]

    def test_a_number_from_the_design_passes(self):
        assert problems("Placed 3 of 6 requested; pair 3's left pad is 70.4 C.") == []

    def test_an_invented_number_is_refused(self):
        found = problems("Placed 4 of 6 requested.")
        assert len(found) == 1 and found[0].startswith("4 ")

    def test_rounding_and_percentages_are_fine(self):
        assert problems("Pair 3's left pad is about 70 C, with 52% GC.") == []

    def test_thousands_commas_and_exponents_are_fine(self):
        assert problems("The target starts at 5,530,000; the worst P(bound) "
                        "is 0.00002.") == []


class TestComparisons:
    def test_the_recorded_backwards_comparison_is_refused(self):
        # From the chat's own history: a small model called 70.4 higher than 73.8.
        found = problems("Pair 3's left-pad Tm of 70.4 is higher than pair 2's 73.8.")
        assert any("wrong way round" in p for p in found)

    def test_the_right_way_round_passes(self):
        assert problems("Pair 3's left-pad Tm of 70.4 is lower than pair 2's 73.8.") == []

    def test_pair_numbers_are_not_compared(self):
        """'pair 3 ... lower than pair 2' compares Tms, not the indexes 3 and 2."""
        assert problems("Pair 3 has a Tm of 70.4, lower than pair 2 at 73.8.") == []

    def test_more_than_is_not_read_as_a_comparison(self):
        assert problems("The gap is 20 nt, and a pair needs more than 51.") == []

    def test_numbers_in_another_sentence_are_not_compared(self):
        assert problems("Pair 2's Tm is 73.8. Pair 3's is lower than that, "
                        "at 70.4.") == []


class TestRejection:
    def test_rejected_is_refused_when_the_window_is_not_enforced(self):
        assert any("not enforced" in p
                   for p in problems("Three windows were rejected for Tm."))

    def test_saying_nothing_was_rejected_is_fine(self):
        assert problems("Nothing was rejected for Tm; windows are only ranked.") == []

    def test_rejected_is_fine_when_the_window_is_enforced(self):
        assert problems("Three windows were rejected for Tm.", enforce_window=True) == []


class TestCheckedNote:
    def settle(self, note, again=None):
        asked = []

        def ask_again(found):
            asked.append(found)
            return again

        out = checked_note(note, GROUNDS, enforce_window=False,
                           ask_again=ask_again, fallback="FACTS")
        return out, asked

    def test_a_good_note_is_shown_without_asking_again(self):
        (note, how), asked = self.settle("Placed 3 of 6 requested.")
        assert (note, how, asked) == ("Placed 3 of 6 requested.", "model", [])

    def test_a_bad_note_is_retried_with_its_problems(self):
        (note, how), asked = self.settle("Placed 4 of 6.", again="Placed 3 of 6.")
        assert (note, how) == ("Placed 3 of 6.", "retried")
        assert asked and "4 does not appear" in asked[0][0]

    def test_two_bad_notes_fall_back_to_the_facts(self):
        (note, how), _ = self.settle("Placed 4 of 6.", again="Placed 5 of 6.")
        assert (note, how) == ("FACTS", "fallback")

    def test_no_second_reply_falls_back_too(self):
        (note, how), _ = self.settle("Placed 4 of 6.", again=None)
        assert (note, how) == ("FACTS", "fallback")

    def test_the_retry_request_says_why(self):
        text = retry_request("REQUEST", ["4 does not appear in CURRENT DESIGN"])
        assert text.startswith("REQUEST") and "NOT SHOWN" in text
        assert "4 does not appear" in text


@pytest.mark.parametrize("note", [
    "The chat placed 3 of 6 pairs; at 20 nt spacing, gaps were too small.",
    "Hello! Ask me to design probes for a region.",
])
def test_ordinary_replies_pass(note):
    assert problems(note) == []
