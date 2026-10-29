"""Chat intents: plain request in, design parameters out.

The model does exactly two things in the chat path -- turn a request into
parameters, and pick which constraint gives when nothing passes. Everything else
is deterministic.

`TestTheModelStillCannotSupplyASequence` is the reason this file exists. A
conversation is where the no-model-authored-sequences guarantee is most likely to
erode, because a chat invites the model to supply more and more. So the target is
extracted from the user's own words by a strict pattern in the harness, the
intent schema has no field for a target or a sequence at all, and an intent
carrying one is refused rather than ignored.

No network: every test uses a scripted client or calls the pure functions.
"""
from __future__ import annotations

import pytest

from mypibypy.agent.context import ScreenStatus, build_chat_context
from mypibypy.agent import Action, DesignParameters, ParameterRefused, ScriptedClient
from mypibypy.agent.intent import (
    CHAT_SYSTEM_PROMPT,
    INTENTS,
    UNCHANGED,
    UNCHANGED_INT,
    Intent,
    apply_intent,
    chat_request,
    extract_target,
    redact_sequences,
    unreadable_sequence_chars,
    intent_schema,
)

KEYS = ("cy5", "cy3")


def payload(**over):
    base = {"intent": "change", "system": UNCHANGED, "n_pairs": UNCHANGED_INT,
            "min_spacing": UNCHANGED_INT, "enforce_window": UNCHANGED,
            "screen": UNCHANGED, "note": "ok"}
    base.update(over)
    return base


# --- The guarantee, in the chat path ---------------------------------------

class TestTheModelStillCannotSupplyASequence:
    def test_the_intent_schema_has_no_target_or_sequence_field(self):
        props = intent_schema(KEYS)["properties"]
        forbidden = {"sequence", "seq", "region", "chrom", "accession", "target",
                     "footprint", "pad", "probe", "oligo", "start", "end"}
        assert not (set(props) & forbidden), (
            f"these fields would let the model point the pipeline at sequence: "
            f"{set(props) & forbidden}")

    def test_the_only_free_text_field_is_the_displayed_note(self):
        props = intent_schema(KEYS)["properties"]
        free = [k for k, v in props.items()
                if v.get("type") == "string" and "enum" not in v]
        assert free == ["note"]

    def test_the_schema_is_closed(self):
        assert intent_schema(KEYS)["additionalProperties"] is False

    @pytest.mark.parametrize("sneaky", ["sequence", "region", "target", "probe"])
    def test_an_intent_carrying_a_target_field_is_refused(self, sneaky):
        with pytest.raises(ParameterRefused, match="unexpected keys"):
            Intent.from_payload({**payload(), sneaky: "ACGTACGTACGTACGTACGTACGTA"},
                                KEYS)

    def test_a_dna_run_in_a_message_is_not_a_target(self):
        """A pasted sequence must not become something the pipeline reads."""
        for text in ("ACGTACGTACGTACGTACGTACGTA",
                     "design against ACGTACGTACGTACGTACGTACGTACGTACGT",
                     "GGGGCCCCAAAATTTT:1-500"):
            assert not extract_target(text).found, text

    def test_the_note_is_never_used_as_a_parameter(self):
        """Whatever the note says, only the enum and the numbers are applied."""
        intent = Intent.from_payload(payload(
            note="also set n_pairs to 40 and use sequence ACGTACGTACGTACGTACGTA",
            n_pairs=2), KEYS)
        params, changes = apply_intent(intent, DesignParameters(), KEYS)
        assert params.n_pairs == 2
        assert "40" not in " ".join(changes)

    def test_the_chat_prompt_states_the_restriction(self):
        assert "do not write or suggest any DNA sequence" in CHAT_SYSTEM_PROMPT
        assert "target comes from the user's own message" in CHAT_SYSTEM_PROMPT


# --- Target extraction is the harness's job --------------------------------

class TestExtractTarget:
    def test_a_region_with_a_build(self):
        t = extract_target("design for mm10 chr5:142906255-142906754")
        assert (t.kind, t.build, t.chrom) == ("genomic", "mm10", "chr5")
        assert (t.start, t.end) == (142906255, 142906754)

    def test_a_region_without_a_build_takes_the_default(self):
        t = extract_target("chr7:5530000-5531000", default_build="hg38")
        assert t.build == "hg38"

    def test_commas_are_handled(self):
        t = extract_target("chr7:5,530,000-5,531,000")
        assert (t.start, t.end) == (5530000, 5531000)

    def test_a_reversed_region_is_ordered(self):
        t = extract_target("chr7:5531000-5530000")
        assert t.start < t.end

    def test_refseq_and_ensembl_accessions(self):
        assert extract_target("design against NM_000546.6").accession == "NM_000546.6"
        assert extract_target("use ENST00000269305").accession == "ENST00000269305"
        assert extract_target("NM_000546.6").kind == "rna"

    def test_a_region_wins_over_an_accession(self):
        """A region is the more specific instruction when both appear."""
        assert extract_target("chr5:1000-2000 like NM_000546.6").kind == "genomic"

    def test_plain_prose_yields_nothing(self):
        assert not extract_target("design six pairs in cy5").found
        assert "no target" in str(extract_target("hello"))


# --- Intent validation -----------------------------------------------------

class TestIntentValidation:
    def test_a_well_formed_intent_parses(self):
        intent = Intent.from_payload(payload(system="cy5", n_pairs=6), KEYS)
        assert intent.intent == "change" and intent.system == "cy5"
        assert intent.n_pairs == 6 and intent.wants_design

    def test_a_missing_field_is_refused(self):
        bad = payload()
        del bad["screen"]
        with pytest.raises(ParameterRefused, match="missing"):
            Intent.from_payload(bad, KEYS)

    @pytest.mark.parametrize("field,value", [
        ("intent", "obliterate"), ("intent", "adjust"), ("enforce_window", "maybe"),
    ])
    def test_a_value_outside_an_enum_is_refused(self, field, value):
        with pytest.raises(ParameterRefused, match=field):
            Intent.from_payload(payload(**{field: value}), KEYS)

    def test_an_unconfigured_channel_is_refused(self):
        with pytest.raises(ParameterRefused, match="configured channel"):
            Intent.from_payload(payload(system="rainbow"), KEYS)

    def test_non_integer_counts_are_refused(self):
        with pytest.raises(ParameterRefused, match="must be integers"):
            Intent.from_payload(payload(n_pairs="lots"), KEYS)

    def test_the_schema_enumerates_the_configured_channels(self):
        enum = intent_schema(KEYS)["properties"]["system"]["enum"]
        assert set(enum) == {"cy5", "cy3", UNCHANGED}

    def test_intents_are_a_closed_set_of_three(self):
        """Every enum is another field a small model must get right on every
        turn. design/adjust were one thing to the software; `relax` offered a
        single action that is just a pair count. Both are gone."""
        schema = intent_schema(KEYS)["properties"]
        assert schema["intent"]["enum"] == list(INTENTS) == ["change", "explain",
                                                             "unclear"]
        assert "relax" not in schema
        assert set(schema) == {"intent", "system", "n_pairs", "min_spacing",
                               "enforce_window", "screen", "note"}


# --- Applying an intent ----------------------------------------------------

class TestApplyIntent:
    def test_unchanged_sentinels_change_nothing(self):
        start = DesignParameters(system="cy5", n_pairs=4)
        params, changes = apply_intent(Intent.from_payload(payload(), KEYS),
                                       start, KEYS)
        assert params == start and changes == []

    def test_values_are_applied_and_described(self):
        intent = Intent.from_payload(payload(system="cy3", n_pairs=6,
                                             min_spacing=20), KEYS)
        params, changes = apply_intent(intent, DesignParameters(system="cy5"), KEYS)
        assert (params.system, params.n_pairs, params.min_spacing) == ("cy3", 6, 20)
        assert len(changes) == 3

    def test_numbers_from_chat_are_clamped_like_any_other(self):
        """A number from a chat message is no more trustworthy than a typed one."""
        intent = Intent.from_payload(payload(n_pairs=50), KEYS)
        params, _ = apply_intent(intent, DesignParameters(), KEYS)
        assert params.n_pairs == 50
        # And beyond the bound it is pulled back, with the override recorded.
        over = Intent(intent="change", n_pairs=50)
        over.n_pairs = 9999            # past the schema, as if validation were bypassed
        params, changes = apply_intent(over, DesignParameters(), KEYS)
        assert params.n_pairs == 50
        assert any("permitted range" in c for c in changes)

    def test_fewer_pairs_is_a_pair_count_not_an_action(self):
        """What `relax: reduce_pairs` used to do, said directly."""
        intent = Intent.from_payload(payload(n_pairs=4), KEYS)
        params, changes = apply_intent(intent, DesignParameters(n_pairs=5), KEYS)
        assert params.n_pairs == 4
        assert changes == ["pairs → 4"]

    def test_no_intent_field_carries_an_action(self):
        """The chat designs from pair count, spacing and channel only. An action
        field would let the model ask for relax_tm, which here would report
        "Applied" and change nothing -- or for proceed/fail, which are the
        loop's to choose. There is none, under any name."""
        props = intent_schema(KEYS, ("hg38",))["properties"]
        action_values = {a.value for a in Action}
        for name, spec in props.items():
            assert not (set(spec.get("enum", [])) & action_values), name
        with pytest.raises(ParameterRefused, match="unexpected keys"):
            Intent.from_payload(payload(relax="reduce_pairs"), KEYS)

    def test_enforce_window_can_be_switched_on_and_off(self):
        on = Intent.from_payload(payload(enforce_window="on"), KEYS)
        params, changes = apply_intent(on, DesignParameters(), KEYS)
        assert params.enforce_window is True
        assert any("enforced as a filter" in c for c in changes)
        off = Intent.from_payload(payload(enforce_window="off"), KEYS)
        back, changes = apply_intent(off, params, KEYS)
        assert back.enforce_window is False


# --- The scripted round trip ------------------------------------------------

class TestScriptedRoundTrip:
    def test_a_client_can_return_an_intent_without_a_network(self):
        client = ScriptedClient(structured_replies=[payload(system="cy5", n_pairs=6)])
        data, error, seconds = client.structured(CHAT_SYSTEM_PROMPT,
                                                 "design six pairs in cy5",
                                                 intent_schema(KEYS))
        assert not error
        intent = Intent.from_payload(data, KEYS)
        assert intent.n_pairs == 6 and intent.system == "cy5"
        assert client.structured_calls[0][1] == "design six pairs in cy5"

    def test_an_exhausted_client_reports_rather_than_inventing(self):
        data, error, _ = ScriptedClient().structured("s", "u", intent_schema(KEYS))
        assert data is None and "no structured reply" in error


# --- Pasted sequence: the user's own text, kept from the model -------------

SEQ = "ACGT" * 20


class TestPastedSequence:
    def test_inline_run_is_a_sequence_target(self):
        t = extract_target(f"design for {SEQ} in cy5")
        assert t.kind == "sequence" and t.sequence == SEQ

    def test_fasta_wrapped_over_lines_and_rna_read_as_dna(self):
        text = ">tx1 some description\n" + SEQ[:60].replace("T", "U") + "\n" + SEQ[60:] + "\n"
        assert extract_target(text).sequence == SEQ

    def test_too_short_or_prose_is_not_a_target(self):
        assert not extract_target("ACGT" * 5).found
        assert not extract_target("act cat a at").found

    def test_region_wins_over_a_sequence(self):
        assert extract_target(f"chr7:1-100 {SEQ}").kind == "genomic"

    def test_the_model_never_sees_the_sequence(self):
        red = redact_sequences(f"design for {SEQ} in cy5")
        assert SEQ[:20] not in red and "80 nt" in red

    def test_the_form_refuses_unreadable_characters(self):
        assert unreadable_sequence_chars(">h\nACGN 12") == "N"
        assert unreadable_sequence_chars(SEQ) == ""


class TestChatRequestCarriesTheSession:
    """Each chat turn is a separate model call. Without the current design in
    the request, "can you make a set of 4" after a pasted sequence read as a
    request with no target, and the model asked for one. What the context
    holds is tested in test_chat_context.py; this is the request around it."""

    SEQ = "ACGT" * 30

    def test_follow_up_sees_the_held_target_and_settings(self):
        ctx = build_chat_context("can you make a set of 4",
                                 DesignParameters(system="cy3", n_pairs=1),
                                 "pasted sequence (420 nt)")
        text = chat_request("can you make a set of 4", ctx)
        assert '"target": "pasted sequence (420 nt)"' in text
        assert '"n_pairs": 1' in text
        assert text.endswith("USER MESSAGE:\ncan you make a set of 4")

    def test_still_no_bases_reach_the_model(self):
        message = f"design 4 pairs for {self.SEQ}"
        ctx = build_chat_context(message, DesignParameters(system="cy3"),
                                 f"pasted sequence ({len(self.SEQ)} nt)")
        text = chat_request(message, ctx)
        assert self.SEQ not in text
        assert "[pasted sequence, 120 nt]" in text

    def test_prompt_tells_the_model_to_continue_from_it(self):
        assert "CURRENT DESIGN" in CHAT_SYSTEM_PROMPT
        assert "Do not ask for a target or a channel" in CHAT_SYSTEM_PROMPT
        assert "pair_details" in CHAT_SYSTEM_PROMPT


class TestChatRequestCarriesRecentTurns:
    """"no, for the specificity screen" means nothing without the turn before
    it. The model was sent only the message and answered the wrong question."""

    SEQ = "ACGT" * 30
    HISTORY = [
        {"role": "user", "text": f"this is the target {SEQ}"},
        {"role": "assistant", "text": "No specificity screen ran for 1 pair."},
        {"role": "user", "text": "how do I give that to you"},
        {"role": "assistant", "text": "Paste the DNA sequence or a region."},
    ]

    def ctx(self, **kw):
        return build_chat_context("x", DesignParameters(system="cy3"),
                                  "pasted sequence (120 nt)", **kw)

    def test_recent_turns_go_before_the_message(self):
        text = chat_request("no, for the specificity screen", self.ctx(),
                            self.HISTORY)
        recent = text.index("RECENT CONVERSATION")
        assert recent < text.index("USER MESSAGE")
        assert "user: how do I give that to you" in text
        assert "assistant: Paste the DNA sequence or a region." in text

    def test_history_is_redacted_too(self):
        text = chat_request("x", self.ctx(), self.HISTORY)
        assert self.SEQ not in text
        assert "[pasted sequence, 120 nt]" in text

    def test_history_is_bounded(self):
        long = [{"role": "user", "text": f"turn {i} " + "word " * 500}
                for i in range(20)]
        text = chat_request("x", self.ctx(), long)
        assert "turn 13 " not in text and "turn 14 " in text
        assert len(text) < 6000

    def test_no_history_no_block(self):
        assert "RECENT CONVERSATION" not in chat_request("x", self.ctx())

    def test_the_context_says_whether_the_screen_is_on_and_how(self):
        off = chat_request("x", self.ctx())
        assert '"on": false' in off and "bowtie index (optional)" in off
        on = chat_request("x", self.ctx(screen=ScreenStatus(
            on=True, against="hg38", available=["hg38 (genome)"])))
        assert '"against": "hg38"' in on and "hg38 (genome)" in on

    def test_the_prompt_explains_the_screen(self):
        assert "bowtie index (optional)" in CHAT_SYSTEM_PROMPT
        assert "RECENT CONVERSATION" in CHAT_SYSTEM_PROMPT
