"""The pydantic decision model and the PydanticAI client built on it."""
import pytest
from pydantic import ValidationError

from mypibypy.agent import ACTION_SCHEMA, Action, Decision, ParameterRefused
from mypibypy.agent.models import DecisionModel

GOOD = {"action": "relax_tm", "reason": "Tm is limiting.", "primary_signal": "thermodynamics"}


class TestDecisionModel:
    def test_schema_matches_the_hand_written_one(self):
        """One contract, two spellings: they must not drift."""
        schema = DecisionModel.model_json_schema()
        props = schema["properties"]
        assert set(props) == set(ACTION_SCHEMA["properties"])
        assert set(schema["required"]) == set(ACTION_SCHEMA["required"])
        assert schema["additionalProperties"] is False
        assert {a.value for a in Action} == set(ACTION_SCHEMA["properties"]["action"]["enum"])
        sig = props["primary_signal"]["enum"]
        assert set(sig) == set(ACTION_SCHEMA["properties"]["primary_signal"]["enum"])

    def test_unknown_key_refused(self):
        with pytest.raises(ValidationError):
            DecisionModel.model_validate({**GOOD, "tm_min": 40})

    def test_unknown_action_refused(self):
        with pytest.raises(ValidationError):
            DecisionModel.model_validate({**GOOD, "action": "order_probes"})

    def test_reason_cannot_carry_a_sequence(self):
        with pytest.raises(ValidationError, match="oligo-length"):
            DecisionModel.model_validate({**GOOD, "reason": "use ACGTACGTACGTACGTACGTA"})

    def test_reason_is_length_bounded(self):
        with pytest.raises(ValidationError):
            DecisionModel.model_validate({**GOOD, "reason": "x" * 601})

    def test_from_payload_applies_the_typed_rules_to_every_client(self):
        with pytest.raises(ParameterRefused, match="primary_signal"):
            Decision.from_payload({**GOOD, "primary_signal": "vibes"})
        with pytest.raises(ParameterRefused, match="oligo-length"):
            Decision.from_payload({**GOOD, "reason": "ACGTACGTACGTACGTACGTA"})
        assert Decision.from_payload(GOOD).action is Action.RELAX_TM

