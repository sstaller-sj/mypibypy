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


# --- the PydanticAI client ---------------------------------------------------

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import ModelResponse, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from mypibypy.agent import DesignAgent, DesignParameters, build_client  # noqa: E402
from mypibypy.agent.pydantic_ai_client import PydanticAIDecisionClient  # noqa: E402


def scripted(*payloads):
    """A PydanticAI model that answers with each payload in turn, via the output tool."""
    queue, calls = list(payloads), []

    def fn(messages, info):
        calls.append(info)
        payload = queue.pop(0)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

    return FunctionModel(fn), calls


def client_for(model, **kw):
    return PydanticAIDecisionClient(model=model, native_output=False, **kw)


class TestPydanticAIClient:
    def test_a_valid_reply_becomes_a_decision(self):
        model, _ = scripted(GOOD)
        reply = client_for(model).decide("system", "evidence")
        assert reply.usable and reply.decision.action is Action.RELAX_TM
        assert reply.attempts == 1 and reply.error == ""

    def test_an_invalid_reply_is_sent_back_and_retried(self):
        model, calls = scripted({**GOOD, "action": "order_probes"}, GOOD)
        reply = client_for(model).decide("system", "evidence")
        assert reply.usable and reply.attempts == 2 and len(calls) == 2

    def test_retries_are_bounded_and_failure_is_a_reply_not_an_exception(self):
        model, calls = scripted(*[{**GOOD, "action": "nope"}] * 10)
        reply = client_for(model, max_attempts=2).decide("system", "evidence")
        assert not reply.usable and reply.error
        assert len(calls) <= 2

    def test_a_sequence_in_the_reason_is_never_accepted(self):
        bad = {**GOOD, "reason": "ACGTACGTACGTACGTACGTACGT"}
        model, _ = scripted(bad, bad, bad)
        reply = client_for(model).decide("system", "evidence")
        assert not reply.usable

    def test_the_agent_has_no_tools_beyond_the_output(self):
        model, calls = scripted(GOOD)
        client_for(model).decide("system", "evidence")
        assert [t.name for t in calls[0].function_tools] == []

    def test_ollama_thinking_is_switched_off_through_v1(self):
        client = PydanticAIDecisionClient()
        assert client.settings()["extra_body"] == {"reasoning_effort": "none"}
        assert client.name == "pydantic-ai:qwen3.5:9b"

    def test_unreachable_server_degrades_instead_of_raising(self):
        client = PydanticAIDecisionClient(host="http://127.0.0.1:9", timeout=2,
                                          max_attempts=1)
        reply = client.decide("system", "evidence")
        assert not reply.usable and reply.error

    def test_build_client_selects_it(self):
        assert isinstance(build_client("pydantic_ai"), PydanticAIDecisionClient)
