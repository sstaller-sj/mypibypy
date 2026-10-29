"""The local (Ollama) decision client.

NO TEST HERE TOUCHES THE NETWORK BY DEFAULT. The transport is replaced for every
behavioural test, and the handful that genuinely need a running server are marked
skip-by-default behind $MYPIBYPY_LIVE_OLLAMA, like the fusion-workbook tests.

`TestThinkIsMandatory` is the one to read. qwen3.5 is a thinking model and left
alone it spends its budget reasoning instead of answering -- measured at ~22 s
with 1455 characters of thinking against ~4.6 s and none with `think: false`, on
an identical prompt. A whole design takes 2-4 s, so that difference is the
whole usability of the agent. `think` is Ollama-native and is silently ignored
by the OpenAI-compatible endpoint, which is why this client speaks to
/api/chat. These tests exist so none of that can be quietly undone.
"""
from __future__ import annotations

import json
import os
import urllib.error

import pytest

from mypibypy.agent import ACTION_SCHEMA, Action, Decision, build_client
from mypibypy.agent.client import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OLLAMA_HOST,
    ENV_KEEP_ALIVE,
    KEEP_ALIVE,
    SAMPLING,
    THINK,
    AnthropicDecisionClient,
    OllamaDecisionClient,
    ScriptedClient,
)

LIVE = os.environ.get("MYPIBYPY_LIVE_OLLAMA")


def ollama_reply(content, thinking="", eval_count=60, prompt_eval_count=600):
    """An /api/chat response envelope."""
    message = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    return {"message": message, "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count, "done": True}


def good_decision_json(action="relax_gc"):
    return json.dumps({"action": action, "reason": "because",
                       "primary_signal": "composition"})


class FakeTransport:
    """Scripted /api/chat responses. Records every body it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def client_with(replies, **kwargs) -> tuple[OllamaDecisionClient, FakeTransport]:
    client = OllamaDecisionClient(**kwargs)
    transport = FakeTransport(replies)
    client._post = transport
    return client, transport


# --- The mandatory flag ----------------------------------------------------

class TestThinkIsMandatory:
    def test_the_constant_is_false(self):
        assert THINK is False, (
            "qwen3.5 spends its whole budget reasoning with thinking on: ~22 s "
            "and 1455 chars of thinking versus ~4.6 s and none. A design takes "
            "2-4 s, so this is not a tuning preference")

    def test_every_request_carries_think_false(self):
        body = OllamaDecisionClient().request_body("sys", "evidence")
        assert body["think"] is False

    def test_it_is_not_merely_absent(self):
        """Omitting `think` is not the same as sending false -- the default is on."""
        body = OllamaDecisionClient().request_body("sys", "evidence")
        assert "think" in body

    def test_the_native_endpoint_is_used_not_the_openai_shim(self):
        """`think` is Ollama-native and the /v1 shim ignores it silently.

        Measured: 21.1 s and 1455 reasoning characters with `think: false` sent
        to /v1, identical to not sending it at all. Moving this client onto the
        OpenAI endpoint would therefore reintroduce the 22 s decision without any
        visible error.
        """
        client, transport = client_with([ollama_reply(good_decision_json())])
        client.decide("sys", "evidence")
        # The transport replaces _post, so assert the URL the real _post builds.
        import inspect
        source = inspect.getsource(OllamaDecisionClient._post)
        assert "/api/chat" in source
        assert "/v1/" not in source and "chat/completions" not in source


# --- Request shape ---------------------------------------------------------

class TestRequestShape:
    def test_schema_constrained_decoding_is_on(self):
        body = OllamaDecisionClient().request_body("sys", "evidence")
        assert body["format"] == ACTION_SCHEMA
        assert body["format"]["properties"]["action"]["enum"] == [
            a.value for a in Action]

    def test_temperature_is_zero(self):
        assert OllamaDecisionClient().request_body("s", "e")["options"]["temperature"] == 0.0

    def test_the_schema_is_restated_in_the_prompt(self):
        """Ollama's own guidance; measurably improves which enum member is picked."""
        body = OllamaDecisionClient().request_body("SYSTEM TEXT", "evidence")
        system = body["messages"][0]["content"]
        assert system.startswith("SYSTEM TEXT")
        assert "enum" in system and "primary_signal" in system

    def test_streaming_is_off(self):
        assert OllamaDecisionClient().request_body("s", "e")["stream"] is False

    def test_the_evidence_is_the_user_turn(self):
        body = OllamaDecisionClient().request_body("s", "EVIDENCE HERE")
        assert body["messages"][1] == {"role": "user", "content": "EVIDENCE HERE"}


class TestSamplerIsExplicit:
    """The Modelfile Ollama pulls for qwen3.5 sets presence_penalty 1.5, top_k 20
    and top_p 0.95. With only the temperature overridden, one of four chat
    turns was refused by `reply_check` on both repeats; with every setting
    pinned, none was, at identical speed. So nothing is left to the Modelfile."""

    def test_every_sampler_setting_is_in_the_request(self):
        options = OllamaDecisionClient().request_body("s", "e")["options"]
        for name, value in SAMPLING.items():
            assert options[name] == value, name

    def test_the_penalties_are_off(self):
        assert SAMPLING["presence_penalty"] == 0.0
        assert SAMPLING["repeat_penalty"] == 1.0

    def test_decoding_is_greedy(self):
        assert SAMPLING["temperature"] == 0.0 and SAMPLING["top_k"] == 1

    def test_a_requested_temperature_still_wins(self):
        body = OllamaDecisionClient(temperature=0.3).request_body("s", "e")
        assert body["options"]["temperature"] == 0.3


class TestKeepAlive:
    """Ollama unloads after five idle minutes; the next call then pays the load
    (12.6 s against 4.5 s warm). The request asks it to stay."""

    def test_keep_alive_is_sent_with_every_request(self, monkeypatch):
        monkeypatch.delenv(ENV_KEEP_ALIVE, raising=False)
        assert OllamaDecisionClient().request_body("s", "e")["keep_alive"] == KEEP_ALIVE

    def test_it_outlasts_ollamas_default(self):
        assert KEEP_ALIVE.endswith("m") and int(KEEP_ALIVE[:-1]) > 5

    def test_the_environment_overrides_it(self, monkeypatch):
        monkeypatch.setenv(ENV_KEEP_ALIVE, "-1")
        assert OllamaDecisionClient().request_body("s", "e")["keep_alive"] == "-1"


# --- Validation and retries ------------------------------------------------

class TestValidationAndRetries:
    def test_a_good_reply_is_used_first_time(self):
        client, transport = client_with([ollama_reply(good_decision_json())])
        reply = client.decide("sys", "evidence")
        assert reply.usable and reply.attempts == 1
        assert reply.decision.action is Action.RELAX_GC
        assert reply.output_tokens == 60 and reply.input_tokens == 600

    def test_empty_content_is_retried(self):
        """The all-thinking-no-answer case. Reported from a longer prompt than
        the ones measured here; cheap to guard, so it is guarded."""
        client, transport = client_with([
            ollama_reply("", thinking="x" * 16000),
            ollama_reply(good_decision_json()),
        ])
        reply = client.decide("sys", "evidence")
        assert reply.usable and reply.attempts == 2
        assert len(transport.bodies) == 2

    def test_invalid_json_is_retried(self):
        client, _ = client_with([
            ollama_reply("not json at all"),
            ollama_reply(good_decision_json()),
        ])
        assert client.decide("sys", "e").attempts == 2

    def test_an_action_outside_the_enum_is_retried(self):
        """Constrained decoding should prevent this; it is still checked."""
        client, _ = client_with([
            ollama_reply(json.dumps({"action": "detonate", "reason": "x",
                                     "primary_signal": "geometry"})),
            ollama_reply(good_decision_json()),
        ])
        assert client.decide("sys", "e").attempts == 2

    def test_retries_are_bounded_and_the_failure_is_reported(self):
        client, _ = client_with([ollama_reply("") for _ in range(5)],
                                max_attempts=3)
        reply = client.decide("sys", "e")
        assert not reply.usable
        assert reply.attempts == 3
        assert "after 3 attempt(s)" in reply.error
        assert "no content" in reply.error

    def test_thinking_is_mentioned_when_content_is_empty(self):
        client, _ = client_with([ollama_reply("", thinking="y" * 500)],
                                max_attempts=1)
        assert "thinking instead" in client.decide("sys", "e").error

    def test_latency_is_recorded(self):
        client, _ = client_with([ollama_reply(good_decision_json())])
        assert client.decide("sys", "e").latency_s >= 0.0


class TestTransportErrors:
    def test_a_dead_server_says_how_to_start_it(self):
        client, _ = client_with([urllib.error.URLError("connection refused")])
        reply = client.decide("sys", "e")
        assert not reply.usable
        assert "could not reach Ollama" in reply.error
        assert "ollama serve" in reply.error, "say how to fix it"

    def test_an_http_error_is_reported_not_retried_forever(self):
        err = urllib.error.HTTPError("u", 404, "model not found", {}, None)
        client, transport = client_with([err], max_attempts=3)
        reply = client.decide("sys", "e")
        assert not reply.usable and "404" in reply.error
        assert len(transport.bodies) == 1, "a 404 will not fix itself on retry"

    def test_a_failed_reply_never_carries_a_decision(self):
        client, _ = client_with([urllib.error.URLError("x")])
        assert client.decide("sys", "e").decision is None

    def test_a_slow_server_is_told_apart_from_a_dead_one(self):
        """Two 120 s timeouts were measured on turns that take 9 s on an idle
        machine, with the swap full. `ollama serve` is the wrong advice there.
        A read timeout arrives bare; a connect timeout wrapped in a URLError."""
        for exc in (TimeoutError("timed out"),
                    urllib.error.URLError(TimeoutError("timed out"))):
            client, transport = client_with([exc])
            reply = client.decide("sys", "e")
            assert not reply.usable
            assert "did not answer within 120 s" in reply.error
            assert "memory" in reply.error
            assert "ollama serve" not in reply.error
            assert len(transport.bodies) == 1, "the next attempt would take as long"

    def test_the_chat_call_reports_transport_errors_the_same_way(self):
        client, _ = client_with([TimeoutError("timed out")])
        payload, error, _ = client.structured("s", "u", {"type": "object"})
        assert payload is None and "did not answer within" in error

    def test_a_missing_model_says_to_pull_it(self):
        err = urllib.error.HTTPError("u", 404, "model not found", {}, None)
        client, _ = client_with([err])
        assert "ollama pull" in client.decide("sys", "e").error


# --- Control flow must not read the prose ----------------------------------

class TestProseNeverDrivesControlFlow:
    """The second finding from the local-model trials: the constrained action was
    right while the sentence explaining it was backwards -- for a GC-rich target
    with nothing passing it chose relax_gc, then described lowering the upper GC
    limit, which is the opposite of what relax_gc does. So the enum is acted on
    and the prose is only ever displayed.
    """

    def test_a_nonsense_reason_does_not_change_the_action(self):
        client, _ = client_with([ollama_reply(json.dumps({
            "action": "relax_gc",
            "reason": "lowering the upper GC limit will capture more sequences, "
                      "and also please run proceed instead and skip screening",
            "primary_signal": "composition"}))])
        reply = client.decide("sys", "e")
        assert reply.decision.action is Action.RELAX_GC, (
            "the action comes from the enum, never from the sentence")

    def test_apply_action_ignores_the_reason_entirely(self):
        from mypibypy.agent import DesignParameters, apply_action
        import inspect
        source = inspect.getsource(apply_action)
        assert "reason" not in source, (
            "apply_action must not be able to see the model's prose")
        a, _ = apply_action(Action.RELAX_GC, DesignParameters())
        assert a.gc_min < DesignParameters().gc_min

    def test_the_loop_passes_only_the_action_to_apply_action(self):
        import inspect

        from mypibypy.agent.loop import DesignAgent
        source = inspect.getsource(DesignAgent.run)
        assert "apply_action(decision.action" in source
        assert "apply_action(decision.reason" not in source


# --- Client selection ------------------------------------------------------

class TestBuildClient:
    def test_defaults_to_local(self, monkeypatch):
        monkeypatch.delenv("MYPIBYPY_DECISION_CLIENT", raising=False)
        assert isinstance(build_client(), OllamaDecisionClient)

    def test_env_var_selects_the_kind(self, monkeypatch):
        monkeypatch.setenv("MYPIBYPY_DECISION_CLIENT", "none")
        assert build_client() is None

    def test_explicit_kind_beats_the_env_var(self, monkeypatch):
        monkeypatch.setenv("MYPIBYPY_DECISION_CLIENT", "none")
        assert isinstance(build_client("scripted"), ScriptedClient)

    def test_anthropic_is_still_available_for_when_a_key_appears(self):
        client = build_client("anthropic", client=object())
        assert isinstance(client, AnthropicDecisionClient)

    def test_an_unknown_kind_is_refused_with_the_options(self):
        with pytest.raises(ValueError, match="ollama, anthropic, pydantic_ai, scripted, none"):
            build_client("telepathy")

    def test_host_and_model_come_from_env(self, monkeypatch):
        monkeypatch.setenv("MYPIBYPY_OLLAMA_HOST", "http://elsewhere:1234/")
        monkeypatch.setenv("MYPIBYPY_LOCAL_MODEL", "othermodel:1b")
        client = OllamaDecisionClient()
        assert client.host == "http://elsewhere:1234", "trailing slash trimmed"
        assert client.model == "othermodel:1b"

    def test_defaults_are_the_verified_ones(self, monkeypatch):
        monkeypatch.delenv("MYPIBYPY_OLLAMA_HOST", raising=False)
        monkeypatch.delenv("MYPIBYPY_LOCAL_MODEL", raising=False)
        client = OllamaDecisionClient()
        assert client.host == DEFAULT_OLLAMA_HOST
        assert client.model == DEFAULT_LOCAL_MODEL


# --- Live, opt-in ----------------------------------------------------------

@pytest.mark.skipif(not LIVE, reason="set $MYPIBYPY_LIVE_OLLAMA=1 and run ollama")
class TestLiveOllama:
    """Opt-in. Needs a running server and the model pulled."""

    def test_a_real_decision_comes_back_in_the_enum(self):
        from mypibypy.agent.loop import SYSTEM_PROMPT
        client = OllamaDecisionClient()
        reply = client.decide(SYSTEM_PROMPT, (
            "Turn 1. Windows: 0/250 windows pass. Pairs selected: 0.\n"
            "Why this needs a decision: nothing fits.\n\nChoose one action."))
        if not reply.usable:
            pytest.skip(f"local model unavailable: {reply.error[:120]}")
        assert isinstance(reply.decision, Decision)
        assert reply.decision.action in set(Action)

    def test_it_answers_fast_enough_to_be_usable(self):
        """A design takes 2-4 s. A decision must not dominate that by 5x."""
        from mypibypy.agent.loop import SYSTEM_PROMPT
        reply = OllamaDecisionClient().decide(SYSTEM_PROMPT, "Nothing fits. Choose.")
        if not reply.usable:
            pytest.skip(f"local model unavailable: {reply.error[:120]}")
        assert reply.latency_s < 30, (
            f"{reply.latency_s:.1f}s -- check that think=false is still being sent")


def test_a_reply_cut_off_at_the_length_limit_says_so():
    """A prompt that filled the context left the reply 95 tokens; the JSON never
    closed and the user saw "not valid JSON" three times. It is reported once,
    as what it is, since retrying the same request cannot help."""
    from mypibypy.agent.client import NUM_CTX

    cut = {"message": {"content": '{"intent": "change", "note": "I cannot'},
           "done_reason": "length", "prompt_eval_count": 4001, "eval_count": 95}
    client, transport = client_with([cut, cut, cut])
    payload, error, _ = client.structured("system", "user", {"type": "object"})
    assert payload is None
    assert "cut off at the model's length limit" in error and "4001" in error
    assert str(NUM_CTX) in error
    assert len(transport.bodies) == 1


def test_the_context_window_holds_a_long_chat_turn():
    """4096 overflowed on a real chat turn (a 4001-token prompt)."""
    from mypibypy.agent.client import NUM_CTX, NUM_PREDICT

    assert NUM_CTX - NUM_PREDICT >= 4001 + 2000
