"""Model clients for the decision step.

The loop depends on `DecisionClient`, not on any vendor, so the endpoint is
swappable: `AnthropicDecisionClient` talks to the Claude API (or any
Anthropic-compatible gateway via `base_url`), and `ScriptedClient` returns
canned decisions for tests. Nothing in `loop` imports `anthropic`.

Three things in the Anthropic path are not optional and are easy to get wrong:

1. **`stop_reason == "refusal"` is checked BEFORE `response.content` is read.**
   A refusal returns HTTP 200 with content that may not hold what you expect, so
   reading content first and checking afterwards is the wrong order. Probe design
   is routine laboratory work, but the prompt talks about oligonucleotides
   targeting human genes, which is close enough to the biology classifier's
   territory that refusals must be handled as a normal branch rather than an
   exception.
2. **Server-side fallback is opted into** (`fallbacks="default"` with the
   matching beta), so a policy decline is re-run on a fallback model inside the
   same call rather than failing the run.
3. **The action enum arrives as structured output**, not as prose to be parsed.

A refusal that survives the fallback chain is not an error here: it returns
`None`, and the loop falls back to its deterministic default. The model is an
advisor in this design, so losing it degrades the run rather than ending it.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .actions import ACTION_SCHEMA, Decision, ParameterRefused

#: Default model. Pinned rather than "latest" so a run record names something
#: reproducible; override per-client when the lab wants a cheaper or local one.
DEFAULT_MODEL = "claude-opus-5-5"

#: Opus 5.5 defaults to `medium` effort, unlike its predecessor. Set explicitly
#: so a model change does not silently alter how hard it thinks about a routing
#: decision. `low` is plenty for picking one of nine actions from a prepared
#: evidence summary.
DEFAULT_EFFORT = "low"

#: Opt into server-side fallback on a policy decline. The scalar "default" form
#: routes by refusal category, so no model list has to be maintained here.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ModelReply:
    """What a decision call returned, including the shape of a non-answer."""

    decision: Decision | None
    refused: bool = False
    refusal_category: str = ""
    refusal_explanation: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: str = ""
    attempts: int = 1
    latency_s: float = 0.0

    @property
    def usable(self) -> bool:
        return self.decision is not None


class DecisionClient(Protocol):
    """Anything that can turn an evidence summary into one `Decision`."""

    name: str

    def decide(self, system: str, evidence: str) -> ModelReply:
        ...


class ScriptedClient:
    """A canned client for tests and dry runs. Makes no network calls.

    Every test in this package uses this. The real client is exercised only by
    shape assertions on the request it would build, because no credentials are
    configured in this environment and a test suite that needs them is a test
    suite that stops working.
    """

    name = "scripted"

    def __init__(self, decisions: list[dict] | None = None,
                 refuse: bool = False, error: str = "",
                 structured_replies: list[dict] | None = None):
        self._decisions = list(decisions or [])
        self.refuse = refuse
        self.error = error
        self._structured = list(structured_replies or [])
        self.calls: list[tuple[str, str]] = []
        self.structured_calls: list[tuple[str, str, dict]] = []

    def structured(self, system: str, user: str, schema: dict,
                   ) -> tuple[dict | None, str, float]:
        self.structured_calls.append((system, user, schema))
        if self.error:
            return None, self.error, 0.0
        if not self._structured:
            return None, "scripted client has no structured reply queued", 0.0
        return self._structured.pop(0), "", 0.0

    def decide(self, system: str, evidence: str) -> ModelReply:
        self.calls.append((system, evidence))
        if self.error:
            return ModelReply(decision=None, error=self.error, model=self.name)
        if self.refuse:
            return ModelReply(decision=None, refused=True,
                              refusal_category="bio",
                              refusal_explanation="scripted refusal",
                              model=self.name)
        if not self._decisions:
            return ModelReply(decision=None, error="scripted client exhausted",
                              model=self.name)
        payload = self._decisions.pop(0)
        return ModelReply(decision=Decision.from_payload(payload, source="model"),
                          input_tokens=len(evidence) // 4, output_tokens=24,
                          model=self.name)


class AnthropicDecisionClient:
    """Claude API client for the decision step.

    `base_url` is passed through, so an Anthropic-compatible gateway or proxy can
    be used instead of the public API. A local OpenAI-compatible server speaks a
    different protocol and needs its own `DecisionClient` implementation rather
    than a base_url override -- the seam is here precisely so that can be added
    without touching the loop.
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
                 base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 1024, timeout: float = 60.0,
                 use_fallbacks: bool = True, client=None):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_fallbacks = use_fallbacks
        self.name = f"anthropic:{model}"
        self._client = client
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on install
                raise RuntimeError(
                    "the anthropic package is not installed; install it or use a "
                    "different DecisionClient") from exc
            kwargs = {"timeout": timeout}
            if base_url or os.environ.get("ANTHROPIC_BASE_URL"):
                kwargs["base_url"] = base_url or os.environ["ANTHROPIC_BASE_URL"]
            if api_key:
                kwargs["api_key"] = api_key
            # No api_key argument means the SDK resolves credentials itself:
            # ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN, then an `ant auth
            # login` profile. Do not pre-empt that with a key check here.
            self._client = anthropic.Anthropic(**kwargs)

    def request_kwargs(self, system: str, evidence: str) -> dict:
        """The exact request body. Separated out so tests can assert its shape
        without a network call or a credential."""
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": evidence}],
            # Structured output: the action is an enum the API enforces, not
            # prose this code has to parse out of a paragraph.
            "output_config": {
                "format": {"type": "json_schema", "schema": ACTION_SCHEMA},
                "effort": self.effort,
            },
        }
        if self.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def structured(self, system: str, user: str, schema: dict,
                   ) -> tuple[dict | None, str, float]:
        """Schema-constrained call for non-decision work (chat intents)."""
        import anthropic

        kwargs = self.request_kwargs(system, user)
        kwargs["output_config"]["format"]["schema"] = schema
        started = time.time()
        try:
            response = (self._client.beta.messages.create(**kwargs)
                        if self.use_fallbacks else
                        self._client.messages.create(**kwargs))
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            return None, str(exc), time.time() - started
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            return None, (f"model declined (category "
                          f"{getattr(details, 'category', '') or 'unspecified'})"), \
                time.time() - started
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text), "", time.time() - started
        except json.JSONDecodeError as exc:
            return None, f"structured output was not valid JSON: {exc}", \
                time.time() - started

    def decide(self, system: str, evidence: str) -> ModelReply:
        import anthropic

        kwargs = self.request_kwargs(system, evidence)
        try:
            if self.use_fallbacks:
                response = self._client.beta.messages.create(**kwargs)
            else:
                kwargs.pop("betas", None)
                kwargs.pop("fallbacks", None)
                response = self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            return ModelReply(decision=None, model=self.model,
                              error=f"API error {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            return ModelReply(decision=None, model=self.model,
                              error=f"could not reach the API: {exc}")

        usage = getattr(response, "usage", None)
        reply = ModelReply(
            decision=None,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=getattr(response, "model", self.model),
        )

        # CHECK REFUSAL BEFORE READING CONTENT. On a refusal the content is not
        # the decision, and `stop_details` is populated only in this case.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            reply.refused = True
            reply.refusal_category = getattr(details, "category", "") or ""
            reply.refusal_explanation = getattr(details, "explanation", "") or ""
            return reply

        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text.strip():
            reply.error = "model returned no text content"
            return reply
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            reply.error = f"structured output was not valid JSON: {exc}"
            return reply
        try:
            reply.decision = Decision.from_payload(payload, source="model")
        except ParameterRefused as exc:
            # The schema is enforced server-side, so this should not happen --
            # but a decision outside the closed set is refused here too rather
            # than trusted because the API said it was valid.
            reply.error = f"decision rejected by the harness: {exc}"
        return reply


# --- Local models ----------------------------------------------------------
#
# WHY LOCAL IS THE WORKING PATH, not a fallback. The lab's Anthropic account is a
# Claude Enterprise seat, which covers chat and Claude Code but not the API, and
# a managed user in a HIPAA-tier organisation cannot self-provision an API key.
# So there is no key to use. `AnthropicDecisionClient` stays for when one
# appears; `OllamaDecisionClient` is what actually runs.

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_LOCAL_MODEL = "qwen3.5:9b"

ENV_OLLAMA_HOST = "MYPIBYPY_OLLAMA_HOST"
ENV_LOCAL_MODEL = "MYPIBYPY_LOCAL_MODEL"
ENV_CLIENT_KIND = "MYPIBYPY_DECISION_CLIENT"

#: MANDATORY, and measured. qwen3.5 is a thinking model: left to itself it
#: spends its budget reasoning instead of answering. On one identical prompt,
#: warm, native endpoint:
#:
#:     think=false : ~4.6 s,  96 eval tokens, 0 thinking chars, 5/5 usable
#:     default     : ~22 s,   74 eval tokens, 1455 thinking chars
#:
#: A whole design takes 2-4 s, so a 22 s decision makes the agent useless, and
#: the failure mode reported from a longer prompt -- all thinking, no content at
#: all -- is worse than slow. `think` is an OLLAMA-NATIVE parameter and is
#: SILENTLY IGNORED by the OpenAI-compatible shim: measured at 21.1 s with 1455
#: reasoning characters whether it was sent or not, i.e. identical to omitting
#: it. That is the whole reason this client speaks to /api/chat rather than /v1.
#: Do not "simplify" this onto the OpenAI endpoint; there is a test.
THINK = False


#: Context window, prompt and reply together. Was 4096, measured too small: a
#: chat turn after ten lines of conversation sent a 4001-token prompt, leaving
#: the reply 95 tokens -- cut off mid-`note` (done_reason "length"), so the JSON
#: never closed and the user saw "not valid JSON" three times. The chat prompt
#: alone measures 2,900-3,400 tokens with a design held. 8192 keeps the KV cache
#: bounded (the reason for setting one at all) with room for history and reply.
NUM_CTX = 8192
NUM_PREDICT = 512

#: Sampling, set explicitly rather than inherited from the Modelfile. The one
#: Ollama ships for qwen3.5 carries `presence_penalty 1.5`, `top_k 20` and
#: `top_p 0.95`, and this client used to override only the temperature. A
#: presence penalty still moves logits under greedy decoding and under the
#: `format` grammar, and it penalises tokens seen in the last 64 tokens of
#: context -- at the start of a reply, the end of the prompt: the very numbers
#: `note` is told to repeat as written. Measured on the same four chat turns,
#: twice each, idle machine: with the stock options "why did I only get 4?" was
#: refused by `reply_check` on both repeats (it said windows were "rejected"
#: while the window was not enforced); with these, every reply passed first
#: time. Speed was identical (8.6 s against 8.7 s per turn), so this is a
#: correctness setting, not a tuning preference.
SAMPLING = {"temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "presence_penalty": 0.0, "repeat_penalty": 1.0}

#: How long Ollama keeps the model resident after a request. The server default
#: is five minutes, after which the next call pays the load again: 12.6 s against
#: 4.5 s warm, measured. Sent per request, so no server configuration is needed:
#: a working session stays warm and the 5.7 GB is released afterwards. Override
#: with $MYPIBYPY_OLLAMA_KEEP_ALIVE (Ollama's duration syntax; "0" unloads at
#: once, "-1" pins the model).
KEEP_ALIVE = "30m"
ENV_KEEP_ALIVE = "MYPIBYPY_OLLAMA_KEEP_ALIVE"


def transport_error(host: str, model: str, timeout: float, exc: BaseException) -> str:
    """What went wrong with the HTTP call, and what to do about it.

    Three different failures used to share one message ending in `ollama
    serve`: a server that is not running, a model that is not pulled, and a
    server that is up but did not answer in time. The last is the common case
    on a laptop whose memory is full -- measured as two 120 s timeouts on chat
    turns that take 9 s on an idle machine -- and telling that user to start
    the server sends them the wrong way.
    """
    if isinstance(exc, urllib.error.HTTPError):
        hint = (f"; is the model pulled? `ollama pull {model}`" if exc.code == 404
                else "")
        return f"{host} returned HTTP {exc.code}: {exc.reason}{hint}"
    reason = getattr(exc, "reason", exc)
    if (isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError)
            or "timed out" in str(exc).lower()):
        return (f"Ollama at {host} did not answer within {timeout:.0f} s. The server "
                f"is up but slow: usually the machine is short of memory (check "
                f"`ollama ps` and swap, and close other model or genome-index "
                f"processes), or {model} is still loading")
    return (f"could not reach Ollama at {host} ({exc}). Is the server running? "
            f"`ollama serve`, then `ollama pull {model}`")


class OllamaDecisionClient:
    """Decision client for a local Ollama model, over the native /api/chat API.

    Native rather than the OpenAI-compatible endpoint because `think` only works
    there -- see `THINK`. Schema-constrained decoding is used (`format` set to
    the action schema), temperature is 0, and the schema is repeated in the
    prompt as Ollama's own guidance suggests. Every response is validated and a
    malformed or empty one is retried a bounded number of times.

    Stdlib HTTP on purpose: no new dependency for one POST.
    """

    def __init__(self, model: str | None = None, host: str | None = None,
                 timeout: float = 120.0, max_attempts: int = 3,
                 temperature: float = 0.0, keep_alive: str | None = None):
        self.model = model or os.environ.get(ENV_LOCAL_MODEL, DEFAULT_LOCAL_MODEL)
        self.host = (host or os.environ.get(ENV_OLLAMA_HOST,
                                            DEFAULT_OLLAMA_HOST)).rstrip("/")
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.temperature = temperature
        self.keep_alive = keep_alive or os.environ.get(ENV_KEEP_ALIVE, KEEP_ALIVE)
        self.name = f"ollama:{self.model}"

    # -- request -----------------------------------------------------------
    def system_prompt(self, system: str, schema: dict | None = None) -> str:
        """The system prompt with the schema appended.

        Ollama's own guidance is to restate the schema in the prompt as well as
        passing it in `format`; constrained decoding gets the shape right either
        way, but restating it measurably improves which enum member is chosen.
        """
        return (f"{system}\n\nReply with JSON matching exactly this schema:\n"
                f"{json.dumps(schema or ACTION_SCHEMA)}")

    def request_body(self, system: str, evidence: str,
                     schema: dict | None = None) -> dict:
        """The exact POST body. Separated so tests can assert it without a server."""
        return {
            "model": self.model,
            "stream": False,
            # Schema-constrained decoding: the enum is enforced by the sampler,
            # not hoped for.
            "format": schema or ACTION_SCHEMA,
            # See THINK. Non-negotiable.
            "think": THINK,
            # Every sampler setting pinned, not inherited (see SAMPLING), and
            # bounded on purpose: an unbounded context allocates a large KV
            # cache, and unbounded generation can run until the timeout.
            "options": {**SAMPLING, "temperature": self.temperature,
                        "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
            # See KEEP_ALIVE: the model stays resident between turns.
            "keep_alive": self.keep_alive,
            "messages": [
                {"role": "system", "content": self.system_prompt(system, schema)},
                {"role": "user", "content": evidence},
            ],
        }

    # -- call --------------------------------------------------------------
    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def structured(self, system: str, user: str, schema: dict,
                   ) -> tuple[dict | None, str, float]:
        """One schema-constrained call returning raw JSON. (payload, error, seconds).

        Used for anything that is not a design decision -- currently turning a
        chat request into parameters. Validation of the payload belongs to the
        caller, which knows the schema's meaning; this only guarantees it parsed.
        """
        body = self.request_body(system, user, schema=schema)
        started = time.time()
        last = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                payload = self._post(body)
            except (urllib.error.URLError, OSError) as exc:
                return None, transport_error(self.host, self.model, self.timeout, exc), \
                    time.time() - started
            content = ((payload.get("message") or {}).get("content") or "").strip()
            if payload.get("done_reason") == "length":
                # Said for what it is: retrying the same request cannot help.
                return None, (
                    f"the reply was cut off at the model's length limit (prompt "
                    f"{payload.get('prompt_eval_count', '?')} tokens, context "
                    f"{NUM_CTX}, reply limit {NUM_PREDICT})"), time.time() - started
            if not content:
                last = f"attempt {attempt}: no content returned"
                continue
            try:
                return json.loads(content), "", time.time() - started
            except json.JSONDecodeError as exc:
                last = f"attempt {attempt}: not valid JSON ({exc})"
        return None, (f"no usable reply after {self.max_attempts} attempt(s). "
                      f"Last: {last}"), time.time() - started

    def decide(self, system: str, evidence: str) -> ModelReply:
        body = self.request_body(system, evidence)
        last_error = ""
        started = time.time()

        for attempt in range(1, self.max_attempts + 1):
            try:
                payload = self._post(body)
            except (urllib.error.URLError, OSError) as exc:
                # HTTPError is a URLError; `transport_error` tells a dead
                # server, a missing model and a slow server apart. None of them
                # is retried: the next attempt would end the same way.
                return ModelReply(decision=None, model=self.model, attempts=attempt,
                                  latency_s=time.time() - started,
                                  error=transport_error(self.host, self.model,
                                                        self.timeout, exc))
            except json.JSONDecodeError as exc:
                last_error = f"Ollama response was not JSON: {exc}"
                continue

            message = payload.get("message") or {}
            content = (message.get("content") or "").strip()
            tokens = payload.get("eval_count") or 0
            prompt_tokens = payload.get("prompt_eval_count") or 0

            # The empty-content case: all thinking and no answer. Reported from a
            # longer prompt than the ones measured here, and cheap to guard, so
            # it is retried rather than trusted not to happen.
            if not content:
                last_error = (f"attempt {attempt}: model returned no content"
                              + (f" ({len(message.get('thinking') or '')} chars of "
                                 f"thinking instead)"
                                 if message.get("thinking") else ""))
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = f"attempt {attempt}: content was not valid JSON ({exc})"
                continue
            try:
                decision = Decision.from_payload(parsed, source="model")
            except ParameterRefused as exc:
                # Constrained decoding should make this impossible; it is still
                # checked, because "the sampler guaranteed it" is not a reason to
                # trust an action this layer has not agreed to.
                last_error = f"attempt {attempt}: {exc}"
                continue

            return ModelReply(decision=decision, model=self.model, attempts=attempt,
                              latency_s=time.time() - started,
                              input_tokens=prompt_tokens, output_tokens=tokens)

        return ModelReply(decision=None, model=self.model,
                          attempts=self.max_attempts,
                          latency_s=time.time() - started,
                          error=f"no usable decision after {self.max_attempts} "
                                f"attempt(s). Last: {last_error}")


def build_client(kind: str | None = None, **kwargs) -> DecisionClient | None:
    """Pick a decision client from configuration. Defaults to the local model.

    `kind` comes from the argument, then ``$MYPIBYPY_DECISION_CLIENT``, then
    "ollama". Swapping to a future Anthropic key, or to a different local model,
    is therefore configuration rather than a code change.

    "none" returns None, which the loop treats as "flag for review instead of
    guessing" -- a legitimate way to run the pipeline with no model at all.
    """
    kind = (kind or os.environ.get(ENV_CLIENT_KIND) or "ollama").lower()
    if kind in ("none", "off", "disabled"):
        return None
    if kind in ("ollama", "local"):
        return OllamaDecisionClient(**kwargs)
    if kind in ("anthropic", "claude", "api"):
        return AnthropicDecisionClient(**kwargs)
    if kind in ("pydantic_ai", "pydantic-ai", "pydanticai"):
        from .pydantic_ai_client import PydanticAIDecisionClient
        return PydanticAIDecisionClient(**kwargs)
    if kind == "scripted":
        return ScriptedClient(**kwargs)
    raise ValueError(
        f"unknown decision client {kind!r}; choose from ollama, anthropic, "
        f"pydantic_ai, scripted, none")
