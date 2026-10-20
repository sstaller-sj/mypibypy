"""A `DecisionClient` built on PydanticAI.

This is the third client behind the same seam as the Anthropic and Ollama ones;
the loop cannot tell them apart. What PydanticAI adds is the validate-and-retry
cycle: the reply is parsed into `models.DecisionModel`, and a reply that fails
validation is sent back to the model with the errors. Everything the loop relies
on stays outside it:

* **Retries stay bounded.** `max_attempts` is both the output-retry allowance and
  the hard `request_limit`, so one `decide` is at most that many requests, as
  with the Ollama client. The loop's model-call budget still counts one per
  `decide`.
* **Nothing is thrown.** Any failure -- unreachable server, exhausted retries,
  an HTTP error -- comes back as `ModelReply(error=...)`, which the loop turns
  into "flag for review". The model stays an advisor.
* **Tools are not registered.** The agent has no tools, so the only thing a
  model can do is return a `DecisionModel`.

The optional dependency is imported lazily: `pip install -e ".[pydantic-ai]"`.

Ollama specifics, measured on qwen3.5:9b (see `client.THINK`): PydanticAI reaches
Ollama through the OpenAI-compatible `/v1` endpoint, where the native `think`
flag is ignored (15.6 s, 1333 reasoning characters). `reasoning_effort: "none"`
is honoured there (0.5 s, none), so it is sent as `extra_body`. Two native
options have no `/v1` equivalent and are therefore not applied here: `num_ctx`
(the server default context is used; the evidence summary is a few hundred
tokens) and schema-constrained sampling beyond what `response_format` gives.
"""
from __future__ import annotations

import os
import time

from .actions import Decision
from .client import (DEFAULT_LOCAL_MODEL, DEFAULT_OLLAMA_HOST, ENV_LOCAL_MODEL,
                     ENV_OLLAMA_HOST, NUM_PREDICT, ModelReply)
from .models import DecisionModel

#: Sent as `extra_body`. See the module docstring for why it is not `think`.
OLLAMA_REASONING_OFF = {"reasoning_effort": "none"}


class PydanticAIDecisionClient:
    """Decision client over PydanticAI. `model` is a name or a PydanticAI `Model`.

    A string is taken as an Ollama model name and reached at `host`. Passing a
    `Model` instance uses it as given -- that is how the tests run it, with
    PydanticAI's own `FunctionModel`, and how another provider is selected.
    """

    def __init__(self, model=None, host: str | None = None,
                 timeout: float = 120.0, max_attempts: int = 3,
                 temperature: float = 0.0, native_output: bool = True,
                 extra_body: dict | None = None):
        # The library prints a banner on first import; not wanted in a CLI/UI run.
        os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
        try:
            from pydantic_ai import Agent, NativeOutput  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "the PydanticAI client needs the optional dependency: "
                "pip install -e '.[pydantic-ai]'") from exc

        self.max_attempts = max(1, max_attempts)
        self.timeout = timeout
        self.temperature = temperature
        self.native_output = native_output
        self.host = (host or os.environ.get(ENV_OLLAMA_HOST,
                                            DEFAULT_OLLAMA_HOST)).rstrip("/")
        if model is None or isinstance(model, str):
            model_name = model or os.environ.get(ENV_LOCAL_MODEL, DEFAULT_LOCAL_MODEL)
            self.model = self._ollama_model(model_name)
            self.extra_body = (OLLAMA_REASONING_OFF if extra_body is None
                               else extra_body)
        else:
            model_name = getattr(model, "model_name", type(model).__name__)
            self.model = model
            self.extra_body = extra_body or {}
        self.name = f"pydantic-ai:{model_name}"

    def _ollama_model(self, model_name: str):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        # Ollama ignores the key, but the client library insists on one.
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=f"{self.host}/v1", api_key="ollama"))

    # -- request -----------------------------------------------------------
    def settings(self) -> dict:
        """The model settings sent with each request. Separate so tests can read them."""
        settings = {"temperature": self.temperature, "max_tokens": NUM_PREDICT,
                    "timeout": self.timeout}
        if self.extra_body:
            settings["extra_body"] = dict(self.extra_body)
        return settings

    def build_agent(self, system: str):
        from pydantic_ai import Agent, NativeOutput
        output = NativeOutput(DecisionModel) if self.native_output else DecisionModel
        return Agent(self.model, output_type=output, instructions=system,
                     retries={"tools": 0, "output": self.max_attempts - 1})

    # -- call --------------------------------------------------------------
    def decide(self, system: str, evidence: str) -> ModelReply:
        from pydantic_ai.usage import UsageLimits
        started = time.time()
        try:
            result = self.build_agent(system).run_sync(
                evidence, model_settings=self.settings(),
                usage_limits=UsageLimits(request_limit=self.max_attempts))
        except Exception as exc:  # noqa: BLE001 -- the contract is "never raises"
            # The boundary the loop relies on: whatever went wrong, the reply is
            # data, and the run degrades to a human decision.
            return ModelReply(decision=None, model=self.name,
                              latency_s=time.time() - started,
                              error=f"{type(exc).__name__}: {exc}")

        usage = result.usage
        return ModelReply(
            decision=Decision.from_model(result.output, source="model"),
            model=self.name, attempts=max(1, usage.requests),
            latency_s=time.time() - started,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0)
