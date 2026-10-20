"""Pydantic models for what crosses the model boundary.

`actions.ACTION_SCHEMA` is the hand-written JSON schema the Ollama and Anthropic
clients send. This is the same contract as a typed model, so it can be handed to
a library that wants one (PydanticAI) and so validation is declared once instead
of re-implemented in `Decision.from_payload`. `test_models.py` pins the two
together, so they cannot drift.

The shape is closed on purpose. Unknown keys are refused, `action` is the
`Action` enum, `primary_signal` is an enum, and `reason` -- the only free text --
is length-bounded and refused if it carries an oligo-length run of bases. A
model has no field that could smuggle a sequence into the run record, and no
second route in.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .actions import Action

PRIMARY_SIGNALS = ("specificity", "accessibility", "thermodynamics",
                   "composition", "geometry", "capacity")

#: Longest `reason` accepted. Two sentences is the ask; this is generous.
MAX_REASON_CHARS = 600

#: 15 nt is shorter than any pad (25) or arm, so no oligo fits under it. Same
#: threshold the MCP tests use to prove tool output carries no bases.
DNA_RUN = re.compile(r"[ACGTUacgtu]{15,}")


class DecisionModel(BaseModel):
    """One model decision, as the model must produce it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Action = Field(description="Which single step to take next.")
    reason: str = Field(
        max_length=MAX_REASON_CHARS,
        description=("One or two sentences on why, citing the evidence given. "
                     "Read by a human reviewing the run record; not interpreted "
                     "by the code."))
    primary_signal: Literal[PRIMARY_SIGNALS] = Field(  # type: ignore[valid-type]
        description="Which signal drove the decision.")

    @field_validator("reason")
    @classmethod
    def _no_sequence(cls, value: str) -> str:
        if DNA_RUN.search(value):
            raise ValueError(
                "reason contains an oligo-length run of bases; describe the "
                "evidence in words, never with a sequence")
        return value
