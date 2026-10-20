"""The closed action set, and the parameter bounds enforced around it.

Division of labour, and it is the whole safety story of this layer:

    the model chooses WHICH constraint gives.
    this module decides BY HOW MUCH.

The model's only output is an action name from `Action` plus a reason. It never
supplies a number, so there is no number to validate, clamp, or get wrong -- and
no field anywhere in its output schema that could carry a sequence. That is
structural, not a matter of prompt discipline: `ACTION_SCHEMA` has no free-form
string field other than `reason`, which is stored and displayed and never read
back as data.

Parameters still need bounding, because a *human* supplies the starting ones
through the CLI or the UI. `clamp` is for them. An out-of-range value is clamped
and the clamp is recorded as an event; an unknown key is refused outright. A
prompt instruction would be neither.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum

from pydantic import ValidationError


class Action(str, Enum):
    """Every move the model is permitted to make. A closed set, by construction.

    Anything not on this list cannot be requested: the structured-output schema
    enumerates exactly these values, and `Action(value)` raises on anything else.
    """

    PROCEED = "proceed"
    RELAX_TM = "relax_tm"
    RELAX_GC = "relax_gc"
    WIDEN_REGION = "widen_region"
    REDUCE_PAIRS = "reduce_pairs"
    SWITCH_SYSTEM = "switch_system"
    ACCEPT_SINGLE_ARM_HITS = "accept_single_arm_hits"
    FLAG_FOR_REVIEW = "flag_for_review"
    FAIL = "fail"


#: Actions that loosen a constraint. Each application is one "relaxation", and
#: the loop caps how many may happen -- three, then it reports failure. Without
#: that cap the loop would loosen until something passed and would therefore
#: always "succeed", which is the failure mode worth designing against.
RELAXING_ACTIONS = frozenset({
    Action.RELAX_TM, Action.RELAX_GC, Action.WIDEN_REGION,
    Action.REDUCE_PAIRS, Action.ACCEPT_SINGLE_ARM_HITS,
})

#: Actions that end the run.
TERMINAL_ACTIONS = frozenset({Action.PROCEED, Action.FLAG_FOR_REVIEW, Action.FAIL})


@dataclass(frozen=True)
class DesignParameters:
    """Everything the loop is allowed to vary. Numbers and keys only.

    There is deliberately no sequence field, no region sequence, no probe
    string. A tool that took one could be handed model-authored bases; none
    does. `agent.tools` is where that is enforced and tested.
    """

    gc_min: float = 0.40
    gc_max: float = 0.60
    tm_min: float = 55.0
    tm_max: float = 62.0
    n_pairs: int = 1
    min_spacing: int = 3
    top: int = 5
    max_mismatches: int = 3
    accept_single_arm_hits: bool = False
    widen_by_nt: int = 0
    system: str = ""          # a registry key such as "cy5"; validated against the registry
    #: Whether the GC/Tm window REJECTS windows or merely describes them.
    #:
    #: Off by default, and that default is load-bearing. `design._score`
    #: penalises GC and Tm distance but never rejects on them, so the verified
    #: designs were all produced with no hard window at all -- at the lab's own
    #: GC 0.40-0.60 / Tm 55-62, zero of 1692 windows at one real locus satisfy
    #: both, yet that locus has a working validated design. An agent that
    #: filtered by default would therefore disagree with every verified design by
    #: construction. Turn it on to impose a window deliberately; the loop then
    #: has something real to relax.
    enforce_window: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


#: (low, high) for every numeric parameter. Bounds are the lab's, not the
#: model's: GC outside 0.2-0.8 and Tm outside 40-75 C are outside anything this
#: assay has been run at, and a pad count over 50 exceeds what the published
#: method used per locus per channel.
BOUNDS: dict[str, tuple[float, float]] = {
    "gc_min": (0.20, 0.80),
    "gc_max": (0.20, 0.80),
    "tm_min": (40.0, 75.0),
    "tm_max": (40.0, 75.0),
    "n_pairs": (1, 50),
    "min_spacing": (0, 10_000),
    "top": (1, 500),
    "max_mismatches": (0, 3),      # bowtie v1 -v is exhaustive only to 3
    "widen_by_nt": (0, 100_000),
}

#: How much one application of each relaxing action gives. Fixed here so the
#: step size is a property of the code and not of whatever the model felt like
#: asking for. Deliberately small: three steps of this size is a visible,
#: explainable loosening, not a slide into "anything passes".
STEPS = {
    "tm_c": 1.5,          # widen the Tm window by this much at each end
    "gc_frac": 0.05,      # widen the GC window by this much at each end
    "widen_nt": 250,      # extend the search region by this much at each end
}


@dataclass
class ClampEvent:
    """One parameter the harness overrode, and why. Always recorded."""

    parameter: str
    requested: object
    applied: object
    reason: str

    def __str__(self) -> str:
        return (f"{self.parameter}: requested {self.requested!r}, applied "
                f"{self.applied!r} ({self.reason})")


class ParameterRefused(ValueError):
    """A parameter could not be made safe by clamping."""


def clamp(params: DesignParameters,
          known_systems: tuple[str, ...] = ()) -> tuple[DesignParameters, list[ClampEvent]]:
    """Force `params` inside `BOUNDS`. Returns the safe parameters and what changed.

    Clamping rather than rejecting, for the numeric fields, because a run that
    stops dead on a slightly-too-wide GC window helps nobody -- but silently
    honouring it would mean the bounds were decorative. So the value is pulled
    into range and the override is recorded in the run record, where it can be
    read six months later.

    Inverted windows (min above max) are a different kind of wrong: they cannot
    be clamped into something meaningful, so they are swapped and recorded.
    """
    events: list[ClampEvent] = []
    values = params.to_dict()

    for name, (low, high) in BOUNDS.items():
        current = values[name]
        if current is None:
            continue
        bounded = min(max(current, low), high)
        if isinstance(values[name], int) and not isinstance(values[name], bool):
            bounded = int(round(bounded))
        if bounded != current:
            events.append(ClampEvent(name, current, bounded,
                                     f"outside the permitted range {low}-{high}"))
            values[name] = bounded

    for lo_name, hi_name in (("gc_min", "gc_max"), ("tm_min", "tm_max")):
        if values[lo_name] > values[hi_name]:
            events.append(ClampEvent(
                f"{lo_name}/{hi_name}", (values[lo_name], values[hi_name]),
                (values[hi_name], values[lo_name]), "window was inverted; swapped"))
            values[lo_name], values[hi_name] = values[hi_name], values[lo_name]

    if values["system"] and known_systems and values["system"] not in known_systems:
        raise ParameterRefused(
            f"unknown system {values['system']!r}; configured systems are "
            f"{sorted(known_systems)}. Refused rather than clamped: there is no "
            f"nearest valid channel, and guessing one would order the wrong dye.")

    return DesignParameters(**values), events


def apply_action(action: Action, params: DesignParameters,
                 known_systems: tuple[str, ...] = ()) -> tuple[DesignParameters, str]:
    """The deterministic consequence of an action. Returns (parameters, what changed).

    This is the only place parameters move in response to a model decision, and
    every step comes from `STEPS`. The model said which constraint gives; the
    size of the give is decided here.
    """
    if action in TERMINAL_ACTIONS:
        return params, "no parameter change (terminal action)"

    if action is Action.RELAX_TM:
        step = STEPS["tm_c"]
        new = replace(params, tm_min=params.tm_min - step, tm_max=params.tm_max + step)
        change = (f"Tm window {params.tm_min:.1f}-{params.tm_max:.1f} -> "
                  f"{new.tm_min:.1f}-{new.tm_max:.1f} C")
    elif action is Action.RELAX_GC:
        step = STEPS["gc_frac"]
        new = replace(params, gc_min=params.gc_min - step, gc_max=params.gc_max + step)
        change = (f"GC window {params.gc_min:.2f}-{params.gc_max:.2f} -> "
                  f"{new.gc_min:.2f}-{new.gc_max:.2f}")
    elif action is Action.WIDEN_REGION:
        step = STEPS["widen_nt"]
        new = replace(params, widen_by_nt=params.widen_by_nt + step)
        change = f"search region widened by {new.widen_by_nt} nt each side"
    elif action is Action.REDUCE_PAIRS:
        new = replace(params, n_pairs=max(1, params.n_pairs - 1))
        change = f"pair count {params.n_pairs} -> {new.n_pairs}"
    elif action is Action.ACCEPT_SINGLE_ARM_HITS:
        new = replace(params, accept_single_arm_hits=True)
        change = ("single-arm off-target hits accepted (they cannot recruit P2 "
                  "alone, so they cannot make a spot)")
    elif action is Action.SWITCH_SYSTEM:
        others = [s for s in known_systems if s != params.system]
        if not others:
            return params, "no alternative system configured; nothing to switch to"
        new = replace(params, system=others[0])
        change = f"system {params.system or '(unset)'} -> {new.system}"
    else:  # pragma: no cover - Action is closed and every member is handled above
        raise ParameterRefused(f"no handler for action {action!r}")

    safe, events = clamp(new, known_systems)
    if events:
        change += " (then clamped: " + "; ".join(str(e) for e in events) + ")"
    return safe, change


#: The structured-output schema for a model decision.
#:
#: `action` is a closed enum. `reason` is free text because a human reads it,
#: and it is the ONLY free-text field -- it is never parsed, matched, or fed
#: back into a tool, so it cannot carry an instruction or a sequence into the
#: deterministic layer. There is no field for parameter values at all.
ACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [a.value for a in Action],
            "description": "Which single step to take next.",
        },
        "reason": {
            "type": "string",
            "description": (
                "One or two sentences on why, citing the evidence given. Read by "
                "a human reviewing the run record; not interpreted by the code."
            ),
        },
        "primary_signal": {
            "type": "string",
            "enum": ["specificity", "accessibility", "thermodynamics",
                     "composition", "geometry", "capacity"],
            "description": "Which signal drove the decision.",
        },
    },
    "required": ["action", "reason", "primary_signal"],
    "additionalProperties": False,
}


@dataclass
class Decision:
    """A parsed model decision."""

    action: Action
    reason: str
    primary_signal: str
    raw: dict = field(default_factory=dict)
    source: str = "model"      # "model" | "deterministic" | "fallback"

    @staticmethod
    def from_payload(payload: dict, source: str = "model") -> Decision:
        """Validate a decision payload. Raises on anything outside the schema."""
        missing = {"action", "reason", "primary_signal"} - set(payload)
        if missing:
            raise ParameterRefused(f"decision is missing {sorted(missing)}")
        extra = set(payload) - {"action", "reason", "primary_signal"}
        if extra:
            # Refused, not ignored: an unexpected key means the model is trying
            # to say something this layer has not agreed to hear.
            raise ParameterRefused(
                f"decision carries unexpected keys {sorted(extra)}; the action "
                f"schema is closed")
        try:
            action = Action(payload["action"])
        except ValueError as exc:
            raise ParameterRefused(
                f"{payload['action']!r} is not one of "
                f"{[a.value for a in Action]}") from exc
        # The checks above give the readable messages; the typed model adds the
        # rest of the contract (signal enum, bounded reason, no bases in it) so
        # every client is held to it, not only the one built on pydantic.
        from .models import DecisionModel
        try:
            checked = DecisionModel.model_validate(payload)
        except ValidationError as exc:
            raise ParameterRefused(
                "decision failed validation: "
                + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                            for e in exc.errors())) from exc
        return Decision.from_model(checked, raw=dict(payload), source=source)

    @staticmethod
    def from_model(model, raw: dict | None = None, source: str = "model") -> Decision:
        """A `Decision` from an already-validated `models.DecisionModel`."""
        return Decision(action=model.action, reason=model.reason,
                        primary_signal=model.primary_signal,
                        raw=raw if raw is not None else model.model_dump(mode="json"),
                        source=source)
