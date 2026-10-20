"""The run's state machine: which step may follow which, written down once.

`DesignAgent.run` used to hold this implicitly, in the order of its `if`
branches. That worked while one loop was the only driver, but "what may happen
next" is the thing a compliance argument rests on, so it is a table here and the
loop is a driver over it:

    INTAKE --clamped--> DESIGN --designed--> EVALUATE
    EVALUATE --clear_pass--> SUCCEEDED
    EVALUATE --clear_fail--> FAILED
    EVALUATE --needs_judgement--> DECIDE
    DECIDE --proceed / flag / fail--> SUCCEEDED / FLAGGED / FAILED
    DECIDE --apply--> RELAX --applied--> DESIGN          (the loop)
    RELAX --refused--> FAILED
    any live state --exhausted--> FAILED

Two things are enforced here rather than in the caller, because they are the
two ways a run could be made to "succeed" by not stopping:

* an event that is not in the table for the current state raises
  `IllegalTransition`. Nothing can skip SCREEN by jumping from DESIGN to a
  verdict, and nothing leaves a terminal state.
* `apply` of a relaxing action is refused with `BudgetExceeded` once the
  relaxation budget is spent, and re-entering DESIGN is refused once the turn
  budget is. The caller turns that into a reported failure.

The machine holds counters and a trace, and nothing else: it neither designs
nor decides. The same table is meant to gate MCP tools (a run id plus its
state), so both drivers obey one definition of the legal order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import record as rec


class State(str, Enum):
    INTAKE = "intake"            # parameters supplied, not yet bounded
    DESIGN = "design"            # design + screen the candidates
    EVALUATE = "evaluate"        # classify the evidence, no judgement yet
    DECIDE = "decide"            # a genuine tradeoff; the model may be asked
    RELAX = "relax"              # apply the chosen action to the parameters
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    FLAGGED = "flagged"          # stopped for a human


class Event(str, Enum):
    CLAMPED = "clamped"
    DESIGNED = "designed"
    CLEAR_PASS = "clear_pass"
    CLEAR_FAIL = "clear_fail"
    NEEDS_JUDGEMENT = "needs_judgement"
    PROCEED = "proceed"
    FAIL = "fail"
    FLAG = "flag"
    APPLY = "apply"              # a non-terminal action; relaxing ones are budgeted
    APPLIED = "applied"
    REFUSED = "refused"          # the action could not be applied
    EXHAUSTED = "exhausted"      # a hard budget ran out


TERMINAL = frozenset({State.SUCCEEDED, State.FAILED, State.FLAGGED})

#: The whole legal order. Anything not here is refused.
TRANSITIONS: dict[tuple[State, Event], State] = {
    (State.INTAKE, Event.CLAMPED): State.DESIGN,
    (State.DESIGN, Event.DESIGNED): State.EVALUATE,
    (State.EVALUATE, Event.CLEAR_PASS): State.SUCCEEDED,
    (State.EVALUATE, Event.CLEAR_FAIL): State.FAILED,
    (State.EVALUATE, Event.NEEDS_JUDGEMENT): State.DECIDE,
    (State.DECIDE, Event.PROCEED): State.SUCCEEDED,
    (State.DECIDE, Event.FAIL): State.FAILED,
    (State.DECIDE, Event.FLAG): State.FLAGGED,
    (State.DECIDE, Event.APPLY): State.RELAX,
    (State.RELAX, Event.APPLIED): State.DESIGN,
    (State.RELAX, Event.REFUSED): State.FAILED,
    # A budget can end the run from any live state.
    **{(s, Event.EXHAUSTED): State.FAILED for s in State if s not in TERMINAL},
}


class IllegalTransition(RuntimeError):
    """An event the table does not allow from the current state."""

    def __init__(self, state: State, event: Event):
        legal = sorted(e.value for e in legal_events(state))
        super().__init__(
            f"{event.value!r} is not allowed from state {state.value!r}; "
            f"legal events: {legal or 'none (terminal)'}")
        self.state, self.event, self.legal = state, event, legal


class BudgetExceeded(IllegalTransition):
    """The event is in the table but a hard budget forbids it right now."""

    def __init__(self, state: State, event: Event, what: str):
        RuntimeError.__init__(self, what)
        self.state, self.event, self.legal = state, event, []
        self.what = what


def legal_events(state: State) -> frozenset[Event]:
    return frozenset(e for (s, e) in TRANSITIONS if s == state)


@dataclass(frozen=True)
class Transition:
    source: State
    event: Event
    target: State
    turn: int

    def to_dict(self) -> dict:
        return {"from": self.source.value, "event": self.event.value,
                "to": self.target.value, "turn": self.turn}


@dataclass
class Machine:
    """One run's position in the table, its budgets, and the path it took."""

    max_turns: int = 6
    max_relaxations: int = 3
    state: State = State.INTAKE
    turn: int = 0                # DESIGN entries so far
    relaxations: int = 0
    trace: list[Transition] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL

    @property
    def relaxations_left(self) -> int:
        return self.max_relaxations - self.relaxations

    def legal(self) -> frozenset[Event]:
        return legal_events(self.state)

    def fire(self, event: Event, *, relaxing: bool = False) -> Transition:
        """Take `event`, or raise. `relaxing` marks an APPLY that loosens a constraint."""
        target = TRANSITIONS.get((self.state, event))
        if target is None:
            raise IllegalTransition(self.state, event)
        if event is Event.APPLY and relaxing and self.relaxations_left <= 0:
            raise BudgetExceeded(
                self.state, event,
                f"relaxation budget of {self.max_relaxations} is spent")
        if target is State.DESIGN and event is Event.APPLIED \
                and self.turn >= self.max_turns:
            raise BudgetExceeded(
                self.state, event,
                f"turn budget of {self.max_turns} is exhausted")

        if event is Event.APPLY and relaxing:
            self.relaxations += 1
        if target is State.DESIGN:
            self.turn += 1
        step = Transition(self.state, event, target, self.turn)
        self.state = target
        self.trace.append(step)
        return step


def fire_logged(machine: Machine, record: rec.RunRecord, event: Event,
                **kwargs) -> Transition:
    """Take a transition and write it to the run record.

    Shared by every driver (the agent loop, the MCP tools) so a run's path reads
    the same wherever it came from. An `IllegalTransition` is a bug in the
    driver or a refused client call, not a run outcome, so it propagates.
    """
    step = machine.fire(event, **kwargs)
    record.log(rec.TRANSITION, step.turn,
               f"{step.source.value} --{step.event.value}--> {step.target.value}",
               **{k: v for k, v in step.to_dict().items() if k != "turn"})
    return step


def mermaid() -> str:
    """The table as a Mermaid state diagram, for the README."""
    lines = ["stateDiagram-v2", f"    [*] --> {State.INTAKE.value}"]
    for (s, e), t in TRANSITIONS.items():
        if e is Event.EXHAUSTED:
            continue
        lines.append(f"    {s.value} --> {t.value}: {e.value}")
    lines.append("    note right of intake: exhausted (a hard budget) -> failed, "
                 "from any live state")
    return "\n".join(lines)
