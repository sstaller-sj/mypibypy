"""The loop: request -> parameters -> design -> screen -> evaluate -> act.

The model is not in charge here, and that is deliberate. The deterministic
classifier decides first; the model is consulted only when several signals
genuinely conflict and there is no rule that settles it. Four situations:

    CLEAR_PASS        something passed everything. Proceed. No model call.
    CLEAR_FAIL        nothing passed and no relaxation is left. Fail. No call.
    UNSCREENED        a pair has no specificity screen. Stop for a human. No call.
    NEEDS_JUDGEMENT   a real tradeoff. Ask the model which constraint gives.

That split is the point. A probe design where the geometry already answers the
question -- a lone-arm off-target hit whose partner arm has nothing nearby
cannot make a spot, so it is harmless -- needs no language model, and routing it
through one would add cost, latency and a chance of a confident wrong answer for
no gain. What needs judgement is the case where tightening is impossible
without giving something up, or where two scores point in opposite directions.

Budgets are hard. `max_turns` caps the loop, `max_relaxations` (default 3) caps
how far constraints may be loosened before the run reports failure rather than
loosening further, and `max_model_calls` caps spend. Without the relaxation cap
the loop would loosen until something passed, and would therefore always
"succeed" -- which is the specific failure this design is built against.

A caller whose design function reads only some parameters -- the Streamlit
Form's pipeline call takes a pair count, a spacing and a channel -- passes
`allowed_actions`. A model choice outside that set would design the identical
thing again and ask the same question until the turn budget failed a perfectly
good design, so the loop stops for a human instead, recording both the model's
choice and the code's.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from . import record as rec
from .actions import (
    RELAXING_ACTIONS,
    TERMINAL_ACTIONS,
    Action,
    Decision,
    DesignParameters,
    ParameterRefused,
    apply_action,
    clamp,
)
from .client import DecisionClient
from .states import BudgetExceeded, Event, Machine, fire_logged
from .tools import ScreenEvidence, WindowStats, screen_evidence, window_stats


class Situation(str, Enum):
    CLEAR_PASS = "clear_pass"
    CLEAR_FAIL = "clear_fail"
    UNSCREENED = "unscreened"          # a missing input, settled without a model
    NEEDS_JUDGEMENT = "needs_judgement"


#: Below this pass fraction, the constraints are judged too tight to be worth
#: re-running unchanged -- the Myod1 PRR case, where 7 of 250 windows survived
#: and a slightly tighter window yields none. Not a quality threshold: it only
#: decides whether a human-or-model judgement is warranted.
TIGHT_PASS_FRACTION = 0.05

#: Accessibility below this, with specificity otherwise clean, is the
#: two-signals-disagree case: the most accessible site may be the less specific
#: one. Uncalibrated, like every accessibility number.
LOW_ACCESSIBILITY_P = 0.30


@dataclass
class Evaluation:
    """Everything known after one design+screen pass. Pure facts, no judgement."""

    turn: int
    windows: WindowStats
    n_selected: int
    screens: list[ScreenEvidence] = field(default_factory=list)
    capacity: int = 0
    notes: list[str] = field(default_factory=list)
    #: True when the GC/Tm window is being applied as a filter. When it is not,
    #: window statistics describe the locus rather than constrain it, and a low
    #: pass fraction says nothing about whether this design is in trouble.
    window_enforced: bool = False
    #: True when the footprint is fixed by the target -- a fusion junction sits
    #: on the breakpoint and there is no window to choose. Window statistics are
    #: meaningless there and must not drive a judgement call.
    fixed_footprint: bool = False

    @property
    def any_pairs(self) -> bool:
        return self.n_selected > 0

    @property
    def all_screens_pass(self) -> bool:
        return bool(self.screens) and all(s.verdict == "pass" for s in self.screens)

    @property
    def any_screen_fails(self) -> bool:
        return any(s.verdict == "fail" for s in self.screens)

    @property
    def unscreened(self) -> bool:
        return any(not s.screened for s in self.screens)

    @property
    def paralog_arms(self) -> bool:
        return any(s.paralog_like for s in self.screens)

    @property
    def low_accessibility(self) -> bool:
        return any(s.accessibility_evaluated
                   and s.accessibility_p is not None
                   and s.accessibility_p < LOW_ACCESSIBILITY_P
                   for s in self.screens)

    def summary(self) -> str:
        window_note = ("enforced as a filter" if self.window_enforced
                       else "NOT enforced -- reported for context only, since the "
                            "pipeline ranks on GC/Tm distance rather than "
                            "rejecting on it")
        lines = [f"Turn {self.turn}.",
                 f"Windows: {self.windows}. GC/Tm window is {window_note}.",
                 f"Pairs selected: {self.n_selected} "
                 f"(region holds about {self.capacity} at this spacing)."]
        for i, screen in enumerate(self.screens, start=1):
            lines.append(f"Pair {i}: {screen}.")
        lines.extend(self.notes)
        return "\n".join(lines)


def classify(ev: Evaluation, relaxations_left: int) -> tuple[Situation, str]:
    """Decide whether this case needs judgement at all. Returns (situation, why).

    Ordered most-decisive first. Every branch states its reasoning, because the
    reasoning is what goes into the run record and what a reader checks.
    """
    if not ev.any_pairs:
        if relaxations_left <= 0:
            return (Situation.CLEAR_FAIL,
                    "no window satisfies the constraints and the relaxation "
                    "budget is spent")
        return (Situation.NEEDS_JUDGEMENT,
                f"no window satisfies the constraints ({ev.windows}); which "
                f"constraint should give is a tradeoff with no fixed answer")

    if ev.unscreened:
        # Never silently proceed on an unscreened design: that is the mistake
        # the order workbook already guards against, and it must not come back
        # in through the agent. Nor is it a judgement call -- a missing screen
        # is a missing input, and the only defensible answer a model could give
        # is to stop for a human, so the code stops for one without the call.
        # Ahead of a failing screen on purpose: mixed evidence (one pair
        # failing, another never screened) is a human's to look at, not a
        # tradeoff the model may resolve with `proceed`.
        return (Situation.UNSCREENED,
                "at least one pair was not screened, so nothing here "
                "establishes specificity; stopped for a human without "
                "consulting the model")

    if ev.any_screen_fails:
        if relaxations_left <= 0:
            return (Situation.CLEAR_FAIL,
                    "a selected pair has a signal-capable off-target site and "
                    "the relaxation budget is spent")
        return (Situation.NEEDS_JUDGEMENT,
                "a selected pair has a signal-capable off-target site; whether "
                "to move the window or reduce the pair count is a judgement")

    if ev.paralog_arms:
        return (Situation.NEEDS_JUDGEMENT,
                "one arm matches several loci while its partner arm has nothing "
                "there. The pair passes on geometry -- a lone arm cannot recruit "
                "P2 -- but whether an arm that decorates several paralogous loci "
                "is acceptable depends on the experiment")

    if ev.low_accessibility and ev.all_screens_pass:
        return (Situation.NEEDS_JUDGEMENT,
                "specificity is clean but the target site is poorly accessible; "
                "the more accessible site may be the less specific one, so the "
                "two signals point in opposite directions")

    if (ev.window_enforced and not ev.fixed_footprint
            and ev.windows.total and ev.windows.pass_fraction < TIGHT_PASS_FRACTION):
        # Only meaningful when the window is actually being enforced. Without
        # that guard this branch fires at every real locus, because most genomic
        # 25-mers sit outside a 55-62 C Tm window and the pipeline has never
        # cared -- which would send every known-good design to the model.
        return (Situation.NEEDS_JUDGEMENT,
                f"only {ev.windows.passing} of {ev.windows.total} windows "
                f"survive the enforced constraints, so the design is on the edge "
                f"of feasibility even though it passed")

    if ev.all_screens_pass:
        # Only mention window counts when the window was actually applied --
        # otherwise "0 of 1692 windows passed" reads as alarming when it merely
        # describes a locus whose 25-mers sit outside a window nobody imposed.
        detail = (f", and {ev.windows.passing} of {ev.windows.total} windows "
                  f"met the enforced constraints"
                  if ev.window_enforced and ev.windows.total else "")
        return (Situation.CLEAR_PASS,
                f"{ev.n_selected} pair(s) selected and every screen passes"
                f"{detail} -- no constraint is in tension")

    return (Situation.NEEDS_JUDGEMENT, "signals are mixed")


SYSTEM_PROMPT = """\
You are advising a piFISH split-probe design pipeline at a molecular biology \
laboratory. This is routine laboratory probe design: short DNA oligonucleotides \
that label a genomic locus or transcript for imaging under a microscope.

Deterministic code has already designed and screened the probes. It handles \
every clear case by itself. You are consulted only when signals genuinely \
conflict and there is no rule that settles it.

You choose exactly ONE action from the enum. You do not write, edit or suggest \
any DNA sequence -- every oligo comes from the deterministic code, and you have \
no means of supplying one. You do not choose numeric parameters either: you say \
which constraint should give, and the code decides by how much.

What the actions mean:
- proceed: accept the current design and stop.
- relax_tm / relax_gc: widen that window by one fixed step when nothing fits.
- widen_region: search further either side of the target.
- reduce_pairs: ask for fewer tiled pairs when the region cannot hold them.
- switch_system: move to the other amplifier channel.
- accept_single_arm_hits: accept off-target hits where only ONE arm matches. A \
lone arm cannot recruit the secondary amplifier, so it cannot produce a spot; \
this is sometimes the right call and sometimes not, depending on the experiment.
- flag_for_review: stop and ask a human. Correct whenever the tradeoff is the \
experimenter's to make rather than yours.
- fail: stop, reporting that no acceptable design was found.

Prefer flag_for_review over guessing. A design that stops for a human decision \
is a better outcome than one that quietly loosened a constraint nobody agreed to.
"""


@dataclass
class AgentResult:
    """The end of a run: what was decided, and the record that explains it."""

    status: str                     # succeeded | failed | flagged
    parameters: DesignParameters
    evaluation: Evaluation | None
    record: rec.RunRecord
    design: object = None           # the DesignResult, when one was produced
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"


class Budget:
    """Hard limits. Checked in code; not advertised to the model as a request."""

    def __init__(self, max_turns: int = 6, max_relaxations: int = 3,
                 max_model_calls: int = 4, max_total_tokens: int = 40_000):
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.max_turns = max_turns
        self.max_relaxations = max_relaxations
        self.max_model_calls = max_model_calls
        self.max_total_tokens = max_total_tokens


class DesignAgent:
    """Drives design -> screen -> evaluate -> act until something settles it.

    `design_fn(params) -> (DesignResult, candidates)` is injected, so the agent
    is independent of whether the target is genomic, RNA or a fusion: the caller
    supplies the closure that knows. It is also what keeps the tests free of
    network and aligner dependencies.
    """

    def __init__(self, design_fn, client: DecisionClient | None = None,
                 budget: Budget | None = None,
                 known_systems: tuple[str, ...] = (),
                 store: rec.RunStore | None = None,
                 allowed_actions=None):
        self.design_fn = design_fn
        self.client = client
        self.budget = budget or Budget()
        self.known_systems = tuple(known_systems)
        self.store = store
        #: The actions `design_fn` can act on. Terminal actions are always in;
        #: a model choice outside the set stops the run for a human rather
        #: than designing the identical thing again. None means every action.
        self.allowed_actions = (frozenset(Action) if allowed_actions is None
                                else frozenset(allowed_actions) | TERMINAL_ACTIONS)

    @property
    def relaxations_offered(self) -> list[str]:
        """The non-terminal actions this run may apply, by name."""
        return sorted(a.value for a in self.allowed_actions - TERMINAL_ACTIONS)

    # -- the loop ----------------------------------------------------------
    def run(self, params: DesignParameters, target_label: str = "",
            request: str = "") -> AgentResult:
        record = rec.RunRecord(
            target=target_label, request=request,
            model=getattr(self.client, "name", "none"))

        params, clamps = clamp(params, self.known_systems)
        for event in clamps:
            record.log(rec.PARAMETER_CLAMP, 0, str(event),
                       parameter=event.parameter, requested=repr(event.requested),
                       applied=repr(event.applied))
        record.log_parameters(0, params, "initial (after clamping)")

        machine = Machine(max_turns=self.budget.max_turns,
                          max_relaxations=self.budget.max_relaxations)
        fire_logged(machine, record, Event.CLAMPED)
        design = None

        while True:
            turn = machine.turn
            record.turns_used = turn

            design, candidates = self.design_fn(params)
            stats = window_stats(candidates, params)
            screens = [screen_evidence(s, a) for _, s, a in design.rows()] if design else []
            ev = Evaluation(turn=turn, windows=stats,
                            n_selected=design.n_pairs if design else 0,
                            screens=screens,
                            capacity=len(candidates),
                            window_enforced=params.enforce_window,
                            fixed_footprint=len(candidates) <= 1)
            record.log(rec.TOOL_CALL, turn,
                       f"designed and screened: {ev.n_selected} pair(s), {stats}",
                       windows=stats.__dict__.copy(),
                       verdicts=[s.verdict for s in screens])
            fire_logged(machine, record, Event.DESIGNED)

            situation, why = classify(ev, machine.relaxations_left)

            if situation is Situation.CLEAR_PASS:
                record.log(rec.DETERMINISTIC_DECISION, turn,
                           f"proceed -- {why}", action=Action.PROCEED.value)
                fire_logged(machine, record, Event.CLEAR_PASS)
                return self._finish(record, "succeeded", params, ev, design,
                                    f"accepted without consulting the model: {why}")

            if situation is Situation.CLEAR_FAIL:
                record.log(rec.DETERMINISTIC_DECISION, turn,
                           f"fail -- {why}", action=Action.FAIL.value)
                fire_logged(machine, record, Event.CLEAR_FAIL)
                return self._finish(record, "failed", params, ev, design, why)

            if situation is Situation.UNSCREENED:
                # Deterministic, and deliberately so: a missing screen is a
                # missing input. Asking a model costs seconds and its only
                # defensible answer is to stop for a human -- so stop for one.
                record.log(rec.DETERMINISTIC_DECISION, turn,
                           f"flag_for_review -- {why}",
                           action=Action.FLAG_FOR_REVIEW.value)
                fire_logged(machine, record, Event.UNSCREENED)
                return self._finish(record, "flagged", params, ev, design, why)

            fire_logged(machine, record, Event.NEEDS_JUDGEMENT)
            decision = self._ask(record, turn, ev, why)

            if decision.action not in self.allowed_actions:
                # The caller said its design function does not act on that
                # parameter. Applying it would design the identical thing again
                # and ask the same question until the turn budget failed a good
                # design, so the run stops for a human -- recorded as the
                # code's decision, with the model's choice beside it.
                record.log(rec.DETERMINISTIC_DECISION, turn,
                           f"flag_for_review -- the model chose "
                           f"{decision.action.value}, which this design "
                           f"function does not act on",
                           action=Action.FLAG_FOR_REVIEW.value,
                           model_chose=decision.action.value,
                           model_reason=decision.reason)
                fire_logged(machine, record, Event.FLAG)
                offered = ", ".join(self.relaxations_offered) or "no relaxation"
                return self._finish(
                    record, "flagged", params, ev, design,
                    f"the model chose {decision.action.value}, which would not "
                    f"change this design (here only {offered} can); a human "
                    f"should decide")

            if decision.action in TERMINAL_ACTIONS:
                event, status = {
                    Action.PROCEED: (Event.PROCEED, "succeeded"),
                    Action.FAIL: (Event.FAIL, "failed"),
                    Action.FLAG_FOR_REVIEW: (Event.FLAG, "flagged"),
                }[decision.action]
                fire_logged(machine, record, event)
                return self._finish(record, status, params, ev, design,
                                    decision.reason)

            try:
                fire_logged(machine, record, Event.APPLY,
                           relaxing=decision.action in RELAXING_ACTIONS)
            except BudgetExceeded:
                record.log(rec.BUDGET, turn,
                           "relaxation budget spent; reporting failure rather "
                           "than loosening further",
                           used=machine.relaxations,
                           limit=self.budget.max_relaxations)
                fire_logged(machine, record, Event.EXHAUSTED)
                return self._finish(
                    record, "failed", params, ev, design,
                    f"no acceptable design after "
                    f"{self.budget.max_relaxations} relaxation(s)")
            record.relaxations_used = machine.relaxations

            try:
                params, change = apply_action(decision.action, params,
                                              self.known_systems)
            except ParameterRefused as exc:
                record.log(rec.REFUSAL, turn, f"action refused: {exc}")
                fire_logged(machine, record, Event.REFUSED)
                return self._finish(record, "failed", params, ev, design, str(exc))
            record.log(rec.TOOL_CALL, turn, f"applied {decision.action.value}: {change}")
            record.log_parameters(turn, params, f"after {decision.action.value}")

            try:
                fire_logged(machine, record, Event.APPLIED)
            except BudgetExceeded:
                record.log(rec.BUDGET, turn,
                           f"turn budget of {self.budget.max_turns} exhausted")
                fire_logged(machine, record, Event.EXHAUSTED)
                return self._finish(
                    record, "failed", params, ev, design,
                    f"turn budget of {self.budget.max_turns} exhausted")

    # -- the one place the model is consulted ------------------------------
    def _ask(self, record: rec.RunRecord, turn: int, ev: Evaluation,
             why: str) -> Decision:
        """Ask the model which constraint gives. Degrades to a safe default."""
        fallback = Decision(
            action=Action.FLAG_FOR_REVIEW,
            reason=("no model decision was available, so the run stops for a "
                    "human rather than choosing a relaxation unprompted"),
            primary_signal="specificity", source="fallback")

        if self.client is None:
            record.log(rec.DETERMINISTIC_DECISION, turn,
                       f"no model configured; flagging for review -- {why}",
                       action=fallback.action.value)
            return fallback

        if record.model_calls >= self.budget.max_model_calls:
            record.log(rec.BUDGET, turn,
                       f"model call budget of {self.budget.max_model_calls} "
                       f"reached; flagging for review")
            return fallback
        if record.input_tokens + record.output_tokens >= self.budget.max_total_tokens:
            record.log(rec.BUDGET, turn, "token budget reached; flagging for review")
            return fallback

        offered = ""
        if self.allowed_actions != frozenset(Action):
            offered = (f"Actions that can change this design: "
                       f"{', '.join(self.relaxations_offered) or 'none'}; any "
                       f"other non-terminal action would change nothing here.\n\n")
        evidence = (f"{ev.summary()}\n\nWhy this needs a decision: {why}\n\n"
                    f"{offered}Choose one action.")
        reply = self.client.decide(SYSTEM_PROMPT, evidence)
        record.model_calls += 1
        record.input_tokens += reply.input_tokens
        record.output_tokens += reply.output_tokens

        if reply.refused:
            # A refusal is a normal branch, not a crash. Routine probe design can
            # trip a biology classifier; the run degrades to a human decision.
            record.log(rec.REFUSAL, turn,
                       f"model declined to answer (category "
                       f"{reply.refusal_category or 'unspecified'}); flagging for "
                       f"review",
                       category=reply.refusal_category,
                       explanation=reply.refusal_explanation)
            return fallback
        if not reply.usable:
            record.log(rec.ERROR, turn,
                       f"no usable decision ({reply.error}); flagging for review",
                       error=reply.error)
            return fallback

        decision = reply.decision
        record.log(rec.MODEL_DECISION, turn,
                   f"{decision.action.value}: {decision.reason}",
                   action=decision.action.value,
                   primary_signal=decision.primary_signal,
                   evidence=ev.summary(), prompted_because=why)
        return decision

    # -- finishing ---------------------------------------------------------
    def _finish(self, record: rec.RunRecord, status: str,
                params: DesignParameters, ev: Evaluation | None,
                design, message: str) -> AgentResult:
        record.status = status
        record.finished_at = rec._now()
        record.log(rec.OUTCOME, record.turns_used, f"{status}: {message}")
        if self.store is not None:
            self.store.append(record)
        return AgentResult(status=status, parameters=params, evaluation=ev,
                           record=record, design=design, message=message)
