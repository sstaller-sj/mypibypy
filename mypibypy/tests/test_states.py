"""The state machine: the legal order is a table, and the loop obeys it."""
import pytest

from mypibypy.agent import (Budget, DesignAgent, DesignParameters, ScriptedClient)
from mypibypy.agent.states import (BudgetExceeded, Event, IllegalTransition,
                                   Machine, State, TERMINAL, TRANSITIONS,
                                   legal_events, mermaid)
from test_agent import (candidates, make_design, region, relax_payload,  # noqa: F401
                        system)


class TestTable:
    def test_terminal_states_have_no_exits(self):
        for state in TERMINAL:
            assert legal_events(state) == frozenset()

    def test_every_live_state_can_be_exhausted(self):
        for state in State:
            if state not in TERMINAL:
                assert Event.EXHAUSTED in legal_events(state)

    def test_every_state_is_reachable_from_intake(self):
        seen, frontier = {State.INTAKE}, [State.INTAKE]
        while frontier:
            here = frontier.pop()
            for (s, _), t in TRANSITIONS.items():
                if s == here and t not in seen:
                    seen.add(t)
                    frontier.append(t)
        assert seen == set(State)

    def test_a_verdict_cannot_skip_the_screen(self):
        m = Machine()
        m.fire(Event.CLAMPED)
        with pytest.raises(IllegalTransition, match="legal events"):
            m.fire(Event.CLEAR_PASS)        # DESIGN -> verdict, no DESIGNED first

    def test_nothing_leaves_a_terminal_state(self):
        m = Machine()
        for e in (Event.CLAMPED, Event.DESIGNED, Event.CLEAR_PASS):
            m.fire(e)
        assert m.finished
        with pytest.raises(IllegalTransition):
            m.fire(Event.CLAMPED)

    def test_diagram_mentions_every_non_budget_edge(self):
        text = mermaid()
        assert "evaluate --> decide: needs_judgement" in text


class TestBudgets:
    def _to_decide(self, m):
        for e in (Event.CLAMPED, Event.DESIGNED, Event.NEEDS_JUDGEMENT):
            m.fire(e)

    def test_relaxation_budget_is_enforced(self):
        m = Machine(max_relaxations=1, max_turns=9)
        self._to_decide(m)
        m.fire(Event.APPLY, relaxing=True)
        m.fire(Event.APPLIED)
        m.fire(Event.DESIGNED); m.fire(Event.NEEDS_JUDGEMENT)
        with pytest.raises(BudgetExceeded, match="relaxation budget"):
            m.fire(Event.APPLY, relaxing=True)

    def test_a_non_relaxing_action_is_not_charged(self):
        m = Machine(max_relaxations=0)
        self._to_decide(m)
        m.fire(Event.APPLY, relaxing=False)
        assert m.relaxations == 0

    def test_turn_budget_is_enforced(self):
        m = Machine(max_turns=1)
        self._to_decide(m)
        m.fire(Event.APPLY)
        with pytest.raises(BudgetExceeded, match="turn budget"):
            m.fire(Event.APPLIED)
        m.fire(Event.EXHAUSTED)
        assert m.state is State.FAILED


class TestLoopDrivesTheTable:
    def _run(self, region, candidates, system, **budget):
        def design_fn(p):
            design = make_design(region, candidates, system)
            design.pairs, design.pair_screens, design.pair_accessibility = [], [], []
            return design, candidates

        agent = DesignAgent(design_fn, client=ScriptedClient(
            [relax_payload() for _ in range(10)]),
            budget=Budget(max_turns=10, max_model_calls=10, **budget))
        return agent.run(DesignParameters())

    def test_every_run_records_a_legal_path_to_a_terminal_state(
            self, region, candidates, system):
        result = self._run(region, candidates, system, max_relaxations=2)
        steps = [e.detail for e in result.record.events if e.kind == "transition"]
        assert steps, "the run record should carry the transitions"
        state = State.INTAKE
        for step in steps:
            assert State(step["from"]) is state
            state = TRANSITIONS[(state, Event(step["event"]))]
            assert state is State(step["to"])
        assert state in TERMINAL
        assert steps[-1]["to"] == "failed"

    def test_relaxation_cap_still_reports_failure(self, region, candidates, system):
        result = self._run(region, candidates, system, max_relaxations=2)
        assert result.status == "failed"
        assert result.record.relaxations_used == 2
