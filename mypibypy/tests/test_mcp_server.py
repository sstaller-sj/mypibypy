"""The MCP server: the pipeline as tools, with the guarantees kept structural.

Under MCP the client drives, so the agent loop's fixed decision structure is
gone and what each tool accepts and returns is the whole safety story. These
tests pin the three guarantees the module docstring states:

* no tool takes a sequence, a target or a path -- checked on the LIVE schemas
  the SDK would advertise, not on a list kept by hand;
* no tool result carries a base;
* a screen that did not run says so, and placeholder inputs are announced.

Designs here are hermetic: genomic fetches are stubbed, and fusions carry their
own sequence. Tests that need the `mcp` package skip without it.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path

import pytest

from mypibypy import mcp_server, pipeline
from mypibypy.agent.record import RunStore
from mypibypy.mcp_server import (
    DNA_RUN,
    FORBIDDEN_MCP_PARAMETERS,
    Session,
    configure_environment,
    load_targets,
)
from mypibypy.systems import systems

EXAMPLE_LOCI = Path(mcp_server.__file__).with_name("loci.example.json")
EXPECTED_TOOLS = {"status", "list_targets", "design_target", "decide_design",
                  "write_order", "diagnose_design", "try_change"}


@pytest.fixture(scope="module")
def keys():
    reg = systems()
    assert len(reg) >= 1, "no amplifier systems configured"
    ks = sorted(reg)
    return ks[0], ks[-1]


@pytest.fixture
def loci_file(tmp_path, keys):
    """A loci file on the configured channels: one locus, the example fusion,
    and one RNA target."""
    example = json.loads(EXAMPLE_LOCI.read_text())
    fusion = dict(example["fusions"][0], system=keys[0], system_antisense=keys[-1])
    data = {
        "loci": [{"name": "enh", "gene": "GENE", "description": "test locus",
                  "build": "hg38", "chrom": "chr1", "start": 1000, "end": 1599,
                  "system": keys[0]}],
        "fusions": [fusion],
        "rna_targets": [{"name": "cds", "gene": "TP53", "accession": "NM_000546.6",
                         "system": keys[0]}],
    }
    path = tmp_path / "loci.test.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def env(monkeypatch, tmp_path, loci_file):
    monkeypatch.setenv("MYPIBYPY_LOCI", str(loci_file))
    monkeypatch.setenv("MYPIBYPY_AGENT_RUNS", str(tmp_path / "runs.jsonl"))
    monkeypatch.setenv(mcp_server.ENV_OUT_DIR, str(tmp_path / "designs"))
    return tmp_path


@pytest.fixture
def offline(monkeypatch):
    """Genomic designs without the network: a fixed random region, no repeats."""
    seq = "".join(random.Random(11).choice("ACGT") for _ in range(600))
    monkeypatch.setattr(pipeline, "fetch_sequence", lambda interval: seq)
    monkeypatch.setattr(pipeline, "rmsk_hits", lambda interval: [])
    monkeypatch.setattr(pipeline, "mask_repeats", lambda s, i, h: s)
    return seq


@pytest.fixture
def session():
    # No backends: every screen is an explicit NOT_SCREENED.
    return Session(genome_reason="no index in tests",
                   transcriptome_reason="no index in tests")


def accepted(session, target_id, **kw):
    """Design a target and, if it stopped for a human, accept it explicitly.

    With no screen backend every pair is NOT SCREENED, which the machine will
    not wave through on its own: the design lands in `flagged`, without a
    model call, and waits for an explicit `proceed` that is recorded against
    the order.
    """
    out = session.design_target(target_id, **kw)
    if out["state"] in ("decide", "flagged"):
        out = session.decide_design(out["design_id"], "proceed",
                                    "accepted unscreened for this test")
    return out


@pytest.fixture
def stubbed(monkeypatch, env):
    """A screen that ran, without an aligner: `Session._run` returns a design
    whose every pair carries the chosen verdict, at the requested pair count.
    `fail` is the way to a genuine `decide` state offline."""
    from test_agent import make_design
    from mypibypy.design import scan
    from mypibypy.sequence import GenomicInterval
    rng = random.Random(4242)
    seq = "".join(rng.choice("ACGT") for _ in range(400))
    iv = GenomicInterval("mm10", "chr1", 1_000_000, 1_000_399)
    cands = scan(seq, iv, top=len(seq))
    sysobj = list(systems().values())[0]

    def make(verdict):
        monkeypatch.setattr(
            Session, "_run",
            lambda self, target, params, screen, pads: [
                make_design((seq, iv), cands, sysobj, verdict=verdict,
                            n_pairs=params.n_pairs)])
    return make


def live_tools():
    pytest.importorskip("mcp")
    server = mcp_server.build_server(Session())
    return asyncio.run(server.list_tools())


class TestNothingSequenceShapedGoesIn:
    def test_the_tool_set_is_exactly_the_reviewed_one(self):
        # A new tool must be added here deliberately, which is the prompt to
        # check it against the invariant below.
        assert {t.name for t in live_tools()} == EXPECTED_TOOLS

    def test_no_live_tool_schema_takes_a_sequence_target_or_path(self):
        offenders = {
            t.name: sorted(p for p in t.input_schema.get("properties", {})
                           if p.lower() in FORBIDDEN_MCP_PARAMETERS)
            for t in live_tools()}
        assert not any(offenders.values()), offenders

    def test_no_free_string_parameter_beyond_ids_and_channel(self):
        # Strings are where bases or coordinates could ride in. Each allowed
        # one is a registry lookup that refuses anything it does not know.
        allowed = {"target_id", "system", "pad_assignment", "design_ids", "design_id",
                   "action", "primary_signal", "reason"}
        for tool in live_tools():
            for name, spec in tool.input_schema.get("properties", {}).items():
                text = json.dumps(spec)
                if '"string"' in text:
                    assert name in allowed, (tool.name, name)
                    if name in ("action", "primary_signal"):
                        assert "enum" in text, f"{name} must be a closed enum"
                    if name == "reason":
                        # the one free-text field is validated server-side:
                        # bounded, and refused if it carries a run of bases.
                        assert tool.name == "decide_design"

    def test_the_forbidden_list_covers_the_agents(self):
        from mypibypy.agent.tools import FORBIDDEN_TOOL_PARAMETERS
        assert FORBIDDEN_TOOL_PARAMETERS <= FORBIDDEN_MCP_PARAMETERS


class TestNoBasesComeOut:
    def test_a_locus_design_returns_no_sequence(self, env, offline, session):
        out = session.design_target("locus:GENE:enh", n_pairs=3)
        text = json.dumps(out)
        assert not DNA_RUN.findall(text)
        for res in session.designs[out["design_id"]].results:
            for pair in res.pairs:
                assert pair.p1l not in text and pair.p1r not in text
                assert pair.candidate.left.seq not in text

    def test_a_fusion_design_returns_no_sequence(self, env, session):
        out = session.design_target("fusion:demo_junction")
        assert len(out["results"]) == 2           # sense and antisense
        assert not DNA_RUN.findall(json.dumps(out))

    def test_listing_targets_does_not_leak_a_fusion_sequence(self, env, session):
        assert not DNA_RUN.findall(json.dumps(session.list_targets()))

    def test_status_carries_no_sequence(self, env, session):
        assert not DNA_RUN.findall(json.dumps(session.status()))


class TestHonestScreens:
    def test_every_pair_says_not_screened_with_a_reason(self, env, offline, session):
        out = session.design_target("locus:GENE:enh", n_pairs=2)
        pairs = out["results"][0]["pairs"]
        assert len(pairs) == 2
        for p in pairs:
            assert p["screened"] is False
            assert p["specificity"] == "NOT SCREENED"
            assert p["specificity_detail"].startswith("NOT SCREENED --")
        assert any("no genome-wide specificity screen" in w for w in out["warnings"])

    def test_skipping_the_screen_is_distinguishable_from_no_index(self, env, offline,
                                                                  session):
        out = session.design_target("locus:GENE:enh", screen=False)
        detail = out["results"][0]["pairs"][0]["specificity_detail"]
        assert "skipped by request" in detail

    def test_status_names_a_missing_screen(self, env, session):
        st = session.status()
        assert st["genome_screen"].startswith("NOT CONFIGURED")
        assert st["transcriptome_screen"].startswith("NOT CONFIGURED")

    def test_shipped_example_loci_are_announced(self, monkeypatch, env, session):
        monkeypatch.setenv("MYPIBYPY_LOCI", str(EXAMPLE_LOCI))
        assert session.status()["loci_are_shipped_example"] is True
        assert any("shipped example" in w for w in session.list_targets()["warnings"])


class TestTargetsAreTheRegistrys:
    def test_ids_cover_every_kind(self, env):
        targets, problems = load_targets()
        assert set(targets) == {"locus:GENE:enh", "fusion:demo_junction",
                                "rna:TP53:cds"}
        assert not problems

    def test_an_unknown_target_is_refused_with_the_real_ids(self, env, session):
        with pytest.raises(ValueError, match="locus:GENE:enh"):
            session.design_target("chr1:1000-1599")

    def test_a_duplicated_id_is_withheld_not_resolved_by_order(self, tmp_path,
                                                                monkeypatch, keys):
        locus = {"name": "x", "gene": "G", "build": "hg38", "chrom": "chr1",
                 "start": 1, "end": 600, "system": keys[0]}
        path = tmp_path / "dupe.json"
        path.write_text(json.dumps({"loci": [locus, dict(locus, start=700, end=1300)]}))
        monkeypatch.setenv("MYPIBYPY_LOCI", str(path))
        targets, problems = load_targets()
        assert "locus:G:x" not in targets
        assert any("more than once" in p for p in problems)


class TestParametersAreBounded:
    def test_n_pairs_is_clamped_and_the_clamp_reported(self, env, offline, session):
        out = session.design_target("locus:GENE:enh", n_pairs=500)
        assert any("n_pairs" in w and "50" in w for w in out["warnings"])

    def test_an_unknown_channel_is_refused_and_recorded(self, env, session):
        from mypibypy.agent.actions import ParameterRefused
        with pytest.raises(ParameterRefused):
            session.design_target("locus:GENE:enh", system="fitc_not_configured")
        record = RunStore().all()[-1]
        assert record.status == "failed"
        assert record.of_kind("refusal")

    def test_a_fusion_cannot_be_tiled(self, env, session):
        with pytest.raises(ValueError, match="fusion junction"):
            session.design_target("fusion:demo_junction", n_pairs=2)

    def test_a_fusion_channel_is_not_overridden_here(self, env, session, keys):
        from mypibypy.agent.actions import ParameterRefused
        with pytest.raises(ParameterRefused, match="two channels"):
            session.design_target("fusion:demo_junction", system=keys[0])

    def test_a_locus_channel_can_be_switched(self, env, offline, session, keys):
        out = session.design_target("locus:GENE:enh", system=keys[-1].upper())
        assert out["results"][0]["system"] == systems()[keys[-1]].name


class TestRecordsAndWorkbook:
    def test_every_design_lands_in_the_run_record(self, env, offline, session):
        out = accepted(session, "locus:GENE:enh")
        record = RunStore().get(out["run_id"])
        assert record is not None and record.status == "succeeded"
        assert record.model == "mcp-client"
        assert record.parameters_history

    def test_write_order_saves_where_the_server_says(self, env, offline, session):
        a = accepted(session, "locus:GENE:enh")["design_id"]
        b = accepted(session, "fusion:demo_junction")["design_id"]
        out = session.write_order([a, b, a])
        path = Path(out["workbook"])
        assert path.is_file() and path.parent == env / "designs"
        assert out["designs"] == [a, b]
        assert out["primaries"] == 2 + 4

    def test_the_workbook_note_carries_the_warnings(self, env, offline, session):
        from openpyxl import load_workbook
        d = accepted(session, "locus:GENE:enh")["design_id"]
        wb = load_workbook(session.write_order([d])["workbook"])
        cells = " ".join(str(c.value) for ws in wb for row in ws.iter_rows()
                         for c in row if c.value)
        assert "no genome-wide specificity screen" in cells

    def test_unknown_design_ids_are_refused(self, env, session):
        with pytest.raises(ValueError, match="unknown design id"):
            session.write_order(["d_nope"])
        with pytest.raises(ValueError):
            session.write_order([])


class TestTheMachineGatesTheTools:
    """The order a client may call things in is the state machine's, not its own."""

    def test_an_unscreened_design_stops_for_a_human_without_a_decision(
            self, env, offline, session):
        out = session.design_target("locus:GENE:enh")
        assert out["state"] == "flagged"
        assert "not screened" in out["why"].lower()
        assert "decide_design" in out["next"] and "proceed" in out["next"]
        assert out["relaxations_left"] == mcp_server.MAX_RELAXATIONS
        record = RunStore().get(out["run_id"])
        assert not record.of_kind("model_decision"), "nobody was asked"
        assert record.of_kind("transition")[-1].detail["event"] == "unscreened"

    def test_write_order_refuses_a_design_that_stopped_for_a_human(
            self, env, offline, session):
        d = session.design_target("locus:GENE:enh")["design_id"]
        with pytest.raises(ValueError, match="have not succeeded.*'flagged'"):
            session.write_order([d])
        assert not (env / "designs").exists()

    def test_proceed_unlocks_the_order_and_the_acceptance_is_recorded(
            self, env, offline, session):
        out = accepted(session, "locus:GENE:enh")
        assert out["state"] == "succeeded"
        assert any("accepted by the MCP client" in w for w in out["warnings"])
        session.write_order([out["design_id"]])

    def test_from_flagged_the_only_decision_is_proceed(self, env, offline, session):
        """A flagged design is accepted or left; a revision is a new run."""
        d = session.design_target("locus:GENE:enh", n_pairs=3)["design_id"]
        for action in ("reduce_pairs", "switch_system", "fail", "flag_for_review"):
            with pytest.raises(ValueError, match="only decision is 'proceed'"):
                session.decide_design(d, action, "try something else")
        assert session.designs[d].state.value == "flagged"
        assert session.designs[d].machine.relaxations == 0

    def test_decide_is_refused_outside_the_decide_state(self, env, offline, session):
        out = accepted(session, "locus:GENE:enh")
        with pytest.raises(ValueError, match="only applies while a design is in 'decide'"):
            session.decide_design(out["design_id"], "fail", "changed my mind")

    def test_flagged_and_failed_designs_cannot_be_ordered(self, env, stubbed, session):
        stubbed("fail")
        for action in ("flag_for_review", "fail"):
            d = session.design_target("locus:GENE:enh")["design_id"]
            out = session.decide_design(d, action, "not happy with it")
            assert out["state"] in ("flagged", "failed")
            with pytest.raises(ValueError, match="have not succeeded"):
                session.write_order([d])

    def test_a_decision_is_held_to_the_closed_schema(self, env, stubbed, session):
        stubbed("fail")
        d = session.design_target("locus:GENE:enh", n_pairs=3)["design_id"]
        from mypibypy.agent import ParameterRefused
        with pytest.raises(ParameterRefused, match="oligo-length"):
            session.decide_design(d, "proceed", "ACGTACGTACGTACGTACGTACGT is fine")
        with pytest.raises(ParameterRefused, match="would change nothing"):
            session.decide_design(d, "relax_tm", "loosen it")
        assert session.designs[d].state.value == "decide", "refusals must not move the run"

    def test_reduce_pairs_redesigns_and_charges_the_budget(self, env, stubbed, session):
        stubbed("fail")
        d = session.design_target("locus:GENE:enh", n_pairs=3)["design_id"]
        out = session.decide_design(d, "reduce_pairs", "fewer pairs, fewer misses")
        assert out["results"][0]["n_pairs"] == 2
        assert out["relaxations_left"] == mcp_server.MAX_RELAXATIONS - 1
        assert out["state"] == "decide"      # still failing its screen

    def test_the_relaxation_budget_ends_the_run_rather_than_loosening(
            self, env, stubbed, session):
        stubbed("fail")
        d = session.design_target("locus:GENE:enh", n_pairs=9)["design_id"]
        for _ in range(mcp_server.MAX_RELAXATIONS):
            out = session.decide_design(d, "reduce_pairs", "again")
        assert out["state"] == "failed"
        assert "budget is spent" in out["why"]
        with pytest.raises(ValueError, match="only applies while"):
            session.decide_design(d, "reduce_pairs", "once more")
        with pytest.raises(ValueError, match="have not succeeded"):
            session.write_order([d])

    def test_a_no_op_action_is_not_charged(self, env, stubbed, session):
        stubbed("fail")
        d = session.design_target("locus:GENE:enh")["design_id"]     # one pair
        with pytest.raises(ValueError, match="Nothing was charged"):
            session.decide_design(d, "reduce_pairs", "already minimal")
        assert session.designs[d].machine.relaxations == 0

    def test_a_fusion_cannot_be_revised(self, env, stubbed, session):
        from mypibypy.agent import ParameterRefused
        stubbed("fail")
        d = session.design_target("fusion:demo_junction")["design_id"]
        with pytest.raises(ParameterRefused, match="fusion takes exactly one pair"):
            session.decide_design(d, "switch_system", "try the other channel")

    def test_the_run_record_holds_a_legal_path(self, env, offline, session):
        from mypibypy.agent.states import TRANSITIONS, TERMINAL, Event, State
        out = accepted(session, "locus:GENE:enh")
        record = RunStore().get(out["run_id"])
        state = State.INTAKE
        events = []
        for e in record.of_kind("transition"):
            assert State(e.detail["from"]) is state
            state = TRANSITIONS[(state, Event(e.detail["event"]))]
            events.append(e.detail["event"])
        assert state in TERMINAL
        assert events[-2:] == ["unscreened", "override"]
        assert record.of_kind("model_decision"), "the client's decision is recorded"

    def test_status_reports_each_design_state(self, env, offline, session):
        d = session.design_target("locus:GENE:enh")["design_id"]
        assert session.status()["design_states"] == {d: "flagged"}


class TestScreenedDesigns:
    """With a screen that ran, a clean design needs no decision at all."""

    def test_a_clean_screen_succeeds_without_a_decision(self, stubbed, session):
        stubbed("pass")
        out = session.design_target("locus:GENE:enh")
        assert out["state"] == "succeeded"
        session.write_order([out["design_id"]])

    def test_a_failed_screen_stops_for_a_decision_and_cannot_be_ordered(
            self, stubbed, session):
        stubbed("fail")
        out = session.design_target("locus:GENE:enh")
        assert out["state"] == "decide"
        with pytest.raises(ValueError, match="have not succeeded"):
            session.write_order([out["design_id"]])


class TestRefusalsReachTheClient:
    """The SDK hides any exception but `ToolError` behind `Error executing tool`.

    Every refusal above is asserted in-process with `pytest.raises`, which says
    nothing about what a client sees. Over the wire, each one must arrive as an
    error result that still carries the hint ("call list_targets", "only applies
    while a design is in 'decide'"), or the model cannot correct itself.
    """

    def server(self, session):
        pytest.importorskip("mcp")
        return mcp_server.build_server(session)

    def call(self, server, name, **arguments) -> str:
        """The error text the SDK would send, or "" if the call succeeded."""
        from mcp.server.mcpserver.exceptions import ToolError
        try:
            asyncio.run(server.call_tool(name, arguments))
        except ToolError as exc:
            return str(exc)
        return ""

    def test_an_unknown_design_id_says_so(self, env, session):
        text = self.call(self.server(session), "write_order", design_ids=["d_nope"])
        assert "unknown design id" in text and "server restart" in text

    def test_an_unknown_target_points_at_the_real_ids(self, env, session):
        text = self.call(self.server(session), "design_target", target_id="chr1:1-2")
        assert "list_targets" in text and "locus:GENE:enh" in text

    def test_a_state_refusal_names_the_state_and_the_way_out(self, env, offline,
                                                              session):
        server = self.server(session)
        d = session.design_target("locus:GENE:enh")["design_id"]
        text = self.call(server, "write_order", design_ids=[d])
        assert "have not succeeded" in text and "'flagged'" in text
        assert "decide_design" in text and "proceed" in text

    def test_a_refused_parameter_is_explained(self, env, session):
        text = self.call(self.server(session), "design_target",
                         target_id="locus:GENE:enh", system="fitc_not_configured")
        assert "unknown system" in text

    def test_a_genuine_bug_still_crashes_generically(self, env, session, monkeypatch):
        from mcp.server.mcpserver.exceptions import UnexpectedToolError
        server = self.server(session)
        monkeypatch.setattr(Session, "status", lambda self: 1 / 0)
        with pytest.raises(UnexpectedToolError) as info:
            asyncio.run(server.call_tool("status", {}))
        assert "division" not in str(info.value), "a crash's text stays on the server"


class TestEnvironment:
    def test_configure_never_overrides_what_is_set(self, monkeypatch, tmp_path):
        for var in ("MYPIBYPY_SYSTEMS", "MYPIBYPY_LOCI", "MYPIBYPY_AGENT_RUNS",
                    mcp_server.ENV_OUT_DIR):
            monkeypatch.delenv(var, raising=False)
        (tmp_path / "loci.local.json").write_text("{}")
        monkeypatch.setenv("MYPIBYPY_AGENT_RUNS", "/keep/me.jsonl")
        configure_environment(tmp_path)
        assert os.environ["MYPIBYPY_LOCI"] == str(tmp_path / "loci.local.json")
        assert "MYPIBYPY_SYSTEMS" not in os.environ      # no file there to point at
        assert os.environ["MYPIBYPY_AGENT_RUNS"] == "/keep/me.jsonl"
        assert os.environ[mcp_server.ENV_OUT_DIR] == str(tmp_path / "designs")


class TestTroubleshooting:
    """`diagnose_design` and `try_change`: the shared diagnosis layer over MCP.
    Read-only -- neither may move a run or spend its budget."""

    def short(self, session):
        """Nine pairs at 40 nt spacing cannot fit the 600 nt test locus."""
        out = session.design_target("locus:GENE:enh", n_pairs=9, min_spacing=40)
        assert out["results"][0]["n_pairs"] < 9
        return out["design_id"]

    def test_facts_say_why_pairs_are_missing(self, env, offline, session):
        d = self.short(session)
        out = session.diagnose_design(d)
        placement = [f["text"] for f in out["facts"] if f["topic"] == "placement"]
        assert placement and "of 9 requested" in placement[0]
        screen = next(f["text"] for f in out["facts"] if f["topic"] == "screen")
        assert "NOT screened" in screen

    def test_a_short_design_comes_with_previews(self, env, offline, session):
        out = session.diagnose_design(self.short(session))
        spacings = [w["min_spacing"] for w in out["what_if"] if w["change"] == "min_spacing"]
        assert spacings and all(s < 40 for s in spacings)
        assert "design_target" in out["what_if_note"]

    def test_a_full_design_has_nothing_to_preview(self, env, offline, session):
        out = session.diagnose_design(session.design_target("locus:GENE:enh")["design_id"])
        assert "what_if" not in out

    def test_a_preview_matches_designing_with_it(self, env, offline, session):
        d = self.short(session)
        preview = session.try_change(d, min_spacing=5)["what_if"]
        designed = session.design_target("locus:GENE:enh", n_pairs=9, min_spacing=5)
        assert preview["pairs_placed"] == designed["results"][0]["n_pairs"]

    def test_neither_tool_moves_the_run(self, env, offline, session):
        d = self.short(session)
        stored = session.designs[d]
        before = (stored.state, stored.machine.turn, stored.machine.relaxations,
                  len(stored.record.events), stored.params)
        session.diagnose_design(d)
        session.try_change(d, n_pairs=3, min_spacing=0)
        assert before == (stored.state, stored.machine.turn, stored.machine.relaxations,
                          len(stored.record.events), stored.params)

    def test_a_preview_is_clamped_like_a_design(self, env, offline, session):
        out = session.try_change(self.short(session), n_pairs=500)
        assert out["what_if"]["n_pairs"] == 50
        assert any("n_pairs" in w for w in out["warnings"])

    def test_a_preview_needs_something_to_change(self, env, offline, session):
        with pytest.raises(ValueError, match="n_pairs, min_spacing"):
            session.try_change(self.short(session))

    def test_a_fusion_cannot_be_previewed(self, env, session):
        d = session.design_target("fusion:demo_junction")["design_id"]
        assert "what_if" not in session.diagnose_design(d)
        with pytest.raises(ValueError, match="cannot be previewed"):
            session.try_change(d, min_spacing=0)

    def test_nothing_sequence_shaped_comes_out(self, env, offline, session):
        d = self.short(session)
        text = json.dumps([session.diagnose_design(d),
                           session.try_change(d, n_pairs=4, min_spacing=10)])
        assert not DNA_RUN.findall(text)


def test_the_server_answers_over_stdio(env):
    """End to end: spawn `python -m mypibypy mcp` and talk MCP to it."""
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mypibypy", "mcp"],
        env={**os.environ, "PYTHONPATH": str(Path(mcp_server.__file__).parents[1])})

    async def talk():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = {t.name for t in (await client.list_tools()).tools}
                result = await client.call_tool("list_targets", {})
                refused = await client.call_tool("write_order",
                                                 {"design_ids": ["d_nothing"]})
                return tools, result, refused

    tools, result, refused = asyncio.run(asyncio.wait_for(talk(), timeout=60))
    assert tools == EXPECTED_TOOLS
    text = "".join(getattr(c, "text", "") for c in result.content)
    assert "fusion:demo_junction" in text
    assert not DNA_RUN.findall(text)
    # A refusal is an error result that still says why, not the SDK's generic
    # "Error executing tool write_order" with the reason left in the server log.
    assert refused.is_error
    why = "".join(getattr(c, "text", "") for c in refused.content)
    assert "unknown design id" in why and "server restart" in why
