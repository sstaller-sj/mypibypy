"""The Streamlit app, run headless with `AppTest`: no browser, no server, no model.

The pure helpers live in `mypibypy.ui` and are tested in test_ui.py; this runs
the real script. The model is a `ScriptedClient` swapped in through
`mypibypy.agent.build_client` -- the script imports it afresh on every run --
and the Ollama health check is pointed at a closed port so it answers at once.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from mypibypy.agent import ScriptedClient  # noqa: E402
from mypibypy.agent.intent import UNCHANGED, UNCHANGED_INT  # noqa: E402
from mypibypy.systems import systems  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1]
_RNG = random.Random(3)
SEQ = "".join(_RNG.choice("ACGT") for _ in range(400))


def intent(**over):
    base = {"intent": "change", "system": UNCHANGED, "n_pairs": UNCHANGED_INT,
            "min_spacing": UNCHANGED_INT, "enforce_window": UNCHANGED,
            "screen": UNCHANGED, "topic": "none", "note": "ok"}
    return base | over


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("MYPIBYPY_OLLAMA_HOST", "http://127.0.0.1:9")   # nothing listens
    monkeypatch.setenv("MYPIBYPY_AGENT_RUNS", str(tmp_path / "runs.jsonl"))
    monkeypatch.delenv("MYPIBYPY_BOWTIE_INDEX", raising=False)
    monkeypatch.syspath_prepend(str(APP_DIR))          # app.py imports app_help
    return AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=120)


def scripted(monkeypatch, replies):
    """Each chat turn builds a client; hand each one the next scripted reply.
    A turn that asks twice, or a third turn, runs the queue dry and fails loudly."""
    queue = list(replies)
    monkeypatch.setattr(
        "mypibypy.agent.build_client",
        lambda kind=None, **kw: ScriptedClient(structured_replies=[queue.pop(0)]))


def on_workspace(app) -> AppTest:
    """Run the app and press Home's button. Later runs stay on the Workspace."""
    return app.run().button(key="open_workspace").click().run()


@pytest.fixture
def example_loci(monkeypatch):
    # The example fusion carries its own sequence: no fetch, no network.
    monkeypatch.setenv("MYPIBYPY_LOCI", str(APP_DIR / "src/mypibypy/loci.example.json"))


def shown(at) -> str:
    return " ".join(m.value for m in at.markdown)


class TestBoot:
    def test_the_app_renders_without_exceptions(self, app):
        at = on_workspace(app)
        assert not at.exception
        assert "Current design" in [h.value for h in at.subheader]

    def test_the_status_says_the_model_is_unreachable_not_merely_selected(self, app):
        at = on_workspace(app)
        assert not at.sidebar.children     # status and settings are in the header
        text = shown(at)
        assert "not reachable" in text and "127.0.0.1:9" in text


class TestHome:
    def test_the_app_opens_on_home(self, app):
        at = app.run()
        assert not at.exception
        assert [t.value for t in at.title] == ["mypibypy"]
        assert at.button(key="open_workspace").label == "Open the workspace"

    def test_the_workspace_opens_on_the_form_when_the_tutorial_asks(self, app):
        """The tutorial's button sets `open_editor` and switches page. AppTest
        cannot open a function page, so this sets the flag the button sets."""
        at = app.run()
        at.session_state["open_editor"] = "Form"
        at.button(key="open_workspace").click().run()
        assert not at.exception
        assert at.segmented_control(key="editor").value == "Form"
        assert "open_editor" not in at.session_state

    def test_the_tutorial_walks_through_the_example_locus(self, monkeypatch):
        monkeypatch.syspath_prepend(str(APP_DIR))

        def page():
            from pathlib import Path

            from app_help import render_tutorial
            from mypibypy import loci
            example = Path(loci.__file__).parent / loci.EXAMPLE_FILENAME
            render_tutorial(loci.load_loci(example)[0], 6)

        at = AppTest.from_function(page).run()
        assert not at.exception
        assert at.button(key="tutorial_fill").label == "Fill in the Form with this example"
        text = shown(at)
        assert "mm10 chr5:142906255-142906754" in text and "6 pairs" in text


@pytest.mark.usefixtures("example_loci")
class TestSeed:
    """`$MYPIBYPY_UI_SEED` designs into a new session, for layout work."""

    def test_a_seeded_session_opens_on_the_design(self, app, monkeypatch):
        monkeypatch.setenv("MYPIBYPY_UI_SEED", "fusion:demo_junction@off")
        at = on_workspace(app)
        assert not at.exception
        picker = next(sb for sb in at.selectbox if sb.label == "Run")
        assert picker.options == ["#1 · seed · fusion demo_junction · 2 pairs"]
        assert [m.value for m in at.metric][1] == "none"

    def test_an_unknown_seed_is_named_and_skipped(self, app, monkeypatch):
        monkeypatch.setenv("MYPIBYPY_UI_SEED", "locus:NOPE:none")
        at = app.run()          # the warning is on whichever page opens first
        assert not at.exception
        assert "no target `locus:NOPE:none`" in at.warning[0].value
        assert "fusion:demo_junction" in at.warning[0].value
        at.button(key="open_workspace").click().run()
        assert "Nothing designed yet" in " ".join(c.value for c in at.caption)


class TestChat:
    def test_a_design_request_designs_and_keeps_the_run(self, app, monkeypatch):
        key = sorted(systems())[0]
        scripted(monkeypatch, [intent(system=key, n_pairs=2)])
        at = on_workspace(app)
        at.chat_input[0].set_value(f"design 2 pairs in {key} for {SEQ}").run()
        assert not at.exception
        assert "run #1" in shown(at)
        assert "pasted sequence (400 nt)" in shown(at)
        # Once: the user's own message. The reply names the target by length,
        # and what the model read is tested in test_chat.py.
        assert shown(at).count(SEQ) == 1

    def test_a_request_that_changes_nothing_is_not_designed_again(self, app,
                                                                   monkeypatch):
        """'too few passed, loosen the Tm' used to redesign the identical run
        under a model comment promising to loosen the Tm."""
        key = sorted(systems())[0]
        scripted(monkeypatch, [
            intent(system=key, n_pairs=2),
            intent(intent="change",
                   note="I will loosen the Tm constraints and run the design again."),
        ])
        at = on_workspace(app)
        at.chat_input[0].set_value(f"design 2 pairs in {key} for {SEQ}").run()
        at.chat_input[0].set_value("too few passed, loosen the Tm").run()
        assert not at.exception
        text = shown(at)
        assert "run #1 stands" in text
        assert "run #2" not in text
        assert "loosen the Tm constraints" not in text, \
            "the model's promise of a change is not shown"

    def test_a_real_change_designs_again(self, app, monkeypatch):
        key = sorted(systems())[0]
        scripted(monkeypatch, [
            intent(system=key, n_pairs=2),
            intent(intent="change", n_pairs=3, note="Three pairs it is."),
        ])
        at = on_workspace(app)
        at.chat_input[0].set_value(f"design 2 pairs in {key} for {SEQ}").run()
        at.chat_input[0].set_value("make it 3 pairs").run()
        assert not at.exception
        text = shown(at)
        assert "pairs → 3" in text and "run #2" in text

    def test_a_question_is_answered_by_the_software_not_the_model(self, app,
                                                                  monkeypatch):
        """The model picks the topic; the reply is the design's own numbers.
        Its note -- here a wrong one -- is not shown."""
        key = sorted(systems())[0]
        scripted(monkeypatch, [
            intent(system=key, n_pairs=2),
            intent(intent="explain", topic="pair_detail",
                   note="Pair 2 has a Tm of 99 C."),
        ])
        at = on_workspace(app)
        at.chat_input[0].set_value(f"design 2 pairs in {key} for {SEQ}").run()
        at.chat_input[0].set_value("what's the Tm of pair 2?").run()
        assert not at.exception
        text = shown(at)
        assert "Pair 2's Tm is" in text and "99 C" not in text
        assert "run #2" not in text, "a question designs nothing"

    def test_a_question_with_a_number_in_it_previews_and_changes_nothing(
            self, app, monkeypatch):
        key = sorted(systems())[0]
        scripted(monkeypatch, [
            intent(system=key, n_pairs=2),
            intent(intent="explain", topic="what_if", n_pairs=3, note="Let me see."),
        ])
        at = on_workspace(app)
        at.chat_input[0].set_value(f"design 2 pairs in {key} for {SEQ}").run()
        at.chat_input[0].set_value("would 3 fit?").run()
        assert not at.exception
        text = shown(at)
        assert "Pairs 2 → 3:" in text and "Previews re-run placement only" in text
        assert "Applied" not in text.split("would 3 fit?")[-1] and "run #2" not in text

    def test_a_follow_up_chip_is_sent_as_the_next_message(self, app, monkeypatch):
        """Each answer offers what to ask next. A chip is an ordinary message:
        picking it runs a turn, exactly as typing it would."""
        key = sorted(systems())[0]
        scripted(monkeypatch, [
            intent(system=key, n_pairs=2),
            intent(intent="explain", topic="summary", note="ok"),
            intent(intent="explain", topic="best_worst", note="ok"),
        ])
        at = on_workspace(app)
        at.chat_input[0].set_value(f"design 2 pairs in {key} for {SEQ}").run()
        offered = at.pills[0].options
        assert "is this design ok?" in offered
        at.pills[0].set_value("is this design ok?").run()
        assert not at.exception
        assert "Run #1 on" in shown(at)
        assert "which pair is the weakest?" in at.pills[0].options
        at.pills[0].set_value("which pair is the weakest?").run()
        assert not at.exception
        assert "Best ranked: pair" in shown(at)

    def test_a_new_chat_offers_somewhere_to_start(self, app):
        at = on_workspace(app)
        assert "how does this work?" in at.pills[0].options


class TestFormAgentLoop:
    def test_an_unscreened_form_design_is_flagged_without_a_model(self, app,
                                                                  monkeypatch):
        """No index is configured here, so every pair is NOT SCREENED. The
        classifier flags the run by itself: the model (at a closed port) is
        never called, and the panel says so. Before, every such design cost a
        5-8 s model call whose only answer was flag_for_review."""
        from mypibypy import pipeline
        monkeypatch.setattr(pipeline, "fetch_sequence", lambda interval: SEQ)
        monkeypatch.setattr(pipeline, "rmsk_hits", lambda interval: [])
        monkeypatch.setattr(pipeline, "mask_repeats", lambda s, i, h: s)
        at = on_workspace(app)
        at.segmented_control(key="editor").set_value("Form").run()
        at.text_input(key="f_region_0").set_value("chr1:1000-1399")
        at.checkbox(key="f_agent").check().run()      # enables the button
        design = next(b for b in at.button if b.label == "Design probes")
        design.click().run()
        assert not at.exception
        verdicts = [i.value for i in at.info if "Agent:" in i.value]
        assert verdicts, "the agent's verdict is shown with the run"
        assert "flagged" in verdicts[0] and "not screened" in verdicts[0]
        assert "without consulting the model" in verdicts[0]
        captions = " ".join(c.value for c in at.caption)
        assert "model was not consulted" in captions


class TestBatch:
    """AppTest cannot drive a file uploader, so the upload is handed in by
    replacing `st.file_uploader`; everything after it is the real editor."""

    @staticmethod
    def upload(monkeypatch, rows: dict[str, list[tuple]], name="order.xlsx"):
        import io
        import streamlit
        from openpyxl import load_workbook
        from mypibypy.batch_sheet import template_bytes

        wb = load_workbook(io.BytesIO(template_bytes(sorted(systems()))))
        for title, values in rows.items():
            for r, row in enumerate(values, start=2):
                for c, value in enumerate(row, start=1):
                    wb[title].cell(row=r, column=c, value=value)
        buf = io.BytesIO()
        wb.save(buf)

        class Uploaded:
            def getvalue(self):
                return buf.getvalue()
        Uploaded.name = name
        monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **kw: Uploaded())

    @staticmethod
    def no_network(monkeypatch):
        from mypibypy import pipeline
        monkeypatch.setattr(pipeline, "fetch_sequence", lambda interval: SEQ)
        monkeypatch.setattr(pipeline, "rmsk_hits", lambda interval: [])
        monkeypatch.setattr(pipeline, "mask_repeats", lambda s, i, h: s)

    def test_the_template_is_offered_before_anything_is_uploaded(self, app):
        at = on_workspace(app)
        at.segmented_control(key="editor").set_value("Batch").run()
        assert not at.exception
        assert at.download_button(key="batch_template").label == "Download the template"
        assert "Up to 50 targets" in " ".join(c.value for c in at.caption)

    def test_an_upload_designs_its_good_rows_into_one_run(self, app, monkeypatch,
                                                          example_loci):
        import datetime as dt
        from mypibypy.loci import load_fusions
        self.no_network(monkeypatch)
        key = sorted(systems())[0]
        fusion = load_fusions(APP_DIR / "src/mypibypy/loci.example.json")[0]
        self.upload(monkeypatch, {
            "Genomic regions": [
                ("a", "GENEA", "hg38", "chr1", 1000, 1399, key),
                ("b", dt.datetime(2026, 9, 7), "hg38", "chr1", 5000, 5399, key),
            ],
            "Fusion junctions": [(fusion.name, fusion.five_prime_gene,
                                  fusion.three_prime_gene, fusion.sequence,
                                  fusion.breakpoint, key)],
            "Sequences": [("enh", "MYOD1", f">enhancer\n{SEQ}", key)],
        })
        at = on_workspace(app)
        at.segmented_control(key="editor").set_value("Batch").run()
        assert not at.exception
        warning = at.warning[0]
        assert "Excel turned gene into a date" in warning.value
        assert at.selectbox(key="b_screen_sequence")   # pasted sequences pick their own
        at.number_input(key="b_pairs").set_value(2)
        at.button(key="b_design").click().run()
        assert not at.exception

        run = at.session_state["design"].current
        assert run.source == "batch" and run.mode == "Batch"
        # A region, both fusion strands, and the pasted sequence.
        assert len(run.results) == 4
        assert [bool(line.results) for line in run.batch] == [True, False, True, True]
        assert run.results[3].region_seq == SEQ
        assert "Excel turned gene into a date" in run.batch[1].problem
        assert run.params.n_pairs == 2
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Targets designed"] == "3 of 4"
        assert "not in the workbook" in " ".join(w.value for w in at.warning)
        assert len(at.selectbox(key=f"batch_show_{run.number}").options) == 4
        assert at.download_button(key=f"download_{run.number}")

        # One order: the region's and the sequence's two tiled pairs, and
        # both fusion strands.
        import io
        from openpyxl import load_workbook
        from mypibypy.ui import workbook_bytes
        order = load_workbook(io.BytesIO(workbook_bytes(run.results, run.note)))
        names = {row[0] for row in order["ORDER_ssDNA"].iter_rows(
            min_row=2, values_only=True) if row[0]}
        assert {"P1L_GENEA_a_t01", "P1R_GENEA_a_t02", "P1L_MYOD1_enh_t01",
                f"P1L_{fusion.name}_sense", f"P1R_{fusion.name}_antisense"} <= names

    def test_a_file_that_is_not_a_workbook_is_refused_in_words(self, app,
                                                                monkeypatch):
        import streamlit

        class Uploaded:
            name = "targets.csv"

            def getvalue(self):
                return b"name,gene\npromoter,Actb\n"
        monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **kw: Uploaded())
        at = on_workspace(app)
        at.segmented_control(key="editor").set_value("Batch").run()
        assert not at.exception
        assert "not an Excel workbook" in at.error[0].value
        assert all(b.key != "b_design" for b in at.button)
