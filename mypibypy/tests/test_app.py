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
            "screen": UNCHANGED, "note": "ok"}
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


def shown(at) -> str:
    return " ".join(m.value for m in at.markdown)


class TestBoot:
    def test_the_app_renders_without_exceptions(self, app):
        at = app.run()
        assert not at.exception
        assert "Current design" in [h.value for h in at.subheader]

    def test_the_sidebar_says_the_model_is_unreachable_not_merely_selected(self, app):
        at = app.run()
        text = " ".join(m.value for m in at.sidebar.markdown)
        assert "not reachable" in text and "127.0.0.1:9" in text


class TestChat:
    def test_a_design_request_designs_and_keeps_the_run(self, app, monkeypatch):
        key = sorted(systems())[0]
        scripted(monkeypatch, [intent(system=key, n_pairs=2)])
        at = app.run()
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
        at = app.run()
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
        at = app.run()
        at.chat_input[0].set_value(f"design 2 pairs in {key} for {SEQ}").run()
        at.chat_input[0].set_value("make it 3 pairs").run()
        assert not at.exception
        text = shown(at)
        assert "pairs → 3" in text and "run #2" in text


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
        at = app.run()
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
