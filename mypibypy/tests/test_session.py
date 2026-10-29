"""The shared design state behind the Form and the Chat.

What is worth guarding: an edit from either side lands in one place, a run
keeps the parameters it was made with (not whatever they became later), and
the form is told to refresh only when something other than the form changed.
"""
from __future__ import annotations

from mypibypy.agent import DesignParameters
from mypibypy.agent.intent import ExtractedTarget
from mypibypy.session import DesignSession, changed_params, target_from_form

REGION = ExtractedTarget("genomic", build="hg38", chrom="chr7",
                         start=5530000, end=5531000)


class _Result:
    """Just enough of a DesignResult for a run label."""

    def __init__(self, n_pairs: int) -> None:
        self.n_pairs = n_pairs


def test_chat_edits_bump_revision_and_form_edits_do_not():
    s = DesignSession()
    s.update(params=DesignParameters(n_pairs=6), target=REGION)
    assert (s.params.n_pairs, s.target, s.revision) == (6, REGION, 1)
    s.update(params=DesignParameters(n_pairs=8), from_form=True)
    assert (s.params.n_pairs, s.revision) == (8, 1)


def test_update_without_a_target_keeps_the_current_one():
    s = DesignSession(target=REGION)
    s.update(params=DesignParameters(n_pairs=3))
    assert s.target == REGION


def test_run_snapshots_parameters():
    s = DesignSession(params=DesignParameters(n_pairs=4))
    run = s.record(source="form", results=[_Result(4)], note="",
                   target_label=str(REGION))
    s.update(params=DesignParameters(n_pairs=12))
    assert run.params.n_pairs == 4
    assert run.label == f"#1 · form · {REGION} · 4 pairs"


def test_previous_is_the_last_run_on_the_same_target():
    s = DesignSession()
    first = s.record(source="form", results=[_Result(1)], note="",
                     target_label="A")
    s.record(source="chat", results=[_Result(1)], note="", target_label="B")
    third = s.record(source="chat", results=[_Result(2)], note="",
                     target_label="A")
    assert s.previous(third) is first
    assert s.previous(first) is None


def test_selecting_an_old_run_and_recording_returns_to_latest():
    s = DesignSession()
    for label in "ABC":
        s.record(source="form", results=[_Result(1)], note="", target_label=label)
    s.select(1)
    assert s.current.target_label == "A"
    s.select(3)
    assert s.current.target_label == "C"
    s.select(2)
    s.record(source="chat", results=[_Result(1)], note="", target_label="D")
    assert s.current.target_label == "D"


def test_no_runs_means_no_current():
    assert DesignSession().current is None


def test_changed_params_reports_old_and_new():
    before = DesignParameters(n_pairs=4, min_spacing=3)
    after = DesignParameters(n_pairs=6, min_spacing=3)
    assert changed_params(before, after) == {"n_pairs": (4, 6)}
    assert changed_params(None, after) == {}


def test_form_target_labels_match_the_chats():
    """The panel compares labels to spot an unapplied target change, so the
    form's targets must print exactly like the chat's."""
    form = target_from_form("Genomic region", build="hg38", chrom="chr7",
                            start=5530000, end=5531000)
    assert str(form) == str(REGION)
    assert target_from_form("RNA transcript", accession="NM_000546.6").kind == "rna"
    assert str(target_from_form("Pasted sequence", sequence="A" * 60)) == \
        "pasted sequence (60 nt)"
    assert target_from_form("Fusion junction") is None
