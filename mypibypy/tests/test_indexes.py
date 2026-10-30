"""Named specificity indexes: what the Form and the chat choose among.

What is worth guarding: a name maps to exactly one path, a malformed file is
refused rather than quietly dropping an index, an index with no files on disk
is reported, and a name that disappears screens nothing instead of silently
falling back to some other genome.
"""
from __future__ import annotations

import json

import pytest

from mypibypy.agent import DesignParameters
from mypibypy.agent.context import build_chat_context
from mypibypy.agent.intent import (
    SCREEN_OFF,
    UNCHANGED,
    Intent,
    apply_screen,
    intent_schema,
)
from mypibypy.agent.actions import ParameterRefused
from mypibypy.indexes import (
    DEFAULT,
    OFF,
    IndexesNotConfigured,
    bowtie_index_problem,
    kind_mismatch,
    load_indexes,
    resolve_screen,
)
from mypibypy.session import DesignSession


def write(tmp_path, entries) -> str:
    path = tmp_path / "indexes.local.json"
    path.write_text(json.dumps({"indexes": entries}))
    return str(path)


@pytest.fixture
def built(tmp_path):
    """A basename with the file bowtie-build would leave next to it."""
    (tmp_path / "hg38.1.ebwt").write_text("")
    return str(tmp_path / "hg38")


def test_no_file_means_no_named_indexes(tmp_path, monkeypatch):
    monkeypatch.delenv("MYPIBYPY_INDEXES", raising=False)
    assert load_indexes(start=tmp_path) == {}


def test_entries_load_in_order_with_their_kind(tmp_path, built):
    idx = load_indexes(write(tmp_path, {
        "hg38": {"path": built, "description": "human"},
        "hg38-tx": {"path": "/nowhere/tx", "kind": "transcriptome"}}))
    assert list(idx) == ["hg38", "hg38-tx"]
    assert idx["hg38"].kind == "genome" and idx["hg38"].problem is None
    assert str(idx["hg38"]) == "hg38 (genome, human)"
    assert "no bowtie v1 index" in idx["hg38-tx"].problem


def test_a_large_index_counts_as_built(tmp_path):
    (tmp_path / "big.1.ebwtl").write_text("")
    assert bowtie_index_problem(str(tmp_path / "big")) is None


@pytest.mark.parametrize("entries", [
    {"hg38": {}},                                   # no path
    {"hg38": {"path": "  "}},                       # blank path
    {"hg38": {"path": "/x", "kind": "exome"}},       # unknown kind
    {"/data/hg38": {"path": "/x"}},                 # a path as a name
    {"off": {"path": "/x"}},                        # shadows the off choice
    {"a" * 40: {"path": "/x"}},                     # not a short identifier
])
def test_malformed_entries_are_refused(tmp_path, entries):
    with pytest.raises(IndexesNotConfigured):
        load_indexes(write(tmp_path, entries))


def test_resolve_a_name_off_and_the_default(tmp_path, built, monkeypatch):
    monkeypatch.delenv("MYPIBYPY_BOWTIE_INDEX", raising=False)
    idx = load_indexes(write(tmp_path, {"hg38": {"path": built}}))
    assert resolve_screen("hg38", idx) == (built, "hg38")
    assert resolve_screen(OFF, idx, "/sidebar/path") == ("", "off")
    assert resolve_screen(DEFAULT, idx, "/sidebar/path") == ("/sidebar/path",
                                                              "/sidebar/path")
    assert resolve_screen(DEFAULT, idx) == ("", "off")


def test_default_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("MYPIBYPY_BOWTIE_INDEX", "/env/mm10")
    assert resolve_screen(DEFAULT, {}) == ("/env/mm10", "/env/mm10")
    assert resolve_screen(DEFAULT, {}, "/sidebar") == ("/sidebar", "/sidebar")


def test_a_vanished_name_screens_nothing_rather_than_the_default(monkeypatch):
    monkeypatch.setenv("MYPIBYPY_BOWTIE_INDEX", "/env/hg38")
    path, label = resolve_screen("mm10", {}, "/sidebar")
    assert path == "" and "not configured" in label


def test_a_wrong_kind_is_explained_in_one_set_of_words(tmp_path, built):
    """The Form warns and the MCP server refuses with this sentence; one
    source, so they cannot drift apart."""
    idx = load_indexes(write(tmp_path, {
        "hg38": {"path": built},
        "tx": {"path": built, "kind": "transcriptome"}}))
    assert kind_mismatch(idx["hg38"], "genome") is None
    assert kind_mismatch(idx["tx"], "transcriptome") is None
    rna = kind_mismatch(idx["hg38"], "transcriptome")
    assert rna.startswith("**hg38** is a genome index.")
    assert "splice junction" in rna and "falsely clean" in rna
    dna = kind_mismatch(idx["tx"], "genome")
    assert dna.startswith("**tx** is a transcriptome index.")
    assert "introns" in dna and "falsely clean" in dna


# --- the chat side ------------------------------------------------------------

def payload(**over):
    base = {"intent": "change", "system": UNCHANGED, "n_pairs": -1,
            "min_spacing": -1, "enforce_window": UNCHANGED, "screen": UNCHANGED,
            "note": "ok"}
    base.update(over)
    return base


def test_the_schema_enumerates_names_never_a_path():
    prop = intent_schema(("cy3",), ("hg38", "mm10"))["properties"]["screen"]
    assert prop["enum"] == ["hg38", "mm10", SCREEN_OFF, UNCHANGED]


@pytest.mark.parametrize("value", ["/data/genomes/hg38", "mm39", "hg38; rm -rf"])
def test_a_screen_outside_the_configured_names_is_refused(value):
    with pytest.raises(ParameterRefused):
        Intent.from_payload(payload(screen=value), ("cy3",), ("hg38",))


def test_apply_screen_changes_only_when_asked():
    s = DesignSession()
    chosen = Intent.from_payload(payload(screen="hg38"), ("cy3",), ("hg38",))
    assert apply_screen(chosen, s.screen) == ("hg38", "specificity screen → hg38")
    assert apply_screen(chosen, "hg38") == ("hg38", None)
    off = Intent.from_payload(payload(screen=SCREEN_OFF), ("cy3",), ("hg38",))
    assert apply_screen(off, "hg38") == ("off", "specificity screen off")
    same = Intent.from_payload(payload(), ("cy3",), ("hg38",))
    assert apply_screen(same, "hg38") == ("hg38", None)


def test_a_run_keeps_what_it_was_screened_against():
    s = DesignSession(screen="hg38")
    run = s.record(source="chat", results=[], note="", target_label="t",
                   screen="hg38")
    s.screen = OFF
    assert run.screen == "hg38"


def test_the_context_names_indexes_without_paths(tmp_path, built):
    from mypibypy.agent.context import ScreenStatus
    ctx = build_chat_context("screen it against human", DesignParameters(),
                             "t", screen=ScreenStatus(
                                 on=False, against="off",
                                 available=["hg38 (genome, human)"]))
    text = ctx.render()
    assert "hg38 (genome, human)" in text and built not in text
