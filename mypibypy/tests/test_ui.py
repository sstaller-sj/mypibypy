"""Presentation helpers for the Streamlit app.

Streamlit itself is not imported here -- `mypibypy.ui` holds the logic precisely
so it can be tested without a browser or a server. What the app does with these
rows (layout, widgets) was verified by running it.

The point worth guarding: a missing screen must render as words, not as a blank
cell. A table is exactly where "not screened" would otherwise be mistaken for
"passed".
"""
from __future__ import annotations

import random

import pytest

from mypibypy.design import build_tiled_pairs, scan, select_tiled
from mypibypy.loci import Locus
from mypibypy.pipeline import DesignResult
from mypibypy.sequence import GenomicInterval
from mypibypy.specificity import SCREENED, SpecificityReport, not_screened
from mypibypy.systems import systems
from mypibypy.expression import ExpressionTable
from mypibypy.fold import EVALUATED as FOLD_EVALUATED
from mypibypy.fold import OligoFold, PairFolding
from mypibypy.specificity import Hit, PairSite
from mypibypy.ui import (
    candidate_rows,
    expression_status,
    folding_cell,
    folding_details,
    on_target_cell,
    unscreened_pairs,
    workbook_bytes,
    worst_offtarget_cell,
)


@pytest.fixture(scope="module")
def system():
    reg = systems()
    assert reg, "no amplifier systems configured"
    return list(reg.values())[0]


def make_result(system, n_pairs=1, screened=False, accessibility=None,
                screens=None, folding=None):
    rng = random.Random(77)
    seq = "".join(rng.choice("ACGT") for _ in range(600))
    iv = GenomicInterval("hg38", "chr1", 1, 600)
    locus = Locus("target", "GENE", "ui test", iv, system.name)
    cands = scan(seq, iv, top=len(seq))
    pairs = build_tiled_pairs("target", "GENE", system, select_tiled(cands, n_pairs))
    if screens is not None:
        screens = [screens[i % len(screens)] for i in range(len(pairs))]
    elif screened:
        screens = [SpecificityReport(status=SCREENED, backend="bowtie v1 -v 3 -a",
                                     database="hg38") for _ in pairs]
    else:
        screens = [not_screened("no index supplied") for _ in pairs]
    access = [accessibility or _not_evaluated() for _ in pairs]
    return DesignResult(locus, system, seq, cands[:5], pairs[0], [], screens[0],
                        accessibility=access[0], pairs=pairs,
                        pair_screens=screens, pair_accessibility=access,
                        pair_folding=list(folding or []))


def _not_evaluated():
    from mypibypy.accessibility import not_evaluated
    return not_evaluated("RNAplfold not installed")


class TestCandidateRows:
    def test_one_row_per_ordered_pair(self, system):
        assert len(candidate_rows(make_result(system, n_pairs=4))) == 4

    def test_columns_cover_what_the_brief_asked_for(self, system):
        row = candidate_rows(make_result(system))[0]
        for column in ("GC L", "GC R", "Tm L", "Tm R", "accessibility",
                       "specificity", "P1L", "P1R"):
            assert column in row, f"{column} missing from the candidate table"

    def test_an_unscreened_pair_says_so_in_words(self, system):
        """A blank cell would read as a pass."""
        row = candidate_rows(make_result(system, screened=False))[0]
        assert row["specificity"] == "NOT SCREENED"

    def test_a_screened_pair_shows_its_verdict(self, system):
        row = candidate_rows(make_result(system, screened=True))[0]
        assert row["specificity"] == "pass"

    def test_unevaluated_accessibility_says_so_in_words(self, system):
        row = candidate_rows(make_result(system))[0]
        assert row["accessibility"] == "not evaluated"

    def test_evaluated_accessibility_is_a_number(self, system):
        from mypibypy.accessibility import (
            EVALUATED,
            AccessibilityResult,
            PadAccessibility,
        )
        pad = PadAccessibility("left", 0, 25, 8, 0.734, 0.9, 0.8)
        right = PadAccessibility("right", 26, 25, 8, 0.812, 0.9, 0.8)
        access = AccessibilityResult(status=EVALUATED, left=pad, right=right,
                                     tool="RNAplfold")
        row = candidate_rows(make_result(system, accessibility=access))[0]
        assert row["accessibility"] == pytest.approx(0.734, abs=1e-3)

    def test_rows_are_numbered_from_one(self, system):
        rows = candidate_rows(make_result(system, n_pairs=3))
        assert [r["#"] for r in rows] == [1, 2, 3]

    def test_names_are_distinct_across_tiled_pairs(self, system):
        rows = candidate_rows(make_result(system, n_pairs=3))
        names = [r["P1L"] for r in rows] + [r["P1R"] for r in rows]
        assert len(set(names)) == len(names)


class TestUnscreenedCount:
    def test_counts_unscreened_pairs(self, system):
        assert unscreened_pairs(make_result(system, n_pairs=3)) == 3
        assert unscreened_pairs(make_result(system, n_pairs=3, screened=True)) == 0


class TestWorkbookBytes:
    def test_produces_a_real_xlsx_in_memory(self, system):
        data = workbook_bytes([make_result(system)], "ui test")
        assert data[:2] == b"PK", "an xlsx is a zip"
        assert len(data) > 4000

    def test_nothing_is_written_into_the_working_tree(self, system, tmp_path,
                                                      monkeypatch):
        """The workbook carries probe sequences; the download is the only exit."""
        monkeypatch.chdir(tmp_path)
        workbook_bytes([make_result(system)], "ui test")
        assert list(tmp_path.iterdir()) == [], (
            "workbook_bytes must leave no file behind")

    def test_tiled_results_render(self, system):
        data = workbook_bytes([make_result(system, n_pairs=5)], "tiled")
        assert data[:2] == b"PK"


# --- self-folding, P(bound), expression weighting --------------------------

def _hit(query, start, gene):
    return Hit(query=query, chrom=f"tx|{gene}", start=start, end=start + 24,
               strand="+", mismatches=0, gaps=0, aligned=25, query_len=25,
               q_start=1, q_end=25, gene=gene)


def _site(gene, fraction=float("nan"), signal_capable=True):
    return PairSite(chrom=f"tx|{gene}", left=_hit("left", 100, gene),
                    right=_hit("right", 126, gene), gap_nt=1,
                    same_strand=signal_capable, correct_order=True,
                    survives_wash=True, min_margin_c=10.0, gene=gene,
                    left_fraction=fraction, right_fraction=1.0)


def _screened(sites=(), on_target=float("nan"), expression=None, gene_aware=True):
    return SpecificityReport(status=SCREENED, backend="bowtie", database="tx",
                             gene_aware=gene_aware, pair_sites=list(sites),
                             on_target_fraction=on_target, expression=expression)


def _folding(pad_structure):
    def oligo(name):
        return OligoFold(name, "N" * 10, -2.0, pad_structure, pad=(0, 10),
                         arm=(0, 0))
    return PairFolding(status=FOLD_EVALUATED, p1l=oligo("P1L"), p1r=oligo("P1R"),
                       tool="RNAfold", temp_c=37.0, salt_m=0.39)


class TestFolding:
    def test_unevaluated_folding_says_so_in_words(self, system):
        result = make_result(system)
        assert candidate_rows(result)[0]["self-folding"] == "not evaluated"
        assert "NOT EVALUATED" in folding_details(result)[0]

    def test_evaluated_folding_shows_the_limiting_label(self, system):
        result = make_result(system, folding=[_folding("((((..))))")])
        assert candidate_rows(result)[0]["self-folding"] == "folded"
        assert folding_cell(_folding("..........")) == "open"

    def test_details_cover_every_pair_and_carry_the_caveat(self, system):
        result = make_result(system, n_pairs=3, folding=[_folding("((((..))))")])
        lines = folding_details(result)
        assert len(lines) == 3
        assert lines[0].startswith("Pair 1: FOLDED")
        assert "UNCALIBRATED" in lines[0].upper()
        assert "NOT EVALUATED" in lines[1], "missing pairs must not borrow pair 1's"


class TestFractionBound:
    def test_unscreened_says_not_screened(self):
        screen = not_screened("no index")
        assert on_target_cell(screen) == "NOT SCREENED"
        assert worst_offtarget_cell(screen) == "NOT SCREENED"

    def test_nan_means_not_computed_never_a_number(self):
        screen = _screened(sites=[_site("OTHER")])
        assert on_target_cell(screen) == "not computed"
        assert worst_offtarget_cell(screen) == "not computed"

    def test_no_off_target_site_is_said_in_words(self):
        assert worst_offtarget_cell(_screened(on_target=0.9)) == "no off-target site"

    def test_a_site_that_cannot_signal_does_not_set_the_worst(self):
        screen = _screened(sites=[_site("OTHER", 0.5, signal_capable=False)])
        assert worst_offtarget_cell(screen) == "no off-target site"

    def test_values_keep_significant_figures(self):
        """A tiny occupancy rounded to fixed decimals would print as 0."""
        screen = _screened(sites=[_site("A", 3e-5), _site("B", 0.004)],
                           on_target=0.97)
        assert on_target_cell(screen) == "0.97"
        assert worst_offtarget_cell(screen) == "0.004"
        assert worst_offtarget_cell(_screened(sites=[_site("A", 3e-5)])) == "3e-05"

    def test_columns_are_in_the_table(self, system):
        row = candidate_rows(make_result(system))[0]
        assert row["on-target P(bound)"] == "NOT SCREENED"
        assert row["worst off-target P(bound)"] == "NOT SCREENED"


TABLE = ExpressionTable("cells.tsv", {"other": 5.0})


class TestExpressionStatus:
    def test_unscreened_is_not_evaluated_with_reason(self, system):
        weighted, text = expression_status(make_result(system))
        assert not weighted
        assert "NOT EVALUATED" in text and "no screen was run" in text

    def test_no_table_is_not_evaluated(self, system):
        weighted, text = expression_status(
            make_result(system, screens=[_screened()]))
        assert not weighted
        assert "no expression table supplied" in text

    def test_genome_screen_cannot_be_weighted(self, system):
        weighted, text = expression_status(
            make_result(system, screens=[_screened(expression=TABLE,
                                                   gene_aware=False)]))
        assert not weighted
        assert "gene names" in text

    def test_weighted_names_the_source_and_the_caveat(self, system):
        weighted, text = expression_status(
            make_result(system, n_pairs=2, screens=[_screened(expression=TABLE)]))
        assert weighted
        assert "cells.tsv" in text and "UNCALIBRATED" in text

    def test_partial_weighting_is_not_a_pass(self, system):
        result = make_result(system, n_pairs=2,
                             screens=[_screened(expression=TABLE), _screened()])
        weighted, text = expression_status(result)
        assert not weighted
        assert "1 of 2 pair(s)" in text


def test_the_no_change_reply_names_the_run_and_what_would_change_it():
    """Shown instead of designing again when a chat request moves no setting."""
    from mypibypy.ui import no_change_reply

    text = no_change_reply("change", 3)
    assert "run #3 stands" in text and "*change*" in text
    for setting in ("channel", "pairs", "spacing", "screen", "GC/Tm"):
        assert setting in text
    assert "deterministic" in text
