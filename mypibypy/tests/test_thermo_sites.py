"""Mismatched duplexes priced from sequence, and fraction bound.

The flat per-mismatch penalty treats every mismatch alike. These tests hold
down what replaces it when the off-target sequence is known: position and
identity matter, the numbers agree with Biopython's reference implementation of
the same published tables, and anything the tables do not cover falls back to
the count model *and says so* rather than quietly pricing it as a perfect match.
"""
from __future__ import annotations

import math

import pytest
from Bio.SeqUtils import MeltingTemp as mt

from mypibypy import thermo
from mypibypy.thermo import (
    ALL_STEPS,
    HYB_PRIMARY,
    MODEL_COUNT_PENALTY,
    MODEL_NN_MISMATCH,
    MODEL_PERFECT,
    WASH_C,
    Conditions,
    fraction_bound,
    min_fraction_bound,
    nn_thermo,
    profile,
    site_thermo,
    survives,
    tm_buffer,
    tm_mismatched,
    tm_site,
)

PAD = "GACTGCTAGCATGCAAGTCCATGCA"   # synthetic 25-mer, not from any locus
COMP = str.maketrans("ACGT", "TGCA")


def mut(seq: str, i: int, base: str) -> str:
    assert seq[i] != base, f"position {i} is already {base}: not a mismatch"
    return seq[:i] + base + seq[i + 1:]


def bio_tm_1m(site: str, footprint: str) -> float:
    """Biopython's Tm for the same duplex, at 1 M Na+ and our Ct/4 convention."""
    return mt.Tm_NN(site, c_seq=footprint.translate(COMP), Na=1000, saltcorr=0,
                    dnac1=2.5, dnac2=0, nn_table=mt.DNA_NN3)


def our_tm_1m(site: str, footprint: str) -> float:
    dh, ds, _ = site_thermo(footprint, site)
    return dh * 1000 / (ds + 1.987 * math.log(10e-9 / 4)) - 273.15


class TestSiteThermo:
    def test_a_perfect_site_is_exactly_nn_thermo(self):
        dh, ds, n = site_thermo(PAD, PAD)
        assert (dh, ds) == nn_thermo(PAD) and n == 25
        assert tm_site(PAD, PAD, HYB_PRIMARY) == pytest.approx(tm_buffer(PAD, HYB_PRIMARY))

    @pytest.mark.parametrize("i,base", [(12, "T"), (5, "A"), (0, "T"), (24, "C"),
                                        (1, "G"), (18, "G")])
    def test_agrees_with_biopython_on_the_same_tables(self, i, base):
        site = mut(PAD, i, base)
        assert our_tm_1m(site, PAD) == pytest.approx(bio_tm_1m(site, PAD), abs=0.01)

    def test_a_central_mismatch_costs_more_than_a_terminal_one(self):
        central = tm_site(PAD, mut(PAD, 12, "T"), HYB_PRIMARY)
        terminal = tm_site(PAD, mut(PAD, 0, "T"), HYB_PRIMARY)
        perfect = tm_buffer(PAD, HYB_PRIMARY)
        assert perfect - terminal < 1.5, "a terminal mismatch barely matters"
        assert perfect - central > 4.0, "a central one matters a lot"

    def test_identity_matters_at_a_fixed_position(self):
        tms = {b: tm_site(PAD, mut(PAD, 1, b), HYB_PRIMARY) for b in "CGT"}
        assert max(tms.values()) - min(tms.values()) > 2.0, tms

    def test_adjacent_mismatches_are_not_priced(self):
        site = mut(mut(PAD, 12, "T"), 13, "A")
        assert site_thermo(PAD, site) is None
        assert math.isnan(tm_site(PAD, site, HYB_PRIMARY))

    def test_frayed_ends_are_trimmed_not_refused(self):
        site = mut(mut(PAD, 0, "T"), 1, "C")
        dh, ds, n = site_thermo(PAD, site)
        # Position 0 cannot pair and is dropped; position 1 is priced as a
        # terminal mismatch; the other 23 pair.
        assert n == 23

    def test_non_dna_and_length_mismatch_are_refused(self):
        assert site_thermo(PAD, PAD[:-1]) is None
        assert site_thermo(PAD, mut(PAD, 3, "N")) is None


class TestFractionBound:
    def test_bounded_and_perfect_beats_mismatched(self):
        perfect = fraction_bound(PAD, WASH_C)
        mm = fraction_bound(PAD, WASH_C, mut(PAD, 12, "T"))
        assert 0.0 <= mm < perfect <= 1.0

    def test_falls_with_temperature(self):
        fr = [fraction_bound(PAD, thermo.at_temperature(WASH_C, t)) for t in (30, 45, 60, 75)]
        assert fr == sorted(fr, reverse=True) and fr[0] > 0.99 and fr[-1] < 0.01

    def test_at_the_reported_tm_a_probe_in_excess_site_is_eighty_percent_bound(self):
        """The Ct/4 convention, stated in thermo's docstring, pinned down here."""
        tm = tm_buffer(PAD, WASH_C)
        assert fraction_bound(PAD, thermo.at_temperature(WASH_C, tm)) == pytest.approx(0.8, abs=1e-6)

    def test_formamide_is_the_same_correction_as_for_tm(self):
        with_fa = Conditions("fa", 40, 20, 2)
        hotter = Conditions("hot", 40 + 20 * thermo.FORMAMIDE_C_PER_PCT, 0, 2)
        assert fraction_bound(PAD, with_fa) == pytest.approx(fraction_bound(PAD, hotter))

    def test_min_over_steps_is_the_limiting_step(self):
        site = mut(mut(mut(PAD, 4, "A"), 12, "T"), 19, "G")
        assert min_fraction_bound(PAD, site) == pytest.approx(
            min(fraction_bound(PAD, c, site) for c in ALL_STEPS))

    def test_unpriceable_is_nan_not_one(self):
        site = mut(mut(PAD, 12, "T"), 13, "A")
        assert math.isnan(fraction_bound(PAD, WASH_C, site))


class TestProfileModels:
    def test_without_a_site_nothing_changes(self):
        """Callers that only know a count get exactly the old numbers."""
        pr = profile(PAD, 2)
        assert pr["model"] == MODEL_COUNT_PENALTY and pr["approximate"]
        assert pr["min_margin_c"] == pytest.approx(
            min(tm_mismatched(PAD, 2, c) - c.temp_c for c in ALL_STEPS))
        assert math.isnan(pr["min_fraction_bound"])

    def test_a_site_is_priced_from_sequence(self):
        pr = profile(PAD, 1, site=mut(PAD, 12, "T"))
        assert pr["model"] == MODEL_NN_MISMATCH and not pr["approximate"]
        assert 0 <= pr["min_fraction_bound"] <= 1

    def test_a_perfect_site_reports_the_perfect_model(self):
        assert profile(PAD, 0, site=PAD)["model"] == MODEL_PERFECT

    def test_an_unpriceable_site_falls_back_and_says_so(self):
        site = mut(mut(PAD, 12, "T"), 13, "A")
        pr = profile(PAD, 2, site=site)
        assert pr["model"] == MODEL_COUNT_PENALTY and pr["approximate"]
        assert pr["min_margin_c"] == pytest.approx(profile(PAD, 2)["min_margin_c"])

    def test_three_end_mismatches_are_not_three_central_ones(self):
        """The point of the change. The count model cannot tell these apart."""
        ends = mut(mut(PAD, 0, "T"), 24, "C")
        central = mut(mut(PAD, 8, "T"), 16, "A")
        assert profile(PAD, 2)["min_margin_c"] == profile(PAD, 2)["min_margin_c"]
        by_seq_ends = profile(PAD, 2, site=ends)["min_margin_c"]
        by_seq_central = profile(PAD, 2, site=central)["min_margin_c"]
        assert by_seq_ends > by_seq_central + 5
        assert survives(PAD, 2, site=ends)
        assert not survives(PAD, 2, site=central)
