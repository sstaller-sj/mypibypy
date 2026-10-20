"""The off-target sequence, carried from the aligner to thermo.

Pricing a mismatch by position needs to know what the probe actually meets.
Both backends report it -- bowtie through SAM's MD tag, BLAST through sseq -- and
these tests check it survives the trip in the right orientation, and that a hit
whose site cannot be reconstructed degrades to the count model rather than to a
perfect match.
"""
from __future__ import annotations

import math
import shutil
import subprocess

import pytest

from mypibypy import thermo
from mypibypy.sequence import revcomp
from mypibypy.specificity import Backend, _site_from_md, screen_pair

LEFT = "ACTGACAGTCATGCAGTACGTACAG"
RIGHT = "GTCAGTCATGACGTCAGTACTGACT"
FILLER = "GATCTTAGCCAGTTACGATCGGTAACCTGAATCGGCTATTGCAACGTTAGC"

HAVE_BOWTIE = bool(shutil.which("bowtie") and shutil.which("bowtie-build"))
HAVE_BLAST = bool(shutil.which("blastn") and shutil.which("makeblastdb"))


def mut(seq, i, base):
    assert seq[i] != base, f"position {i} is already {base}: not a mismatch"
    return seq[:i] + base + seq[i + 1:]


class TestMd:
    def test_plus_strand(self):
        site = mut(mut(LEFT, 4, "G"), 12, "A")
        # MD carries the *reference* base at each mismatch; SEQ is the read.
        md = "4G7A12"
        assert _site_from_md(LEFT, md, reverse=False) == site

    def test_minus_strand_comes_back_in_query_orientation(self):
        site = mut(LEFT, 4, "G")
        sam_seq = revcomp(LEFT)                    # SEQ is reverse-complemented
        # The reference forward strand is revcomp(site); MD is in that frame.
        fwd = revcomp(site)
        pos = next(i for i, (a, b) in enumerate(zip(sam_seq, fwd)) if a != b)
        md = f"{pos}{fwd[pos]}{len(fwd) - pos - 1}"
        assert _site_from_md(sam_seq, md, reverse=True) == site

    @pytest.mark.parametrize("md", ["", "10^AC15", "30", "5A"])
    def test_unusable_md_gives_no_site(self, md):
        assert _site_from_md(LEFT, md, reverse=False) == ""


class TestParsersCarryTheSite:
    def test_bowtie_sam(self):
        sam = (f"left\t0\tchr1\t101\t255\t25M\t*\t0\t0\t{LEFT}\t{'I' * 25}\t"
               f"XA:i:1\tMD:Z:4G20\tNM:i:1\n")
        (h,) = Backend.parse_sam(sam, {"left": LEFT})
        assert h.site_seq(LEFT) == mut(LEFT, 4, "G")
        assert h.mismatch_offsets(LEFT) == (4,)

    def test_bowtie_sam_without_md_degrades(self):
        sam = f"left\t0\tchr1\t101\t255\t25M\t*\t0\t0\t{LEFT}\t{'I' * 25}\tNM:i:1\n"
        (h,) = Backend.parse_sam(sam, {"left": LEFT})
        assert h.site_seq(LEFT) is None and h.mismatch_offsets(LEFT) == ()

    def test_blast_sseq(self):
        site = mut(LEFT, 10, "G")
        line = "\t".join(["left", "chr1", "96.0", "25", "1", "0", "1", "25", "101",
                          "125", "1e-5", "40", "25", LEFT, site])
        (h,) = Backend.parse_blast(line, {"left": LEFT})
        assert h.site_seq(LEFT) == site

    def test_blast_gapped_alignment_has_no_site(self):
        line = "\t".join(["left", "chr1", "96.0", "25", "0", "1", "1", "25", "101",
                          "124", "1e-5", "40", "25", LEFT, LEFT[:10] + "-" + LEFT[11:]])
        (h,) = Backend.parse_blast(line, {"left": LEFT})
        assert h.site_seq(LEFT) is None

    def test_blast_legacy_thirteen_columns_still_parse(self):
        line = "\t".join(["left", "chr1", "100", "25", "0", "0", "1", "25", "101",
                          "125", "1e-5", "40", "25"])
        (h,) = Backend.parse_blast(line, {"left": LEFT})
        assert h.ref_seq == "" and h.aligned == 25


def _decoy(tmp_path, sites, length=20000):
    seq = list((FILLER * (length // len(FILLER) + 1))[:length])
    for pos, sub in sites.items():
        seq[pos : pos + len(sub)] = list(sub)
    fa = tmp_path / "decoy.fa"
    fa.write_text(">decoy1\n" + "".join(seq) + "\n")
    return str(fa)


def _bowtie(tmp_path, fa):
    idx = str(tmp_path / "bt")
    subprocess.run(["bowtie-build", "-q", fa, idx], capture_output=True, check=True)
    return Backend(tool="bowtie", database=idx, executable=shutil.which("bowtie"))


def _blast(tmp_path, fa):
    db = str(tmp_path / "db")
    subprocess.run(["makeblastdb", "-in", fa, "-dbtype", "nucl", "-out", db],
                   capture_output=True, check=True)
    return Backend(tool="blastn", database=db, executable=shutil.which("blastn"))


@pytest.fixture(params=[
    pytest.param("bowtie", marks=pytest.mark.skipif(not HAVE_BOWTIE, reason="no bowtie v1")),
    pytest.param("blastn", marks=pytest.mark.skipif(not HAVE_BLAST, reason="no blastn")),
])
def make_backend(request):
    return {"bowtie": _bowtie, "blastn": _blast}[request.param]


class TestEndToEnd:
    @pytest.mark.parametrize("strand", ["+", "-"])
    def test_site_model_is_used_on_real_alignments(self, tmp_path, make_backend, strand):
        left_site = mut(mut(LEFT, 8, "A"), 16, "C")
        footprint = left_site + "T" + RIGHT
        planted = footprint if strand == "+" else revcomp(footprint)
        rep = screen_pair(LEFT, RIGHT, backend=make_backend(tmp_path, _decoy(tmp_path, {4000: planted})))
        sites = [s for s in rep.pair_sites if s.signal_capable]
        assert sites, rep.summary()
        site = sites[0]
        assert site.left.site_seq(LEFT) == left_site.upper()
        assert site.thermo_model == thermo.MODEL_NN_MISMATCH + " / " + thermo.MODEL_PERFECT
        assert not math.isnan(site.fraction_bound)
        assert site.fraction_bound < rep.on_target_fraction
        assert "P(bound)" in str(site)

    def test_on_target_fraction_is_reported(self, tmp_path, make_backend):
        rep = screen_pair(LEFT, RIGHT, backend=make_backend(tmp_path, _decoy(tmp_path, {})))
        assert rep.verdict == "pass"
        assert 0.9 < rep.on_target_fraction <= 1.0
        assert "on-target P(bound)" in rep.summary()
        assert math.isnan(rep.worst_offtarget_fraction)

    def test_mismatch_position_changes_the_wash_verdict(self, tmp_path, make_backend):
        """Two mismatches at the ends hold; two in the middle melt. Same count."""
        ends = mut(mut(LEFT, 0, "T"), 24, "C")
        middle = mut(mut(LEFT, 8, "A"), 16, "C")
        reps = {}
        for name, left_site in (("ends", ends), ("middle", middle)):
            d = tmp_path / name
            d.mkdir()
            fa = _decoy(d, {4000: left_site + "T" + RIGHT})
            reps[name] = screen_pair(LEFT, RIGHT, backend=make_backend(d, fa))
        end_site = [s for s in reps["ends"].pair_sites if s.signal_capable]
        mid_site = [s for s in reps["middle"].pair_sites if s.signal_capable]
        assert end_site and mid_site
        assert end_site[0].min_margin_c > mid_site[0].min_margin_c + 5
