"""Pair-aware specificity screening.

Two layers. The pure-Python layer (parsing, pairing geometry, thermodynamic
triage, the degraded NOT_SCREENED path) runs everywhere with no aligners. The
integration layer builds a small decoy database on the fly and is skipped when
blastn/bowtie are absent -- see README for the conda environment.

`TestPairAwarenessIsReal` is the test that matters. The whole premise of this
module is that a single-half genome hit is not an off-target and a pair of
nearby hits is; a screen that could not tell those apart would still pass every
other test here.

No real sequences. Footprints are synthetic, and the decoy genome is generated.
"""
from __future__ import annotations

import shutil
import subprocess
import textwrap

import pytest

from mypibypy import thermo
from mypibypy.sequence import GenomicInterval, revcomp
from mypibypy.specificity import (
    DEFAULT_MIN_ALIGNED_NT,
    DEFAULT_SIGNAL_GAP_NT,
    ERROR,
    NOT_SCREENED,
    SCREENED,
    Backend,
    Hit,
    SpecificityNotConfigured,
    bowtie_backend,
    not_screened,
    pair_hits,
    resolve_backend,
    screen_pair,
)

# Synthetic 25-nt footprints: balanced GC, no long runs, not from any locus.
LEFT = "ACTGACAGTCATGCAGTACGTACAG"
RIGHT = "GTCAGTCATGACGTCAGTACTGACT"


def hit(query="left", chrom="chr1", start=1000, strand="+", mismatches=0,
        aligned=25, query_len=25, gaps=0, q_start=1, q_end=None):
    return Hit(query=query, chrom=chrom, start=start, end=start + aligned - 1,
               strand=strand, mismatches=mismatches, gaps=gaps, aligned=aligned,
               query_len=query_len, q_start=q_start,
               q_end=q_end if q_end is not None else q_start + aligned - 1)


class TestHit:
    def test_full_length_hit_has_nothing_trimmed(self):
        h = hit()
        assert h.unaligned == 0
        assert h.effective_mismatches == 0
        assert h.duplex_seq(LEFT) == LEFT

    def test_trimmed_alignment_reports_the_duplex_that_actually_forms(self):
        # BLAST trimmed 5 nt off the 3' end: a 20-bp duplex, not a 25-bp one.
        h = hit(aligned=20, q_start=1, q_end=20)
        assert h.unaligned == 5
        assert h.duplex_seq(LEFT) == LEFT[:20]
        assert len(h.duplex_seq(LEFT)) == 20

    def test_trimmed_5_prime_takes_the_right_bases(self):
        h = hit(aligned=20, q_start=6, q_end=25)
        assert h.duplex_seq(LEFT) == LEFT[5:]

    def test_effective_mismatches_combines_substitutions_gaps_and_trimming(self):
        assert hit(mismatches=2, gaps=1, aligned=23, q_end=23).effective_mismatches == 5

    def test_overlaps_on_target_interval(self):
        h = hit(chrom="chr5", start=1000)
        iv = GenomicInterval("mm10", "chr5", 990, 1010)
        assert h.overlaps(iv)
        assert not h.overlaps(GenomicInterval("mm10", "chr5", 2000, 2050))
        assert not h.overlaps(GenomicInterval("mm10", "chr7", 990, 1010))


class TestParsers:
    def test_blast_tabular_roundtrip(self):
        # qseqid sseqid pident length mismatch gapopen qstart qend sstart send
        # evalue bitscore qlen
        tab = textwrap.dedent("""\
            left\tchr1\t100.000\t25\t0\t0\t1\t25\t5000\t5024\t1e-05\t50.1\t25
            right\tchr1\t92.000\t25\t2\t0\t1\t25\t5100\t5076\t1e-03\t40.1\t25
            """)
        hits = Backend.parse_blast(tab, {"left": LEFT, "right": RIGHT})
        assert len(hits) == 2
        l, r = hits
        assert (l.query, l.chrom, l.start, l.end, l.strand) == ("left", "chr1", 5000, 5024, "+")
        assert l.mismatches == 0 and l.aligned == 25
        # sstart > send means the subject aligned in reverse.
        assert r.strand == "-" and (r.start, r.end) == (5076, 5100)
        assert r.mismatches == 2

    def test_blast_partial_alignment_records_query_coordinates(self):
        tab = "left\tchr2\t100.000\t18\t0\t0\t8\t25\t900\t917\t1e-02\t30.0\t25\n"
        (h,) = Backend.parse_blast(tab, {"left": LEFT})
        assert h.aligned == 18 and h.q_start == 8 and h.q_end == 25
        assert h.duplex_seq(LEFT) == LEFT[7:]

    def test_blast_ignores_comments_and_short_lines(self):
        assert Backend.parse_blast("# comment\n\nleft\tchr1\n", {"left": LEFT}) == []

    def test_sam_roundtrip_with_nm_tag(self):
        sam = textwrap.dedent("""\
            @HD\tVN:1.0
            @SQ\tSN:chr1\tLN:100000
            left\t0\tchr1\t5000\t255\t25M\t*\t0\t0\t{L}\tIIIIIIIIIIIIIIIIIIIIIIIII\tNM:i:0
            right\t16\tchr1\t5100\t255\t25M\t*\t0\t0\t{R}\tIIIIIIIIIIIIIIIIIIIIIIIII\tNM:i:2
            """).format(L=LEFT, R=revcomp(RIGHT))
        hits = Backend.parse_sam(sam, {"left": LEFT, "right": RIGHT})
        assert len(hits) == 2
        assert hits[0].strand == "+" and hits[0].mismatches == 0
        assert hits[1].strand == "-" and hits[1].mismatches == 2
        # bowtie -v N is end-to-end: nothing is ever trimmed.
        assert all(h.unaligned == 0 for h in hits)

    def test_sam_skips_unmapped_reads(self):
        sam = f"left\t4\t*\t0\t0\t*\t*\t0\t0\t{LEFT}\t*\n"
        assert Backend.parse_sam(sam, {"left": LEFT}) == []


class TestPairingGeometry:
    """Both halves bind the plus strand with a 1-nt gap. Off-target sites must
    reproduce that geometry to make a spot."""

    def test_designed_geometry_is_signal_capable(self):
        left = hit("left", start=1000)              # 1000-1024
        right = hit("right", start=1026)            # 1-nt gap at 1025
        (site,) = pair_hits([left], [right], LEFT, RIGHT)
        assert site.gap_nt == 1
        assert site.same_strand and site.correct_order and site.signal_capable

    def test_minus_strand_site_is_recognised_with_order_reversed(self):
        # On a reverse-complemented copy of the target, right25 precedes left25.
        right = hit("right", start=1000, strand="-")
        left = hit("left", start=1026, strand="-")
        (site,) = pair_hits([left], [right], LEFT, RIGHT)
        assert site.gap_nt == 1 and site.correct_order and site.signal_capable

    def test_wrong_order_is_reported_but_cannot_signal(self):
        right = hit("right", start=1000)
        left = hit("left", start=1026)
        (site,) = pair_hits([left], [right], LEFT, RIGHT)
        assert site.same_strand and not site.correct_order
        assert not site.signal_capable, "reversed halves cannot rebuild the P2 site"

    def test_opposite_strands_cannot_signal(self):
        left = hit("left", start=1000, strand="+")
        right = hit("right", start=1026, strand="-")
        (site,) = pair_hits([left], [right], LEFT, RIGHT)
        assert not site.same_strand and not site.signal_capable

    def test_different_chromosomes_never_pair(self):
        assert pair_hits([hit("left", chrom="chr1")],
                         [hit("right", chrom="chr2")], LEFT, RIGHT) == []

    def test_beyond_the_window_does_not_pair(self):
        left = hit("left", start=1000)
        far = hit("right", start=1000 + 5000)
        assert pair_hits([left], [far], LEFT, RIGHT, window_nt=200) == []
        assert len(pair_hits([left], [far], LEFT, RIGHT, window_nt=10000)) == 1

    def test_inside_the_window_but_too_far_apart_is_only_co_localized(self):
        left = hit("left", start=1000)
        right = hit("right", start=1150)
        (site,) = pair_hits([left], [right], LEFT, RIGHT, window_nt=200)
        assert site.gap_nt == 125
        assert not site.signal_capable, "125 nt is past even the flexible-arm reach"

    def test_the_signal_window_is_configurable(self):
        left = hit("left", start=1000)
        right = hit("right", start=1050)   # 25 nt gap
        (loose,) = pair_hits([left], [right], LEFT, RIGHT, signal_gap_nt=30)
        (tight,) = pair_hits([left], [right], LEFT, RIGHT, signal_gap_nt=10)
        assert loose.signal_capable and not tight.signal_capable

    def test_arms_bridge_tens_of_nt_so_a_40_nt_gap_still_signals(self):
        """The corrected threshold, and the reason for it.

        A 10-nt window was originally used here. The lab's own hand-made
        antisense fusion pairs anchor their arms 50 nt apart and give signal at
        the bench, so an off-target pair tens of nt apart is signal-capable too.
        Under the old window this site came back merely "co-localized" -- the
        unsafe direction for a specificity screen.
        """
        left = hit("left", start=1000)
        right = hit("right", start=1065)   # 40 nt between the footprints
        (site,) = pair_hits([left], [right], LEFT, RIGHT)
        assert site.gap_nt == 40
        assert site.signal_capable, (
            f"a 40-nt gap must be treated as signal-capable; "
            f"DEFAULT_SIGNAL_GAP_NT={DEFAULT_SIGNAL_GAP_NT}")
        assert not pair_hits([left], [right], LEFT, RIGHT,
                             signal_gap_nt=10)[0].signal_capable

    def test_the_default_window_is_at_least_the_bench_observed_separation(self):
        """A designed pair is known to work with its arms 50 nt apart, so the
        screen must not call anything inside that merely co-localized."""
        assert DEFAULT_SIGNAL_GAP_NT >= 50, (
            "lowering this below the separation a working design already uses "
            "would hide genuinely signal-capable off-target pairs")


class TestThermoTriage:
    """'The aligner found it' is not 'it would still be bound'."""

    def test_perfect_offtarget_pair_survives_the_wash(self):
        (site,) = pair_hits([hit("left", start=1000)], [hit("right", start=1026)],
                            LEFT, RIGHT)
        assert site.survives_wash and site.min_margin_c > 0

    def test_three_mismatches_per_half_melts_out(self):
        left = hit("left", start=1000, mismatches=3)
        right = hit("right", start=1026, mismatches=3)
        (site,) = pair_hits([left], [right], LEFT, RIGHT)
        assert site.signal_capable, "geometry is still right"
        assert not site.survives_wash, "but it comes apart in the wash"
        assert site.min_margin_c < 0

    def test_thermo_is_evaluated_on_the_aligned_block_not_the_whole_pad(self):
        """A trimmed hit makes a shorter, weaker duplex -- and must be judged as one."""
        full = hit("left", start=1000)
        trimmed = hit("left", start=1000, aligned=16, q_start=1, q_end=16)
        right = hit("right", start=1026)
        (a,) = pair_hits([full], [right], LEFT, RIGHT)
        (b,) = pair_hits([trimmed], [right], LEFT, RIGHT)
        assert b.min_margin_c < a.min_margin_c

    def test_mismatch_penalty_is_tunable(self):
        left = hit("left", start=1000, mismatches=2)
        right = hit("right", start=1026, mismatches=2)
        (strict,) = pair_hits([left], [right], LEFT, RIGHT, penalty_c=12.0)
        (lenient,) = pair_hits([left], [right], LEFT, RIGHT, penalty_c=2.0)
        assert lenient.min_margin_c > strict.min_margin_c
        assert lenient.survives_wash and not strict.survives_wash


class TestDegradesGracefully:
    """A screen that did not run must never look like a screen that passed."""

    def test_no_database_yields_not_screened_not_an_exception(self, monkeypatch):
        monkeypatch.delenv("MYPIBYPY_BOWTIE_INDEX", raising=False)
        monkeypatch.delenv("MYPIBYPY_BLAST_DB", raising=False)
        rep = screen_pair(LEFT, RIGHT)
        assert rep.status == NOT_SCREENED
        assert not rep.screened
        assert rep.verdict == "not screened"
        assert "NOT SCREENED" in rep.summary()
        assert rep.reason

    def test_not_screened_has_no_score_to_mistake_for_a_pass(self, monkeypatch):
        monkeypatch.delenv("MYPIBYPY_BOWTIE_INDEX", raising=False)
        monkeypatch.delenv("MYPIBYPY_BLAST_DB", raising=False)
        import math
        assert math.isnan(screen_pair(LEFT, RIGHT).score)
        assert math.isnan(not_screened("because").score)

    def test_resolve_backend_names_the_env_vars_it_wants(self, monkeypatch):
        monkeypatch.delenv("MYPIBYPY_BOWTIE_INDEX", raising=False)
        monkeypatch.delenv("MYPIBYPY_BLAST_DB", raising=False)
        with pytest.raises(SpecificityNotConfigured) as exc:
            resolve_backend()
        assert "MYPIBYPY_BOWTIE_INDEX" in str(exc.value)
        assert "MYPIBYPY_BLAST_DB" in str(exc.value)

    def test_a_named_bowtie_index_never_falls_back_to_blast(self, tmp_path, monkeypatch):
        """Asked for one index by name, the caller must get that index or a
        reason -- never a quiet switch to $MYPIBYPY_BLAST_DB."""
        monkeypatch.setenv("MYPIBYPY_BLAST_DB", str(tmp_path / "somedb"))
        monkeypatch.setenv("PATH", "")
        index = str(tmp_path / "hg38")
        with pytest.raises(SpecificityNotConfigured, match="bowtie not on PATH"):
            bowtie_backend(index)
        monkeypatch.setattr(shutil, "which", lambda tool, *a, **k: f"/fake/bin/{tool}")
        with pytest.raises(SpecificityNotConfigured, match="not found"):
            bowtie_backend(index)
        for suffix in ("1", "2", "3", "4", "rev.1", "rev.2"):
            (tmp_path / f"hg38.{suffix}.ebwt").write_text("")
        backend = bowtie_backend(index)
        assert backend.tool == "bowtie" and backend.database == index
        assert backend.executable == "/fake/bin/bowtie"

    def test_missing_database_file_is_not_screened(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MYPIBYPY_BOWTIE_INDEX", raising=False)
        monkeypatch.setenv("MYPIBYPY_BLAST_DB", str(tmp_path / "nope"))
        rep = screen_pair(LEFT, RIGHT)
        assert rep.status == NOT_SCREENED and "not found" in rep.reason

    def test_a_crashing_aligner_reports_error_not_success(self, tmp_path):
        broken = Backend(tool="blastn", database=str(tmp_path / "nope"),
                         executable=shutil.which("false") or "/usr/bin/false")
        rep = screen_pair(LEFT, RIGHT, backend=broken)
        assert rep.status == ERROR and not rep.screened
        assert rep.verdict == "not screened"


# --- Integration: real aligners against a generated decoy ------------------

HAVE_BLAST = bool(shutil.which("blastn") and shutil.which("makeblastdb"))
HAVE_BOWTIE = bool(shutil.which("bowtie") and shutil.which("bowtie-build"))

FILLER = "GATCTTAGCCAGTTACGATCGGTAACCTGAATCGGCTATTGCAACGTTAGC"


def _decoy(tmp_path, sites: dict[int, str], length: int = 20000) -> str:
    """A decoy chromosome of neutral filler with `sites` planted at given offsets."""
    seq = list((FILLER * (length // len(FILLER) + 1))[:length])
    for pos, sub in sites.items():
        seq[pos : pos + len(sub)] = list(sub)
    fa = tmp_path / "decoy.fa"
    fa.write_text(">decoy1\n" + "".join(seq) + "\n")
    return str(fa)


def _blast_backend(tmp_path, fasta):
    db = str(tmp_path / "decoydb")
    subprocess.run(["makeblastdb", "-in", fasta, "-dbtype", "nucl", "-out", db],
                   capture_output=True, check=True)
    return Backend(tool="blastn", database=db, executable=shutil.which("blastn"))


def _bowtie_backend(tmp_path, fasta):
    idx = str(tmp_path / "decoybt")
    subprocess.run(["bowtie-build", "-q", fasta, idx], capture_output=True, check=True)
    return Backend(tool="bowtie", database=idx, executable=shutil.which("bowtie"))


@pytest.fixture(params=[
    pytest.param("blastn", marks=pytest.mark.skipif(not HAVE_BLAST, reason="no blastn")),
    pytest.param("bowtie", marks=pytest.mark.skipif(not HAVE_BOWTIE, reason="no bowtie v1")),
])
def backend_factory(request):
    return {"blastn": _blast_backend, "bowtie": _bowtie_backend}[request.param]


class TestPairAwarenessIsReal:
    """The premise of the module, tested against real aligners.

    A lone half-probe hit cannot recruit P2 and so cannot make a spot; two hits
    landing adjacent can. No general-purpose tool distinguishes these, and if
    this module did not either, it would be a worse BLAST wrapper.
    """

    def test_lone_half_hit_makes_no_site_but_costs_score(self, tmp_path, backend_factory):
        # Only left25 exists elsewhere in the genome.
        fasta = _decoy(tmp_path, {4000: LEFT})
        rep = screen_pair(LEFT, RIGHT, backend=backend_factory(tmp_path, fasta))
        assert rep.screened
        assert rep.left_hits, "the lone half must still be found"
        assert rep.pair_sites == [], "one half alone is not an off-target site"
        assert rep.verdict != "fail"
        assert rep.score < 1.0, "it is not free, either"

    def test_both_halves_adjacent_is_found_and_fails(self, tmp_path, backend_factory):
        # left25 + 1-nt gap + right25: the real footprint, somewhere else.
        fasta = _decoy(tmp_path, {4000: LEFT + "T" + RIGHT})
        rep = screen_pair(LEFT, RIGHT, backend=backend_factory(tmp_path, fasta))
        assert rep.screened
        sites = [s for s in rep.pair_sites if s.signal_capable]
        assert sites, f"a duplicated footprint must be found: {rep.summary()}"
        assert sites[0].gap_nt == 1
        assert sites[0].survives_wash
        assert rep.verdict == "fail"
        assert rep.score < 0.5

    def test_halves_far_apart_do_not_pair(self, tmp_path, backend_factory):
        fasta = _decoy(tmp_path, {4000: LEFT, 12000: RIGHT})
        rep = screen_pair(LEFT, RIGHT, backend=backend_factory(tmp_path, fasta),
                          window_nt=200)
        assert rep.screened and rep.left_hits and rep.right_hits
        assert rep.pair_sites == [], "8 kb apart is not a piFISH site"

    def test_on_target_site_is_excluded(self, tmp_path, backend_factory):
        fasta = _decoy(tmp_path, {4000: LEFT + "T" + RIGHT})
        on_target = GenomicInterval("decoy", "decoy1", 4001, 4051)
        rep = screen_pair(LEFT, RIGHT, backend=backend_factory(tmp_path, fasta),
                          on_target=on_target)
        assert rep.screened
        assert rep.pair_sites == [], "the design must not flag itself"
        assert rep.verdict == "pass"

    def test_reverse_complement_copy_is_found(self, tmp_path, backend_factory):
        """Both halves on the minus strand is the same site, read the other way."""
        fasta = _decoy(tmp_path, {4000: revcomp(LEFT + "T" + RIGHT)})
        rep = screen_pair(LEFT, RIGHT, backend=backend_factory(tmp_path, fasta))
        assert rep.screened
        sites = [s for s in rep.pair_sites if s.signal_capable]
        assert sites, f"minus-strand copy missed: {rep.summary()}"
        assert sites[0].left.strand == sites[0].right.strand == "-"

    def test_clean_genome_passes(self, tmp_path, backend_factory):
        rep = screen_pair(LEFT, RIGHT, backend=backend_factory(tmp_path, _decoy(tmp_path, {})))
        assert rep.screened
        assert rep.pair_sites == []
        assert rep.verdict == "pass" and rep.score == 1.0


@pytest.mark.skipif(not HAVE_BOWTIE, reason="no bowtie v1")
class TestBowtieSensitivity:
    """bowtie -v N -a is exhaustive within N mismatches. Confirm that, because
    the whole backend preference rests on it."""

    @pytest.mark.parametrize("n_mismatches", [0, 1, 2, 3])
    def test_recovers_a_spiked_site_at_every_mismatch_count(
            self, tmp_path, n_mismatches):
        variant = list(LEFT)
        for i, pos in enumerate((4, 11, 18, 21)[:n_mismatches]):
            variant[pos] = {"A": "C", "C": "A", "G": "T", "T": "G"}[variant[pos]]
        fasta = _decoy(tmp_path, {4000: "".join(variant)})
        rep = screen_pair(LEFT, RIGHT, backend=_bowtie_backend(tmp_path, fasta))
        found = [h for h in rep.left_hits if h.mismatches == n_mismatches]
        assert found, f"{n_mismatches} mismatches must be recovered: {rep.summary()}"

    def test_beyond_the_budget_is_honestly_missed(self, tmp_path):
        """Four mismatches is outside -v 3. The report must not imply otherwise."""
        variant = list(LEFT)
        for pos in (4, 9, 14, 19):
            variant[pos] = {"A": "C", "C": "A", "G": "T", "T": "G"}[variant[pos]]
        fasta = _decoy(tmp_path, {4000: "".join(variant)})
        rep = screen_pair(LEFT, RIGHT, backend=_bowtie_backend(tmp_path, fasta))
        assert rep.left_hits == []
        # The budget is stated in the report, so a reader can see what was asked.
        assert rep.max_mismatches == 3
        assert "3 mismatches" in rep.backend


@pytest.mark.skipif(not HAVE_BLAST, reason="no blastn")
def test_dust_filtering_would_destroy_short_queries(tmp_path):
    """Why `-dust no` is not optional.

    With DUST on, a 25-nt query is treated as low-complexity and masked, and
    BLAST returns nothing at all -- a silent clean result. This pins that down
    so the flag is never 'tidied away'.
    """
    fasta = _decoy(tmp_path, {4000: LEFT + "T" + RIGHT})
    db = str(tmp_path / "db")
    subprocess.run(["makeblastdb", "-in", fasta, "-dbtype", "nucl", "-out", db],
                   capture_output=True, check=True)
    q = tmp_path / "q.fa"
    q.write_text(f">left\n{LEFT}\n")
    common = [shutil.which("blastn"), "-task", "blastn-short", "-db", db,
              "-query", str(q), "-outfmt", "6", "-evalue", "1000", "-word_size", "5"]
    on = subprocess.run(common + ["-dust", "yes", "-soft_masking", "true"],
                        capture_output=True, text=True, check=True).stdout
    off = subprocess.run(common + ["-dust", "no", "-soft_masking", "false"],
                         capture_output=True, text=True, check=True).stdout
    assert off.strip(), "with -dust no the planted site must be found"
    assert len(off.splitlines()) >= len(on.splitlines())


def test_min_aligned_prefilter_is_derived_from_thermo():
    """The short-alignment cutoff should track what actually melts.

    `DEFAULT_MIN_ALIGNED_NT` claims that below ~15 bp nothing survives the wash.
    If `thermo` or the wash series changes, this catches the constant going stale
    instead of letting it quietly start discarding real sites.
    """
    gc_balanced = "ACGTGCATGCATGCATGCATGCATG"
    below = gc_balanced[: DEFAULT_MIN_ALIGNED_NT - 2]
    assert not thermo.survives(below, 0), (
        f"a {len(below)}-nt perfect duplex now survives the wash; "
        f"DEFAULT_MIN_ALIGNED_NT={DEFAULT_MIN_ALIGNED_NT} is too high")
