"""Expression-weighted off-target scoring.

Two things matter more than the weighting curve itself. A missing table must
read as NOT EVALUATED and change nothing, and a gene the table does not list
must keep its full penalty -- absent is not the same as unexpressed, and
treating it as zero would turn an off-target into a clean report.
"""
from __future__ import annotations

import random
import shutil
import subprocess

import pytest

from mypibypy.expression import (
    ENV_EXPRESSION,
    ExpressionTable,
    load_expression,
    parse_expression,
    resolve_expression,
)
from mypibypy.specificity import (
    SCREENED,
    Backend,
    Hit,
    PairSite,
    SpecificityReport,
    screen_pair,
)

LEFT = "ACTGACAGTCATGCAGTACGTACAG"
RIGHT = "GTCAGTCATGACGTCAGTACTGACT"


class TestParsing:
    def test_two_columns_without_a_header(self):
        t = parse_expression("ACTB\t812.5\nGAPDH\t400\n")
        assert t.lookup("ACTB") == 812.5 and t.lookup("gapdh") == 400 and t.n_rows == 2

    def test_named_columns_in_csv(self):
        t = parse_expression("gene_name,length,TPM\nMYC,2000,12\nFOS,1500,0.3\n")
        assert t.lookup("MYC") == 12 and t.lookup("FOS") == 0.3

    def test_rsem_genes_results(self):
        text = ("gene_id\ttranscript_id(s)\tlength\teffective_length\texpected_count\tTPM\tFPKM\n"
                "ENSG00000075624.17\tENST1,ENST2\t1800\t1650\t9000\t650.0\t500\n")
        t = parse_expression(text)
        assert t.lookup("ENSG00000075624.17") == 650.0
        assert t.lookup("ENSG00000075624") == 650.0, "unversioned ID matches too"
        assert t.lookup("ENSG00000075624.18") == 650.0, "and so does another version"

    def test_salmon_quant_sums_isoforms_into_genes(self):
        hdr = lambda n, g: f"ENST{n:011d}.1|ENSG{n:011d}.1|-|-|{g}-20{n}|{g}|1500|protein_coding|"
        text = ("Name\tLength\tEffectiveLength\tTPM\tNumReads\n"
                f"{hdr(1, 'MYGENE')}\t1500\t1300\t3.0\t10\n"
                f"{hdr(2, 'MYGENE')}\t1500\t1300\t4.5\t12\n"
                f"{hdr(3, 'OTHER')}\t1500\t1300\t1.0\t2\n")
        t = parse_expression(text)
        assert t.lookup("MYGENE") == pytest.approx(7.5)
        assert t.lookup("OTHER") == pytest.approx(1.0)

    def test_comments_and_blank_lines_are_skipped(self):
        assert parse_expression("# from GTEx\n\nACTB\t5\n").lookup("ACTB") == 5

    @pytest.mark.parametrize("text", ["", "gene\tcount\nA\t1\n", "tpm\n1\n",
                                      "gene\ttpm\nA\tNA\n"])
    def test_unusable_tables_are_refused(self, text):
        with pytest.raises(ValueError):
            parse_expression(text)

    def test_load_and_resolve(self, tmp_path, monkeypatch):
        f = tmp_path / "expr.tsv"
        f.write_text("ACTB\t5\n")
        assert load_expression(f).lookup("ACTB") == 5
        monkeypatch.delenv(ENV_EXPRESSION, raising=False)
        assert resolve_expression() is None
        monkeypatch.setenv(ENV_EXPRESSION, str(f))
        assert resolve_expression().source == str(f)


class TestWeights:
    def test_the_saturating_curve(self):
        t = ExpressionTable("t", {"lo": 0.1, "mid": 10.0, "hi": 500.0, "zero": 0.0})
        assert t.weight("lo") == pytest.approx(0.0099, abs=1e-4)
        assert t.weight("mid") == pytest.approx(0.5)
        assert t.weight("hi") == pytest.approx(0.98, abs=0.001)
        assert t.weight("zero") == 0.0

    def test_an_unlisted_gene_keeps_its_full_penalty(self):
        t = ExpressionTable("t", {"lo": 0.1})
        assert t.lookup("ABSENT") is None
        assert t.weight("ABSENT") == 1.0
        assert not t.below_floor("ABSENT"), "absent is never 'low'"


# --- Report-level behaviour on hand-built sites ----------------------------

def _hit(query, start, gene):
    return Hit(query=query, chrom=f"tx|{gene}", start=start, end=start + 24, strand="+",
               mismatches=0, gaps=0, aligned=25, query_len=25, q_start=1, q_end=25,
               gene=gene)


def _site(gene):
    return PairSite(chrom=f"tx|{gene}", left=_hit("left", 100, gene),
                    right=_hit("right", 126, gene), gap_nt=1, same_strand=True,
                    correct_order=True, survives_wash=True, min_margin_c=10.0,
                    gene=gene)


def _report(genes, expression=None, gene_aware=True):
    sites = [_site(g) for g in genes]
    return SpecificityReport(status=SCREENED, left_seq=LEFT, right_seq=RIGHT,
                             gene_aware=gene_aware, pair_sites=sites,
                             left_hits=[s.left for s in sites],
                             right_hits=[s.right for s in sites],
                             expression=expression)


TABLE = ExpressionTable("cells.tsv", {"lowgene": 0.2, "highgene": 500.0})


class TestReportWeighting:
    def test_no_table_is_not_evaluated_and_changes_nothing(self):
        rep = _report(["LOWGENE"])
        assert not rep.expression_evaluated
        assert rep.expression_reason == "no expression table supplied"
        assert rep.verdict == "fail" and rep.score == 0.0

    def test_a_low_expression_offtarget_softens_fail_to_review_never_pass(self):
        rep = _report(["LOWGENE"], TABLE)
        assert rep.expression_evaluated
        assert rep.verdict == "review"
        assert rep.score > 0.95
        assert "softened to review" in rep.summary()
        assert "LOWGENE 0.2 TPM" in rep.summary()

    def test_a_highly_expressed_offtarget_still_fails(self):
        rep = _report(["HIGHGENE"], TABLE)
        assert rep.verdict == "fail" and rep.score < 0.1

    def test_an_unlisted_offtarget_still_fails(self):
        rep = _report(["NOTINTABLE"], TABLE)
        assert rep.verdict == "fail" and rep.score == 0.0
        assert "NOTINTABLE not in table" in rep.summary()

    def test_one_high_gene_is_enough_to_fail(self):
        assert _report(["LOWGENE", "HIGHGENE"], TABLE).verdict == "fail"

    def test_a_genome_screen_says_why_it_was_not_weighted(self):
        rep = _report([""], TABLE, gene_aware=False)
        assert not rep.expression_evaluated
        assert "no gene names" in rep.expression_reason
        assert "NOT EVALUATED" in rep.summary()
        assert rep.verdict == "fail"

    def test_to_dict_carries_the_weighting(self):
        d = _report(["LOWGENE"], TABLE).to_dict()
        assert d["expression_evaluated"] and d["expression_source"] == "cells.tsv"
        assert d["offtarget_expression"] == [("LOWGENE", 0.2)]
        assert "UNCALIBRATED" in d["expression_caveat"]


# --- End to end against a real aligner -------------------------------------

HAVE_BOWTIE = bool(shutil.which("bowtie") and shutil.which("bowtie-build"))


def _gencode(gene, n):
    return f"ENST{n:011d}.1|ENSG{n:011d}.1|-|-|{gene}-20{n}|{gene}|1500|protein_coding|"


def _tx(payload, offset):
    rng = random.Random(offset)
    body = "".join(rng.choice("ACGT") for _ in range(1200))
    return body[:offset] + payload + body[offset + len(payload):]


@pytest.mark.skipif(not HAVE_BOWTIE, reason="no bowtie v1")
class TestTranscriptomeScreenWithExpression:
    @pytest.fixture
    def backend(self, tmp_path):
        footprint = LEFT + "T" + RIGHT
        fa = tmp_path / "tx.fa"
        fa.write_text(f">{_gencode('MYGENE', 1)}\n{_tx(footprint, 200)}\n"
                      f">{_gencode('BADGENE', 2)}\n{_tx(footprint, 500)}\n")
        idx = str(tmp_path / "tx")
        subprocess.run(["bowtie-build", "-q", str(fa), idx], capture_output=True, check=True)
        return Backend(tool="bowtie", database=idx, executable=shutil.which("bowtie"))

    def test_barely_expressed_offtarget_is_review(self, backend):
        rep = screen_pair(LEFT, RIGHT, backend=backend, target_gene="MYGENE",
                          expression=ExpressionTable("t", {"badgene": 0.05}))
        assert rep.offtarget_genes == ["BADGENE"]
        assert rep.verdict == "review"

    def test_abundant_offtarget_is_fail(self, backend):
        rep = screen_pair(LEFT, RIGHT, backend=backend, target_gene="MYGENE",
                          expression=ExpressionTable("t", {"badgene": 300.0}))
        assert rep.verdict == "fail"
