"""Local, pair-aware genome-wide specificity screening.

What makes this different from every general probe-design tool: a piFISH primary
pair only produces signal when **both** halves of the split probe bind adjacent
target. So a genome hit for one half is not an off-target -- it is a hit that
cannot make a spot by itself. The question worth asking is:

    is there anywhere else in the genome where BOTH halves land close together,
    and would those duplexes still be bound after the washes?

This module screens the two 25-nt target footprints (`left25` / `right25`, as
they read on the target's plus strand -- see `design`), collects every hit for
each half within a mismatch budget, and then joins them: a site is reported only
when both halves hit the same chromosome within a window. A hit for one half
alone lowers the score; it is not a rejection.

Two local backends, both shelled out to with explicit arguments:

* ``blastn -task blastn-short``, with ``-dust no -soft_masking false`` -- low
  complexity filtering destroys 25-nt queries, and a masked query silently
  returns nothing.
* ``bowtie`` (v1) ``-v 3 -a``, which is *exhaustive* end-to-end within 3
  mismatches. When a bowtie index is configured this is the better choice: it
  cannot miss a site inside its mismatch budget, whereas any seed-and-extend
  heuristic can.

Measured recovery of planted true sites, 40 x 25-nt queries per cell, from
``scripts/benchmark_specificity.py`` (rerun it; do not trust this table blind):

    backend                                 0mm    1mm    2mm    3mm    4mm   kept
    blastn-short -dust no -word_size 5    40/40  40/40  40/40  40/40  18/40   1316
    blastn-short -dust no -word_size 7    40/40  40/40  40/40  39/40  16/40    765
    blastn-short -dust no -word_size 11   40/40  40/40  27/40  14/40   5/40    151
    bowtie v1 -v 3 -a                     40/40  40/40  40/40  40/40   0/40    160
    bowtie2 --very-sensitive-local        40/40   0/40   0/40   0/40   0/40     40

Three things set the defaults here. ``bowtie -v 3 -a`` is perfect to the edge of
its budget and returns only the planted sites -- 160 kept alignments, no
background -- so it is preferred whenever an index exists. blastn-short's own
default ``-word_size 7`` already loses a site at 3 mismatches, so the default
here is 5; that costs runtime and ~8x more alignments to sift, which is the
price of not coming back falsely clean. And ``bowtie2 --very-sensitive-local``,
the obvious first choice, is unusable at this query length: a 25-nt query with
one mismatch often has no seed left, which is why it is kept in the benchmark as
a baseline rather than offered as a backend.

Filtering is on alignment geometry (aligned length, substitutions, gaps), never
on E-value. E-value scales with database size, so an E-value cutoff tuned on one
database silently changes meaning on another; with short queries it also throws
away true sites. The BLAST call therefore passes a deliberately permissive
``-evalue`` and does the real filtering here.

HOW CLOSE IS "TOGETHER" -- corrected. The pairing window that decides whether
two off-target half-hits could actually make a spot was originally 10 nt, from
the 1-nt design gap plus slack. The lab's own hand-made antisense fusion pairs
then turned out to anchor their arms 50 nt apart and still give signal, because
the arms are flexible single-stranded tethers rather than rigid rods. So the
threshold is now 50 nt, and it is an observed lower bound rather than a measured
limit -- see `DEFAULT_SIGNAL_GAP_NT`. The correction only ever loosens what the
screen is willing to call dangerous.

DEGRADED MODE IS A FIRST-CLASS RESULT. With no database configured the report
comes back ``NOT_SCREENED`` with a reason, and every consumer must render that
honestly. An earlier version of the order workbook asserted a specificity result
unconditionally, which put an unverified "repeat-free, single-copy" claim onto a
real order sheet. `SpecificityReport.screened` is the only thing that licenses
such a claim.

Configuration (first that resolves wins):

1. arguments to `resolve_backend()` / `screen_pair()`
2. ``$MYPIBYPY_BOWTIE_INDEX`` -- basename of a bowtie v1 index
3. ``$MYPIBYPY_BLAST_DB`` -- basename of a ``makeblastdb`` nucleotide database
"""
from __future__ import annotations

import bisect
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import thermo
from .expression import EXPRESSION_IS_UNCALIBRATED, ExpressionTable
from .sequence import GenomicInterval

#: Report statuses. `SCREENED` is the only one that licenses a specificity claim.
SCREENED = "SCREENED"
NOT_SCREENED = "NOT_SCREENED"
ERROR = "ERROR"

ENV_BOWTIE_INDEX = "MYPIBYPY_BOWTIE_INDEX"
ENV_BLAST_DB = "MYPIBYPY_BLAST_DB"

#: Mismatch budget per half. bowtie v1 is exhaustive up to 3; beyond that it is
#: not supported by the aligner at all, which is why 3 is the ceiling here too.
DEFAULT_MAX_MISMATCHES = 3

#: Two halves closer than this (bases between their footprints) are treated as
#: one candidate site. 200 nt is far wider than the 1-nt design gap on purpose:
#: a pair landing even loosely together is worth a human look.
DEFAULT_WINDOW_NT = 200

#: Gap at or below which an off-target pair is treated as able to reconstitute
#: the P2 landing site -- i.e. able to make a real spot, not merely to sit nearby.
#:
#: This was 10 nt, reasoning from the 1-nt design gap plus slack for
#: indel-containing alignments. That was too tight, and too tight in the
#: dangerous direction: it called genuinely signal-capable off-target pairs
#: merely "co-localized". The lab's hand-made antisense fusion pairs anchor
#: their two arms 50 nt apart and still give signal at the bench, because the
#: arms are flexible single-stranded tethers (22 nt is ~14 nm of contour against
#: the ~17 nm spanned by 50 bp of duplex) and loop back to present P2 together.
#: If a *designed* pair bridges 50 nt, an accidental one can too.
#:
#: 50 nt is therefore an observed lower bound on what works, not a measured
#: limit -- nobody has found the separation at which signal actually stops. Raise
#: it if you want more margin; do not lower it without bench evidence.
DEFAULT_SIGNAL_GAP_NT = 50

#: Previous name and value, kept so existing callers do not silently change
#: behaviour on import. Prefer `DEFAULT_SIGNAL_GAP_NT`.
DEFAULT_JUXTAPOSED_GAP_NT = DEFAULT_SIGNAL_GAP_NT

#: Shortest aligned block worth carrying forward. NOT an arbitrary round
#: number: `thermo` puts a perfect 14-bp duplex at roughly -1 C margin against
#: the most stringent wash at ~50% GC, and a 16-bp one at about +2 C. So below
#: ~15 bp a duplex comes apart regardless of how many mismatches it has, and a
#: shorter alignment cannot become an off-target. This is only a tractability
#: pre-filter -- BLAST emits tens of thousands of 7-12 nt scraps per query that
#: would otherwise go into an O(n^2) pairing step. The real decision is still
#: `thermo`'s, on the aligned sub-duplex. Raise it for a GC-rich target, where a
#: short duplex holds better than this default assumes; the count of hits it
#: removed is reported either way.
DEFAULT_MIN_ALIGNED_NT = 15

#: Score deductions. Starting points calibrated against nothing but judgement --
#: there are three designs in hand, far too few to fit these. Treat the hit
#: lists as the result and the score as a sort key.
PENALTY_SURVIVING_SIGNAL_CAPABLE = 1.0
PENALTY_SURVIVING_COLOCALIZED = 0.5
PENALTY_NONSURVIVING_PAIR = 0.1
PENALTY_SURVIVING_SINGLE = 0.02
MAX_SINGLE_PENALTY = 0.3


class SpecificityNotConfigured(RuntimeError):
    """No local database/index is available to screen against."""


# --- Reference names and genes ---------------------------------------------
#
# Screening a transcriptome is not screening a genome with longer names. A gene
# appears once in a genome and ~8.5 times in GENCODE, so a probe against its own
# target hits eight entries that are all the intended gene. Read as reference
# sequences they look like eight off-target loci and the probe gets rejected.
# The gene has to be recoverable from the reference name for that to be sorted
# out, and for plain chromosome FASTAs it simply is not -- so this degrades to
# "no gene information", and a genome screen behaves exactly as before.

#: 1-indexed field holding the gene symbol in a GENCODE FASTA header:
#: ENST|ENSG|OTTHUMG|OTTHUMT|transcript-name|GENE|length|biotype|
GENCODE_GENE_SYMBOL_FIELD = 6
#: 1-indexed field holding the stable gene ID, used when the symbol is absent.
GENCODE_GENE_ID_FIELD = 2
#: Minimum pipe-delimited fields before a name is treated as GENCODE-style.
GENCODE_MIN_FIELDS = 6


def gencode_gene(ref: str) -> str | None:
    """Gene for a pipe-delimited GENCODE-style reference name, else None.

    bowtie's SAM RNAME is the header up to the first whitespace, and GENCODE
    headers contain no whitespace, so the whole pipe-delimited string arrives
    intact and the gene is simply a field in it.

    Prefers the symbol and falls back to the stable gene ID, because a few
    entries carry "-" for the symbol. Returns None rather than guessing when the
    name is not of this shape.
    """
    if "|" not in ref:
        return None
    fields = ref.split("|")
    if len(fields) < GENCODE_MIN_FIELDS:
        return None
    symbol = fields[GENCODE_GENE_SYMBOL_FIELD - 1].strip()
    if symbol and symbol != "-":
        return symbol
    gene_id = fields[GENCODE_GENE_ID_FIELD - 1].strip()
    return gene_id or None


def no_gene(ref: str) -> None:
    """Gene resolver for references that carry none -- a plain chromosome FASTA."""
    return None


def detect_gene_parser(refs) -> "callable":
    """Pick a gene resolver by looking at the reference names that came back.

    Deliberately evidence-based rather than configured: the caller should not
    have to declare what kind of database they pointed at, and getting that
    declaration wrong is worse than detecting it. If any name parses as
    GENCODE-style, the GENCODE parser is used; names that do not parse yield
    None individually and are reported as ungrouped.
    """
    for ref in refs:
        if gencode_gene(ref) is not None:
            return gencode_gene
    return no_gene


# --- Hits ------------------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    """One alignment of one half-probe, in plus-strand genomic coordinates."""

    query: str          # "left" or "right"
    chrom: str
    start: int          # 1-based inclusive
    end: int            # 1-based inclusive
    strand: str         # "+" or "-"
    mismatches: int     # substitutions inside the aligned block
    gaps: int
    aligned: int        # query bases included in the alignment
    query_len: int
    q_start: int = 1    # 1-based inclusive, on the query
    q_end: int = 0      # 0 means "to the end of the query"
    gene: str = ""      # "" when the reference carries no gene information
    #: The reference over the aligned block, in query orientation -- what the
    #: probe actually meets there. "" when the aligner did not report it or the
    #: alignment is gapped; thermodynamics then falls back to the mismatch count.
    ref_seq: str = ""

    @property
    def unaligned(self) -> int:
        """Query bases the aligner left out of the alignment.

        BLAST aligns locally and will happily trim a mismatched end, so a 25-nt
        query can come back as a clean 20-nt hit. Those five bases did not
        vanish -- they are five unpaired positions, and the duplex that forms is
        20 bp, not 25. `duplex_seq` is what to hand to `thermo`; this is how much
        got left behind.
        """
        return max(0, self.query_len - self.aligned)

    @property
    def effective_mismatches(self) -> int:
        """Severity on one scale across backends: substitutions, gaps, trimming.

        Used for *ranking and reporting*. It is deliberately not the admission
        filter -- dropping a trimmed alignment would turn a real site into a
        clean report, which is the one direction a specificity screen must never
        fail in. See `screen_pair`.
        """
        return self.mismatches + self.gaps + self.unaligned

    def duplex_seq(self, query_seq: str) -> str:
        """The part of the query that actually paired, for thermodynamics."""
        end = self.q_end or self.query_len
        return query_seq[max(0, self.q_start - 1) : end]

    def site_seq(self, query_seq: str) -> str | None:
        """The off-target sequence opposite `duplex_seq`, or None if not known."""
        if self.ref_seq and len(self.ref_seq) == len(self.duplex_seq(query_seq)):
            return self.ref_seq
        return None

    def mismatch_offsets(self, query_seq: str) -> tuple[int, ...]:
        """0-based positions in the duplex where the site differs from the query.

        Empty when the site sequence is unknown -- check `site_seq` first if the
        difference between "no mismatches" and "not known" matters.
        """
        site = self.site_seq(query_seq)
        if site is None:
            return ()
        duplex = self.duplex_seq(query_seq).upper()
        return tuple(i for i, (a, b) in enumerate(zip(duplex, site)) if a != b)

    @property
    def interval(self) -> tuple[str, int, int]:
        return (self.chrom, self.start, self.end)

    def overlaps(self, iv: GenomicInterval) -> bool:
        return (self.chrom == iv.chrom and self.start <= iv.end and iv.start <= self.end)

    @property
    def locus_label(self) -> str:
        """Gene when the reference has one, otherwise the reference name.

        A GENCODE reference name is ~80 characters of pipe-delimited metadata;
        printing it in a report is useless to a reader.
        """
        return self.gene or self.chrom

    def __str__(self) -> str:
        trim = f"+{self.unaligned}nt trimmed" if self.unaligned else "full-length"
        return (f"{self.query} {self.locus_label}:{self.start:,}-{self.end:,}"
                f"({self.strand}) {self.mismatches}mm/{self.gaps}gap, {trim}")


@dataclass(frozen=True)
class PairSite:
    """A place where both halves hit close together -- the only thing that signals."""

    chrom: str
    left: Hit
    right: Hit
    gap_nt: int             # bases between the two footprints (design value: 1)
    same_strand: bool
    correct_order: bool     # halves in the order the target geometry requires
    survives_wash: bool     # both duplexes predicted still bound after every step
    min_margin_c: float     # worst per-step Tm margin across both halves
    signal_gap_nt: int = DEFAULT_SIGNAL_GAP_NT
    gene: str = ""      # "" when the reference carries no gene information
    #: Equilibrium occupancy of each half at its limiting step -- see
    #: `thermo.fraction_bound`. NaN when a half was priced by mismatch count,
    #: which gives a Tm but no dH/dS to turn into an occupancy.
    left_fraction: float = float("nan")
    right_fraction: float = float("nan")
    #: Which thermodynamic model priced the two halves, e.g. "nearest-neighbour
    #: mismatch" or "flat per-mismatch penalty". "" for sites built by hand.
    thermo_model: str = ""

    @property
    def locus_label(self) -> str:
        return self.gene or self.chrom

    @property
    def fraction_bound(self) -> float:
        """Both halves must be bound to make a spot: their product, NaN if unknown.

        Treats the halves as independent. That ignores the cooperativity P2
        adds once it bridges them, so it understates a site that has already
        nucleated -- it is a ranking number, comparable with the on-target value
        in `SpecificityReport.on_target_fraction`, not an absolute prediction.
        """
        return self.left_fraction * self.right_fraction

    @property
    def signal_capable(self) -> bool:
        """Close enough, and oriented right, to reconstitute the P2 site.

        "Close enough" is generous on purpose -- the arms are flexible tethers
        and bridge tens of nt, see `DEFAULT_SIGNAL_GAP_NT`. A site that is not
        signal-capable is still reported; it just cannot make a spot.
        """
        return (self.same_strand and self.correct_order
                and 0 <= self.gap_nt <= self.signal_gap_nt)

    @property
    def juxtaposed(self) -> bool:
        """Deprecated alias for `signal_capable`, which says what it means."""
        return self.signal_capable

    @property
    def total_mismatches(self) -> int:
        return self.left.effective_mismatches + self.right.effective_mismatches

    def __str__(self) -> str:
        kind = "SIGNAL-CAPABLE" if self.signal_capable else "co-localized"
        wash = "survives wash" if self.survives_wash else "melts in wash"
        frac = self.fraction_bound
        bound = "" if frac != frac else f", P(bound) {frac:.2g}"
        return (f"{kind} {self.locus_label}:{min(self.left.start, self.right.start):,} "
                f"gap {self.gap_nt} nt, {self.total_mismatches} mm total, "
                f"{wash} (worst margin {self.min_margin_c:+.1f} C{bound})")


# --- Backends --------------------------------------------------------------

def _write_fasta(path: Path, queries: dict[str, str]) -> None:
    path.write_text("".join(f">{k}\n{v}\n" for k, v in queries.items()))


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@dataclass(frozen=True)
class Backend:
    """A local aligner plus the database it searches."""

    tool: str            # "blastn" | "bowtie"
    database: str        # db/index basename
    executable: str
    max_mismatches: int = DEFAULT_MAX_MISMATCHES
    word_size: int = 5   # blastn only; 5, not blastn-short's own 7 -- measured
    threads: int = 1

    def describe(self) -> str:
        if self.tool == "bowtie":
            return (f"bowtie v1 -v {self.max_mismatches} -a (exhaustive within "
                    f"{self.max_mismatches} mismatches) against {self.database}")
        return (f"blastn -task blastn-short -dust no -word_size {self.word_size} "
                f"against {self.database}")

    def search(self, queries: dict[str, str]) -> list[Hit]:
        if self.tool == "blastn":
            return self._blastn(queries)
        if self.tool == "bowtie":
            return self._bowtie(queries)
        raise ValueError(f"unknown backend tool {self.tool!r}")

    # -- blastn ------------------------------------------------------------
    #: qseq/sseq come last so the 13 columns before them keep their indices.
    _FMT = ("6 qseqid sseqid pident length mismatch gapopen qstart qend "
            "sstart send evalue bitscore qlen qseq sseq")

    def _blastn(self, queries: dict[str, str]) -> list[Hit]:
        with tempfile.TemporaryDirectory() as tmp:
            fa = Path(tmp) / "query.fa"
            _write_fasta(fa, queries)
            cmd = [
                self.executable,
                "-task", "blastn-short",
                "-db", self.database,
                "-query", str(fa),
                "-outfmt", self._FMT,
                # Short queries are largely low-complexity by BLAST's reckoning;
                # with DUST on, 25-nt pads come back with no hits at all.
                "-dust", "no",
                "-soft_masking", "false",
                # Permissive on purpose -- see module docstring. Filtering is on
                # alignment geometry, which does not move with database size.
                "-evalue", "1000",
                "-word_size", str(self.word_size),
                "-penalty", "-2",        # -3 (the default) prices mismatched
                "-reward", "1",          # 25mers out of the result set
                "-max_target_seqs", "5000",
                "-num_threads", str(self.threads),
            ]
            proc = _run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"blastn failed ({proc.returncode}): {proc.stderr.strip()}")
        return self.parse_blast(proc.stdout, queries)

    @staticmethod
    def parse_blast(tabular: str, queries: dict[str, str]) -> list[Hit]:
        """Parse ``-outfmt 6`` with the column list in ``Backend._FMT``."""
        hits: list[Hit] = []
        for line in tabular.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 13:
                continue
            qid, sid = f[0], f[1]
            length, mism, gaps = int(f[3]), int(f[4]), int(f[5])
            qstart, qend = int(f[6]), int(f[7])
            sstart, send = int(f[8]), int(f[9])
            qlen = int(f[12])
            strand = "+" if sstart <= send else "-"
            start, end = (sstart, send) if strand == "+" else (send, sstart)
            if qstart > qend:            # minus-strand query coordinates
                qstart, qend = qend, qstart
            # sseq is already in query orientation for either strand. A gapped
            # alignment has no position-for-position site, so it is not kept.
            sseq = f[14].upper() if len(f) >= 15 else ""
            if "-" in sseq or "-" in (f[13] if len(f) >= 15 else ""):
                sseq = ""
            hits.append(Hit(query=qid, chrom=sid, start=start, end=end, strand=strand,
                            mismatches=mism, gaps=gaps, aligned=qend - qstart + 1,
                            query_len=len(queries.get(qid, "")) or qlen,
                            q_start=qstart, q_end=qend, ref_seq=sseq))
        return hits

    # -- bowtie v1 ---------------------------------------------------------
    def _bowtie(self, queries: dict[str, str]) -> list[Hit]:
        with tempfile.TemporaryDirectory() as tmp:
            fa = Path(tmp) / "query.fa"
            _write_fasta(fa, queries)
            cmd = [
                self.executable,
                "-f",                                # FASTA input
                "-v", str(self.max_mismatches),      # end-to-end, <= N mismatches
                "-a",                                # every alignment, not just one
                "--sam",
                "--quiet",
                "--threads", str(self.threads),
                "-x", self.database,
                str(fa),
            ]
            proc = _run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"bowtie failed ({proc.returncode}): {proc.stderr.strip()}")
        return self.parse_sam(proc.stdout, queries)

    @staticmethod
    def parse_sam(sam: str, queries: dict[str, str]) -> list[Hit]:
        """Parse bowtie's SAM output. Mismatches come from the NM tag."""
        hits: list[Hit] = []
        for line in sam.splitlines():
            if not line or line.startswith("@"):
                continue
            f = line.split("\t")
            if len(f) < 9:
                continue
            qid, flag, ref, pos = f[0], int(f[1]), f[2], int(f[3])
            if flag & 0x4 or ref == "*":
                continue
            seq_len = len(queries.get(qid, "")) or len(f[9])
            nm, md = 0, ""
            for tag in f[11:]:
                if tag.startswith("NM:i:"):
                    nm = int(tag[5:])
                elif tag.startswith("MD:Z:"):
                    md = tag[5:]
            reverse = bool(flag & 0x10)
            # bowtie v1 -v N is end-to-end: the whole read aligns, always.
            hits.append(Hit(query=qid, chrom=ref, start=pos, end=pos + seq_len - 1,
                            strand="-" if reverse else "+",
                            mismatches=nm, gaps=0, aligned=seq_len, query_len=seq_len,
                            q_start=1, q_end=seq_len,
                            ref_seq=_site_from_md(f[9], md, reverse)))
        return hits


_REVCOMP = str.maketrans("ACGTN", "TGCAN")


def _site_from_md(sam_seq: str, md: str, reverse: bool) -> str:
    """The reference under an ungapped SAM alignment, in query orientation.

    SAM stores SEQ on the reference's forward strand -- reverse-complemented for
    a minus-strand hit -- and MD lists the reference base at each substitution
    in the same frame. Rebuild the forward reference, then turn it back to the
    query's orientation. "" when MD is absent, carries a deletion (``^``), or
    does not account for every base, rather than a guess.
    """
    if not md or "^" in md:
        return ""
    out, i, num = [], 0, ""
    for ch in md + "$":
        if ch.isdigit():
            num += ch
            continue
        n = int(num or 0)
        out.append(sam_seq[i : i + n])
        i, num = i + n, ""
        if ch == "$":
            break
        out.append(ch)
        i += 1
    site = "".join(out).upper()
    if len(site) != len(sam_seq) or i != len(sam_seq):
        return ""
    return site[::-1].translate(_REVCOMP) if reverse else site


def resolve_backend(bowtie_index: str | None = None, blast_db: str | None = None,
                    prefer: str | None = None,
                    max_mismatches: int = DEFAULT_MAX_MISMATCHES,
                    threads: int = 1) -> Backend:
    """Find a usable local backend, or raise `SpecificityNotConfigured`.

    bowtie v1 is preferred when both are available because ``-v N -a`` is
    exhaustive: within its mismatch budget it cannot miss a site. Pass
    ``prefer="blastn"`` to override.
    """
    bowtie_index = bowtie_index or os.environ.get(ENV_BOWTIE_INDEX)
    blast_db = blast_db or os.environ.get(ENV_BLAST_DB)

    options = []
    if bowtie_index:
        options.append(("bowtie", bowtie_index))
    if blast_db:
        options.append(("blastn", blast_db))
    if prefer:
        options.sort(key=lambda o: o[0] != prefer)
    if not options:
        raise SpecificityNotConfigured(
            f"no local database configured: set ${ENV_BOWTIE_INDEX} (bowtie v1 "
            f"index basename) or ${ENV_BLAST_DB} (makeblastdb basename), or pass "
            f"one explicitly. No genome-wide screen has been run."
        )
    problems = []
    for tool, db in options:
        exe = shutil.which(tool)
        if not exe:
            problems.append(f"{tool} not on PATH")
            continue
        problem = _database_problem(tool, db)
        if problem:
            problems.append(problem)
            continue
        return Backend(tool=tool, database=db, executable=exe,
                       max_mismatches=max_mismatches, threads=threads)
    raise SpecificityNotConfigured("; ".join(problems))


def _database_problem(tool: str, db: str) -> str | None:
    """Why this database is unusable, or None if it looks fine.

    More than an existence check, because existence is not the failure that
    bites. A `bowtie-build` still running has already created all six .ebwt
    files, and bowtie will happily open the half-written index, exit 0 and
    return nothing -- which is indistinguishable from a clean screen. That is
    the one outcome this module exists to prevent, so the mirror index is
    checked for completeness too.

    The check is structural: the forward and mirror indices cover the same
    sequence, so .1/.rev.1 and .2/.rev.2 come out byte-identical in size. That
    is an observed invariant of bowtie v1 output rather than a documented
    guarantee -- verified against two complete indices and one mid-build -- so
    it is reported as "looks incomplete", not asserted as proof.
    """
    p = Path(db)
    if tool != "bowtie":
        if (list(p.parent.glob(p.name + ".n*"))
                or list(p.parent.glob(p.name + ".*.n*"))):
            return None
        return f"blastn database {db!r} not found"

    suffix = ".ebwtl" if list(p.parent.glob(p.name + "*.ebwtl")) else ".ebwt"
    required = [f"{p.name}.{n}{suffix}" for n in ("1", "2", "3", "4")]
    required += [f"{p.name}.rev.{n}{suffix}" for n in ("1", "2")]
    missing = [f for f in required if not (p.parent / f).is_file()]
    if len(missing) == len(required):
        return f"bowtie index {db!r} not found"
    if missing:
        return (f"bowtie index {db!r} is incomplete -- missing {missing}. "
                f"If bowtie-build is still running, wait for it to finish; a "
                f"partial index can return no hits and look like a clean screen")
    for fwd, rev in (("1", "rev.1"), ("2", "rev.2")):
        a = (p.parent / f"{p.name}.{fwd}{suffix}").stat().st_size
        b = (p.parent / f"{p.name}.{rev}{suffix}").stat().st_size
        if a != b:
            return (f"bowtie index {db!r} looks incomplete: {p.name}.{rev}{suffix} "
                    f"is {b:,} bytes against {a:,} for {p.name}.{fwd}{suffix}, so the "
                    f"mirror index is probably still being written. A partial index "
                    f"can return no hits and look like a clean screen -- refusing "
                    f"to screen against it")
    return None


# --- Pairing ---------------------------------------------------------------

@dataclass(frozen=True)
class WashOutcome:
    """What thermo says about the duplex one hit would form."""

    survives: bool
    min_margin_c: float
    fraction_bound: float   # at the limiting step; NaN under the count model
    model: str


def _wash_outcome(hit: Hit, query_seq: str,
                  steps: tuple[thermo.Conditions, ...] = thermo.ALL_STEPS,
                  penalty_c: float = thermo.DEFAULT_MISMATCH_PENALTY_C,
                  ) -> WashOutcome:
    """Survival, worst margin and occupancy for the duplex this hit would form.

    Priced from the off-target sequence when the aligner reported it, so a
    mismatch's position and identity count; by mismatch count otherwise.
    """
    duplex = hit.duplex_seq(query_seq)
    if len(duplex) < 2:
        return WashOutcome(False, float("-inf"), 0.0, thermo.MODEL_PERFECT)
    pr = thermo.profile(duplex, hit.mismatches + hit.gaps, steps, penalty_c,
                        site=hit.site_seq(query_seq))
    return WashOutcome(pr["survives"], pr["min_margin_c"], pr["min_fraction_bound"],
                       pr["model"])


def pair_hits(left_hits: list[Hit], right_hits: list[Hit], left_seq: str,
              right_seq: str, window_nt: int = DEFAULT_WINDOW_NT,
              signal_gap_nt: int = DEFAULT_SIGNAL_GAP_NT,
              steps: tuple[thermo.Conditions, ...] = thermo.ALL_STEPS,
              penalty_c: float = thermo.DEFAULT_MISMATCH_PENALTY_C) -> list[PairSite]:
    """Join per-half hits into candidate signal-producing sites.

    Both halves hybridize to the target's plus strand with a 1-nt gap, so on an
    off-target locus they must still land on the *same* strand and in the right
    order: left before right when both align to the plus strand, right before
    left when the locus is the reverse complement. Order and strand are recorded
    rather than used as a filter -- a co-localized pair in the wrong orientation
    makes no spot, but it is still worth seeing.
    """
    # Indexed by chromosome and sorted by start so each left hit only looks at
    # the right hits that could possibly be in range. A permissive blastn run on
    # a real genome returns thousands of hits per chromosome, and the naive
    # all-against-all is quadratic in that.
    by_chrom: dict[str, tuple[list[int], list[Hit]]] = {}
    for h in sorted(right_hits, key=lambda h: (h.chrom, h.start)):
        starts, hs = by_chrom.setdefault(h.chrom, ([], []))
        starts.append(h.start)
        hs.append(h)
    # Widest a right hit can be, so the search bound is exact rather than a guess.
    longest_right = max((h.end - h.start + 1 for h in right_hits), default=0)

    sites: list[PairSite] = []
    for lh in left_hits:
        starts, chrom_hits = by_chrom.get(lh.chrom, ([], []))
        lo = bisect.bisect_left(starts, lh.start - window_nt - longest_right - 2)
        hi = bisect.bisect_right(starts, lh.end + window_nt + 2)
        for rh in chrom_hits[lo:hi]:
            if lh.strand == rh.strand == "+":
                gap = rh.start - lh.end - 1
                correct_order = rh.start > lh.start
            elif lh.strand == rh.strand == "-":
                gap = lh.start - rh.end - 1
                correct_order = rh.start < lh.start
            else:
                # Opposite strands: no consistent geometry, but measure the
                # separation anyway so the site can be reported.
                gap = max(rh.start - lh.end - 1, lh.start - rh.end - 1)
                correct_order = False
            if abs(gap) > window_nt:
                continue
            # Evaluate the duplex that actually forms -- the aligned block,
            # carrying its own substitutions -- not the full 25-nt pad. A hit
            # BLAST trimmed to 20 nt makes a 20-bp duplex, and asking thermo
            # about the 25-mer would overstate it.
            lw = _wash_outcome(lh, left_seq, steps, penalty_c)
            rw = _wash_outcome(rh, right_seq, steps, penalty_c)
            sites.append(PairSite(
                chrom=lh.chrom, left=lh, right=rh, gap_nt=gap,
                same_strand=lh.strand == rh.strand, correct_order=correct_order,
                survives_wash=lw.survives and rw.survives,
                min_margin_c=min(lw.min_margin_c, rw.min_margin_c),
                signal_gap_nt=signal_gap_nt, gene=lh.gene or rh.gene,
                left_fraction=lw.fraction_bound, right_fraction=rw.fraction_bound,
                thermo_model=(lw.model if lw.model == rw.model
                              else f"{lw.model} / {rw.model}"),
            ))
    sites.sort(key=lambda s: (not s.signal_capable, not s.survives_wash,
                              s.total_mismatches))
    return sites


def collapse_sites_by_gene(sites: list[PairSite]) -> list[PairSite]:
    """One representative site per gene per distinct geometry.

    WHY THIS RUNS AFTER THE PAIR GEOMETRY AND NOT BEFORE. The obvious move is to
    collapse hits to genes first and pair the genes, which would also be ~8x less
    work. It is wrong: the gap between the two halves is what decides whether a
    site can signal, and that gap is a property of one transcript's coordinates.
    Two isoforms of a gene can place the same pair of pads 1 nt apart in one and
    800 nt apart in the other, because an intervening exon is spliced out of
    only one of them. Collapsing before the geometry would have to pick a
    representative coordinate and would silently invent or destroy sites.

    So the geometry runs per reference sequence, as it must, and the redundancy
    is removed here: isoforms of one gene that yield the same geometry collapse
    to one site, and a genuinely different geometry in another isoform survives
    as its own site. Sites with no gene information are never merged, since
    without a gene there is nothing to merge on.
    """
    out: list[PairSite] = []
    seen: set[tuple] = set()
    for site in sites:
        if not site.gene:
            out.append(site)
            continue
        key = (site.gene, site.gap_nt, site.same_strand, site.correct_order,
               site.left.mismatches, site.right.mismatches, site.survives_wash)
        if key in seen:
            continue
        seen.add(key)
        out.append(site)
    return out


# --- Report ----------------------------------------------------------------

@dataclass
class SpecificityReport:
    """The outcome of a screen -- including the outcome "no screen happened"."""

    status: str
    reason: str = ""
    backend: str = ""
    database: str = ""
    max_mismatches: int = DEFAULT_MAX_MISMATCHES
    window_nt: int = DEFAULT_WINDOW_NT
    min_aligned_nt: int = DEFAULT_MIN_ALIGNED_NT
    n_raw_hits: int = 0          # before any filtering
    n_dropped_short: int = 0     # removed by the min_aligned_nt pre-filter
    n_dropped_mismatch: int = 0  # over the mismatch budget
    left_seq: str = ""
    right_seq: str = ""
    target_gene: str = ""        # the intended gene, when screening a transcriptome
    gene_aware: bool = False     # did the reference carry parseable gene names?
    on_target_gene_hits: int = 0  # hits to the intended gene, excluded from pairing
    n_sites_before_collapse: int = 0
    left_hits: list[Hit] = field(default_factory=list)
    right_hits: list[Hit] = field(default_factory=list)
    pair_sites: list[PairSite] = field(default_factory=list)
    on_target: GenomicInterval | None = None
    #: Occupancy of the designed pair on its own target at the limiting step,
    #: on the same scale as `PairSite.fraction_bound`.
    on_target_fraction: float = float("nan")
    #: Per-gene abundance in the imaged cell type, when supplied. See `expression`.
    expression: ExpressionTable | None = None

    @property
    def screened(self) -> bool:
        """True only when a screen actually ran. Gate every claim on this."""
        return self.status == SCREENED

    @property
    def surviving_sites(self) -> list[PairSite]:
        return [s for s in self.pair_sites if s.survives_wash]

    @property
    def offtarget_genes(self) -> list[str]:
        """Distinct genes carrying at least one signal-capable, surviving site.

        This is the number to read on a transcriptome screen. "14 sites" across
        8 isoforms of 2 genes is 2 problems, not 14.
        """
        genes = {s.gene for s in self.pair_sites
                 if s.gene and s.signal_capable and s.survives_wash}
        return sorted(genes)

    @property
    def signal_capable_sites(self) -> list[PairSite]:
        """Off-target sites that could actually make a spot. The ones that matter."""
        return [s for s in self.pair_sites if s.signal_capable]

    @property
    def juxtaposed_sites(self) -> list[PairSite]:
        """Deprecated alias for `signal_capable_sites`."""
        return self.signal_capable_sites

    @property
    def single_half_hits(self) -> list[Hit]:
        """Hits that are not part of any reported pair site: no spot on their own."""
        paired = {id(s.left) for s in self.pair_sites} | {id(s.right) for s in self.pair_sites}
        return [h for h in (*self.left_hits, *self.right_hits) if id(h) not in paired]

    @property
    def surviving_single_half_hits(self) -> list[Hit]:
        """Lone-half hits whose duplex is predicted to hold through the washes."""
        seqs = {"left": self.left_seq, "right": self.right_seq}
        return [h for h in self.single_half_hits
                if seqs.get(h.query) and _wash_outcome(h, seqs[h.query]).survives]

    # -- expression --------------------------------------------------------

    @property
    def expression_evaluated(self) -> bool:
        """True only when off-target genes could be weighted by abundance."""
        return self.screened and self.expression is not None and self.gene_aware

    @property
    def expression_reason(self) -> str:
        """Why weighting did not happen, or "" when it did."""
        if self.expression_evaluated:
            return ""
        if not self.screened:
            return "no screen was run"
        if self.expression is None:
            return "no expression table supplied"
        return ("the reference carries no gene names (a genome screen), so "
                "off-target sites cannot be matched to expression")

    def _weight(self, site: PairSite) -> float:
        if not self.expression_evaluated or not site.gene:
            return 1.0
        return self.expression.weight(site.gene)

    @property
    def offtarget_expression(self) -> list[tuple[str, float | None]]:
        """(gene, TPM or None if absent from the table) per off-target gene."""
        if not self.expression_evaluated:
            return []
        return [(g, self.expression.lookup(g)) for g in self.offtarget_genes]

    @property
    def _failing_sites(self) -> list[PairSite]:
        return [s for s in self.pair_sites if s.survives_wash and s.signal_capable]

    @property
    def low_expression_only(self) -> bool:
        """Every failing site is on a gene the table puts below its TPM floor.

        A gene missing from the table is never "low" -- see `expression`.
        """
        failing = self._failing_sites
        return (self.expression_evaluated and bool(failing)
                and all(s.gene and self.expression.below_floor(s.gene) for s in failing))

    # -- occupancy ---------------------------------------------------------

    @property
    def worst_offtarget_fraction(self) -> float:
        """Highest occupancy among signal-capable sites; NaN if none was priced."""
        vals = [s.fraction_bound for s in self.pair_sites
                if s.signal_capable and not math.isnan(s.fraction_bound)]
        return max(vals) if vals else float("nan")

    @property
    def score(self) -> float:
        """1.0 = nothing found. Lower is worse. A sort key, not a probability."""
        if not self.screened:
            return float("nan")
        score = 1.0
        for s in self.pair_sites:
            # Scaled by the off-target gene's abundance when a table was given;
            # 1.0 otherwise, so an unweighted screen scores exactly as before.
            w = self._weight(s)
            if s.survives_wash and s.signal_capable:
                score -= PENALTY_SURVIVING_SIGNAL_CAPABLE * w
            elif s.survives_wash:
                score -= PENALTY_SURVIVING_COLOCALIZED * w
            else:
                score -= PENALTY_NONSURVIVING_PAIR * w
        # A lone half cannot recruit P2, so it cannot make a spot. It still costs
        # something -- it is probe sequestered elsewhere, and it is where a
        # second nearby hit would turn into a real site -- but only a little, and
        # only when the duplex would actually hold through the washes.
        singles = PENALTY_SURVIVING_SINGLE * len(self.surviving_single_half_hits)
        score -= min(singles, MAX_SINGLE_PENALTY)
        return max(0.0, score)

    @property
    def verdict(self) -> str:
        """``pass`` / ``review`` / ``fail`` / ``not screened``."""
        if not self.screened:
            return "not screened"
        # Low expression softens a fail to a review, never to a pass: the table
        # is an average over some cell population, not the cell being imaged.
        if self._failing_sites:
            return "review" if self.low_expression_only else "fail"
        if self.pair_sites or self.score < 0.9:
            return "review"
        return "pass"

    def summary(self) -> str:
        if self.status == NOT_SCREENED:
            return f"NOT SCREENED -- {self.reason}"
        if self.status == ERROR:
            return f"SCREEN FAILED -- {self.reason}"
        n_surv = len(self.surviving_sites)
        n_cap = len([s for s in self.pair_sites
                     if s.signal_capable and s.survives_wash])
        text = (
            f"{self.backend}: {len(self.left_hits)} left / {len(self.right_hits)} right "
            f"half-hits at <={self.max_mismatches} mismatches; "
            f"{len(self.pair_sites)} off-target site(s) with both halves within "
            f"{self.window_nt} nt, of which {n_surv} predicted to survive the wash "
            f"and {n_cap} close enough and oriented to reconstitute the P2 site"
        )
        if self.gene_aware:
            # Site count is per transcript and inflates with isoform number;
            # gene count is the thing a reader should act on.
            genes = self.offtarget_genes
            text += (
                f" across {len(genes)} off-target gene(s)"
                + (f" ({', '.join(genes[:6])}{'...' if len(genes) > 6 else ''})"
                   if genes else "")
            )
            if self.target_gene:
                text += (f"; {self.on_target_gene_hits} hit(s) to the intended gene "
                         f"{self.target_gene} were excluded as on-target")
            if self.n_sites_before_collapse > len(self.pair_sites):
                text += (f"; {self.n_sites_before_collapse} raw sites collapsed to "
                         f"{len(self.pair_sites)} by gene and geometry")
        if self.expression_evaluated:
            expr = self.offtarget_expression
            if expr:
                shown = ", ".join(f"{g} {'not in table' if t is None else f'{t:g} TPM'}"
                                  for g, t in expr[:6])
                text += f"; off-target expression: {shown}{'...' if len(expr) > 6 else ''}"
            if self.low_expression_only:
                text += (f"; every failing off-target gene is below "
                         f"{self.expression.min_tpm:g} TPM, so fail is softened to review")
            text += f" [weighted by {self.expression.describe()}]"
        elif self.expression is not None:
            text += f"; expression weighting NOT EVALUATED -- {self.expression_reason}"
        if not math.isnan(self.on_target_fraction):
            text += f"; on-target P(bound) {self.on_target_fraction:.2g}"
            worst = self.worst_offtarget_fraction
            if not math.isnan(worst):
                text += f" vs worst signal-capable off-target {worst:.2g}"
        text += (
            f" -- verdict {self.verdict} "
            f"[{self.n_raw_hits} raw alignments; {self.n_dropped_mismatch} over "
            f"budget, {self.n_dropped_short} shorter than {self.min_aligned_nt} nt]"
        )
        return text

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "screened": self.screened,
            "reason": self.reason,
            "backend": self.backend,
            "database": self.database,
            "max_mismatches": self.max_mismatches,
            "window_nt": self.window_nt,
            "n_left_hits": len(self.left_hits),
            "n_right_hits": len(self.right_hits),
            "n_pair_sites": len(self.pair_sites),
            "n_surviving_sites": len(self.surviving_sites),
            "n_single_half_hits": len(self.single_half_hits),
            "n_raw_hits": self.n_raw_hits,
            "n_dropped_short": self.n_dropped_short,
            "n_dropped_mismatch": self.n_dropped_mismatch,
            "min_aligned_nt": self.min_aligned_nt,
            "n_signal_capable_sites": len(self.signal_capable_sites),
            "gene_aware": self.gene_aware,
            "target_gene": self.target_gene,
            "offtarget_genes": self.offtarget_genes,
            "n_offtarget_genes": len(self.offtarget_genes),
            "on_target_gene_hits": self.on_target_gene_hits,
            "n_sites_before_collapse": self.n_sites_before_collapse,
            "score": self.score,
            "verdict": self.verdict,
            "on_target_fraction": self.on_target_fraction,
            "worst_offtarget_fraction": self.worst_offtarget_fraction,
            "expression_evaluated": self.expression_evaluated,
            "expression_reason": self.expression_reason,
            "expression_source": (self.expression.source if self.expression is not None
                                  else ""),
            "offtarget_expression": self.offtarget_expression,
            "expression_caveat": (EXPRESSION_IS_UNCALIBRATED
                                  if self.expression_evaluated else ""),
            "sites": [str(s) for s in self.pair_sites],
            "summary": self.summary(),
        }


def not_screened(reason: str) -> SpecificityReport:
    """The honest no-op result. Use this instead of omitting a report."""
    return SpecificityReport(status=NOT_SCREENED, reason=reason)


def screen_pair(left_seq: str, right_seq: str, backend: Backend | None = None,
                on_target: GenomicInterval | None = None,
                target_gene: str | None = None,
                window_nt: int = DEFAULT_WINDOW_NT,
                max_mismatches: int = DEFAULT_MAX_MISMATCHES,
                signal_gap_nt: int = DEFAULT_SIGNAL_GAP_NT,
                min_aligned_nt: int = DEFAULT_MIN_ALIGNED_NT,
                steps: tuple[thermo.Conditions, ...] = thermo.ALL_STEPS,
                penalty_c: float = thermo.DEFAULT_MISMATCH_PENALTY_C,
                expression: ExpressionTable | None = None,
                **backend_kwargs) -> SpecificityReport:
    """Screen one split-probe footprint. Never raises for a missing database.

    `left_seq` / `right_seq` are the two 25-nt footprints as they read on the
    target's plus strand (``Candidate.left.seq`` / ``Candidate.right.seq``), not
    the synthesized probes -- the arms are constant and belong to `crosshyb`.

    `on_target` is the intended footprint; hits overlapping it are dropped so the
    design does not flag itself. It is a coordinate interval, which is the right
    tool for a genome and useless for a transcriptome: the intended gene appears
    in every one of its isoforms at a different offset, and no single interval
    covers them.

    `target_gene` is the transcriptome equivalent -- hits to that gene are
    excluded from the pairing step and counted separately. Without it, a probe
    screened against GENCODE finds its own target in ~8 isoforms, pairs them up,
    and reports its own gene as eight off-target sites. Matching is on the gene
    field parsed out of the reference name, case-insensitively.

    `expression` weights each off-target gene by its abundance in the imaged
    cell type -- see `expression`. It needs gene names in the reference, so on a
    genome screen it is reported as not evaluated rather than silently ignored.
    """
    if backend is None:
        try:
            backend = resolve_backend(max_mismatches=max_mismatches, **backend_kwargs)
        except SpecificityNotConfigured as exc:
            return not_screened(str(exc))

    queries = {"left": left_seq.upper(), "right": right_seq.upper()}
    # An empty or stub query returns no alignments and would render as a clean
    # screen. Refuse it loudly instead -- found the hard way, by handing this a
    # footprint sliced past the end of a transcript.
    for name, q in queries.items():
        if len(q) < min_aligned_nt:
            return SpecificityReport(
                status=ERROR, backend=backend.describe(), database=backend.database,
                reason=(f"{name} query is {len(q)} nt, below the {min_aligned_nt} nt "
                        f"minimum; a short or empty query returns no hits and would "
                        f"look like a clean screen"))
        bad = set(q) - set("ACGTN")
        if bad:
            return SpecificityReport(
                status=ERROR, backend=backend.describe(), database=backend.database,
                reason=f"{name} query has non-DNA characters {sorted(bad)}")
    try:
        hits = backend.search(queries)
    except (RuntimeError, OSError) as exc:
        return SpecificityReport(status=ERROR, reason=str(exc),
                                 backend=backend.describe(), database=backend.database)

    # Admission is on substitutions and gaps only. Trimming is NOT grounds for
    # dropping a hit: a locally-aligned 20/25 nt hit is a real 20-bp duplex, and
    # discarding it would turn a genuine off-target into a clean report. The
    # short-alignment pre-filter exists for tractability and is reported.
    n_raw = len(hits)
    kept = [h for h in hits if h.mismatches + h.gaps <= max_mismatches]
    n_dropped_mismatch = n_raw - len(kept)
    long_enough = [h for h in kept if h.aligned >= min_aligned_nt]
    n_dropped_short = len(kept) - len(long_enough)
    if on_target is not None:
        long_enough = [h for h in long_enough if not h.overlaps(on_target)]

    # Attach the gene to every hit, if the reference names carry one. Detection
    # is from the names that actually came back, so a plain chromosome FASTA
    # yields no genes and everything below is a no-op -- a genome screen behaves
    # exactly as it did before gene awareness existed.
    gene_of = detect_gene_parser(h.chrom for h in long_enough)
    gene_aware = gene_of is not no_gene
    if gene_aware:
        long_enough = [replace(h, gene=gene_of(h.chrom) or "") for h in long_enough]

    # Exclude the intended gene before pairing. This is the step that stops a
    # good probe being rejected for hitting its own target's other isoforms.
    on_target_gene_hits = 0
    if target_gene and gene_aware:
        wanted = target_gene.strip().lower()
        on_target_gene_hits = sum(1 for h in long_enough if h.gene.lower() == wanted)
        long_enough = [h for h in long_enough if h.gene.lower() != wanted]

    left_hits = [h for h in long_enough if h.query == "left"]
    right_hits = [h for h in long_enough if h.query == "right"]

    sites = pair_hits(left_hits, right_hits, left_seq, right_seq,
                      window_nt=window_nt, signal_gap_nt=signal_gap_nt,
                      steps=steps, penalty_c=penalty_c)
    n_before_collapse = len(sites)
    if gene_aware:
        sites = collapse_sites_by_gene(sites)
    return SpecificityReport(
        status=SCREENED, backend=backend.describe(), database=backend.database,
        max_mismatches=max_mismatches, window_nt=window_nt,
        min_aligned_nt=min_aligned_nt, n_raw_hits=n_raw,
        n_dropped_short=n_dropped_short, n_dropped_mismatch=n_dropped_mismatch,
        left_seq=queries["left"], right_seq=queries["right"],
        left_hits=left_hits, right_hits=right_hits, pair_sites=sites,
        on_target=on_target, target_gene=target_gene or "", gene_aware=gene_aware,
        on_target_gene_hits=on_target_gene_hits,
        n_sites_before_collapse=n_before_collapse,
        on_target_fraction=(thermo.min_fraction_bound(queries["left"], steps=steps)
                            * thermo.min_fraction_bound(queries["right"], steps=steps)),
        expression=expression,
    )
