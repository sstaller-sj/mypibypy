"""End-to-end design orchestration: locus -> fetched sequence -> ranked windows
-> chosen split-probe pair -> specificity screen.

Every `DesignResult` carries a `SpecificityReport`, even when no database is
configured -- in that case it is an explicit NOT_SCREENED, never a missing
field. Consumers that make specificity claims (the order workbook) read
`report.screened` rather than assuming one happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .accessibility import (
    DEFAULT_ACCESSIBILITY_WEIGHT,
    DEFAULT_NUCLEATION_NT,
    AccessibilityResult,
    accessibility_penalty,
    not_evaluated,
    site_accessibility,
)
from .design import (
    DEFAULT_MIN_SPACING_NT,
    PADS_DIRECT,
    Candidate,
    ProbePair,
    arm_separation,
    arms_meet_at_junction,
    build_fusion_pairs,
    build_pair,
    check_partner_specificity,
    build_tiled_pairs,
    fusion_footprint,
    scan,
    scan_transcript,
    tile,
    tiled_names,
)
from .fold import PairFolding, pair_folding
from .fold import not_evaluated as folding_not_evaluated
from .loci import FusionJunction, Locus, RnaTarget
from .sequence import fetch_sequence, fetch_transcript, mask_repeats, rmsk_hits
from .specificity import Backend, SpecificityReport, not_screened, screen_pair
from .systems import AmplifierSystem, get_system


@dataclass
class DesignResult:
    locus: Locus
    system: AmplifierSystem
    region_seq: str
    candidates: list[Candidate]
    pair: ProbePair
    rmsk: list[dict]
    specificity: SpecificityReport
    #: Target-site accessibility. Only meaningful for RNA: a DNA target is
    #: denatured by the protocol, so the question does not arise there, and a
    #: genomic design carries an explicit "not applicable" rather than a number.
    accessibility: AccessibilityResult = field(
        default_factory=lambda: not_evaluated(
            "DNA target: the protocol denatures it, so target-site secondary "
            "structure does not gate binding"))
    #: Every pair ordered for this locus, in genomic/transcript order. A
    #: single-pair design leaves these empty and they fall back to the single
    #: `pair`/`specificity`/`accessibility` above, so nothing that predates
    #: tiling changes shape or output.
    pairs: list[ProbePair] = field(default_factory=list)
    pair_screens: list[SpecificityReport] = field(default_factory=list)
    pair_accessibility: list[AccessibilityResult] = field(default_factory=list)
    #: Self-folding of each ordered pair's P1L/P1R, parallel to `pairs`. Empty
    #: for results built by hand; `folding_at` then says it was not evaluated.
    pair_folding: list[PairFolding] = field(default_factory=list)
    #: The ranked windows an RNA design tiles from, in `design_rna`'s order:
    #: its accessibility shortlist first, then every other window. Kept, with
    #: the cost each was scored at, so a preview of another pair count or
    #: spacing (`diagnosis`) tiles exactly as the pipeline would. Empty for a
    #: genomic design, which tiles from every window of the region.
    tiling_pool: list[Candidate] = field(default_factory=list, repr=False)
    tiling_costs: dict[int, float] = field(default_factory=dict, repr=False)
    #: How many pairs greedy placement fit, when that fell short and the pairs
    #: were re-packed to fit more (`design.tile`); 0 when greedy filled it.
    repacked_from: int = 0
    #: Whether placement was solved, so a shortfall means no layout of the
    #: candidate windows fits more -- not just that greedy stopped. False for
    #: a single pair and for results built by hand.
    packing_solved: bool = False

    def __post_init__(self) -> None:
        if not self.pairs:
            self.pairs = [self.pair]
        if not self.pair_screens:
            self.pair_screens = [self.specificity]
        if not self.pair_accessibility:
            self.pair_accessibility = [self.accessibility]

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    def folding_at(self, i: int) -> PairFolding:
        """Probe self-folding for pair `i`, or an explicit not-evaluated."""
        if i < len(self.pair_folding):
            return self.pair_folding[i]
        return folding_not_evaluated("probe folding not checked for this pair")

    @property
    def folding(self) -> PairFolding:
        return self.folding_at(0)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def tiled(self) -> bool:
        return len(self.pairs) > 1

    def rows(self):
        """(pair, screen, accessibility) per ordered pair, padded if need be.

        The three lists are built together, but a caller constructing a result
        by hand may supply only `pairs`; missing entries degrade to an explicit
        not-screened / not-evaluated rather than silently reusing pair 1's.
        """
        for i, pair in enumerate(self.pairs):
            screen = (self.pair_screens[i] if i < len(self.pair_screens)
                      else not_screened("no screen recorded for this pair"))
            access = (self.pair_accessibility[i] if i < len(self.pair_accessibility)
                      else not_evaluated("no accessibility recorded for this pair"))
            yield pair, screen, access


def _folding(pairs: list[ProbePair], check: bool) -> list[PairFolding]:
    """Fold every ordered primary, or say per pair that it was not asked for."""
    if not check:
        return [folding_not_evaluated("probe-folding check not requested")
                for _ in pairs]
    return [pair_folding(p) for p in pairs]


def rank_locus_windows(seq: str, interval, rmsk: list[dict]) -> list[Candidate]:
    """Every repeat-free 51 nt window of a genomic region, best-ranked first.

    The ranking `design_locus` tiles from. Repeats are soft-masked so a pad is
    never placed in repetitive DNA.
    """
    masked = mask_repeats(seq, interval, rmsk)
    return scan(masked, interval, top=len(masked))


def design_locus(locus: Locus, top: int = 5, backend: Backend | None = None,
                 screen: bool = True, n_pairs: int = 1,
                 min_spacing: int = DEFAULT_MIN_SPACING_NT,
                 check_folding: bool = True,
                 **screen_kwargs) -> DesignResult:
    """Design one locus end to end.

    `screen=False` skips the genome screen outright; the result still carries a
    NOT_SCREENED report saying so, because "we chose not to screen" and "we
    screened and it was clean" must never render the same way.

    `n_pairs` tiles multiple non-overlapping pairs across the locus. One pair at
    ~100x amplification already works -- the lab's validated designs are all
    single-pair -- so tiling is headroom for marginal signal, not a fix for a
    defect. The published pi-FISH work used 35-50 probes per DNA locus per
    channel, so there is a lot of headroom available. EVERY selected pair is
    screened, not just the first: a second pair that lands on a repeat would
    otherwise be ordered unchecked.
    """
    system = get_system(locus.system)
    seq = fetch_sequence(locus.interval)
    rmsk = rmsk_hits(locus.interval)
    # Every window is ranked, so tiling has somewhere to choose from;
    # `candidates` is still trimmed to `top` for reporting.
    ranked = rank_locus_windows(seq, locus.interval, rmsk)
    if not ranked:
        raise ValueError(f"no valid repeat-free 51-nt window found in {locus.name}")
    cands = ranked[:top]

    selected, repacked_from = tile(ranked, n_pairs, min_spacing)
    pairs = build_tiled_pairs(locus.name, locus.gene, system, selected)

    screens = []
    for cand in selected:
        if screen:
            screens.append(screen_pair(cand.left.seq, cand.right.seq, backend=backend,
                                       on_target=cand.interval, **screen_kwargs))
        else:
            screens.append(
                not_screened("genome screen skipped by request (screen=False)"))
    return DesignResult(locus, system, seq, cands, pairs[0], rmsk, screens[0],
                        pairs=pairs, pair_screens=screens,
                        pair_folding=_folding(pairs, check_folding),
                        repacked_from=repacked_from, packing_solved=n_pairs > 1)


class FusionIsSinglePair(ValueError):
    """A junction is one position, so it cannot be tiled."""


def design_fusion(junction: FusionJunction, backend=None, screen: bool = True,
                  pad_assignment: str = PADS_DIRECT, n_pairs: int = 1,
                  check_folding: bool = True,
                  **screen_kwargs) -> list[DesignResult]:
    """Design a fusion junction: one sense result and one antisense result.

    The junction has no coordinates in the reference, so there is no
    RepeatMasker step -- `rmsk` is empty, and the workbook says 0 repeats
    because none were looked for.

    On what the genome screen means here: each pad is, by construction, a real
    25-mer from its own parent gene, so single-half hits are guaranteed and
    expected. They are not a defect. What matters is that no site anywhere has
    *both* halves together -- which is exactly the pair-aware question
    `specificity` asks, and exactly why a per-probe screen is useless for a
    junction. `check_partner_specificity` adds the complementary local check
    that neither pad is present in the *wrong* partner.

    `n_pairs` above 1 is refused rather than quietly ignored. A fusion junction
    is a single position: the design *is* the 1-nt gap sitting on the
    breakpoint, and there is nowhere else to put a pair that would still be
    junction-specific. A second pair would have to lie wholly inside one
    partner, where it reports that partner unfused -- precisely what a fusion
    probe exists to rule out. Tiling a junction is not a smaller version of
    tiling a locus; it is a different and wrong design. (A junction does still
    return two results, sense and antisense, which is a separate axis from
    tiling.)
    """
    if n_pairs != 1:
        raise FusionIsSinglePair(
            f"n_pairs={n_pairs} is not meaningful for a fusion junction: the gap "
            f"must sit on the breakpoint, so there is exactly one junction-"
            f"specific footprint. A second pair could only lie wholly inside one "
            f"partner gene, where it would report that gene unfused. To raise "
            f"signal here, tile the partner genes as ordinary loci instead -- "
            f"knowing those pairs are not junction-specific."
        )
    sense_system = get_system(junction.system)
    anti_system = get_system(junction.antisense_system)
    _, footprint = fusion_footprint(junction.sequence, junction.breakpoint,
                                    junction.gap_side)
    partner_problems = check_partner_specificity(
        footprint, junction.five_prime_seq, junction.three_prime_seq)

    pairs = build_fusion_pairs(
        junction.name, junction.sequence, junction.breakpoint,
        sense_system, anti_system, gap_side=junction.gap_side,
        pad_assignment=pad_assignment,
    )

    results: list[DesignResult] = []
    for pair in pairs:
        cand = pair.candidate
        strand = "antisense" if "antisense" in pair.p1l_name else "sense"
        flags = list(cand.flags)
        flags += [f"partner: {p}" for p in partner_problems]
        if not arms_meet_at_junction(pair, footprint) and not any(
                "loop back" in f for f in flags):
            sep = arm_separation(pair, footprint)
            flags.append(
                f"arms anchored {sep} nt apart rather than either side of the "
                f"breakpoint, so they must loop back to present P2 together. "
                f"The lab's hand-made pairs are built this way and do give "
                f"signal; relative efficiency has not been compared")
        cand.flags = flags
        locus = Locus(
            name=f"{junction.name}_{strand}",
            gene=f"{junction.five_prime_gene}::{junction.three_prime_gene}",
            description=(junction.description
                         or f"fusion junction, {strand} strand").strip(),
            interval=cand.interval, system=pair.system.name,
        )
        if screen:
            spec = screen_pair(cand.left.seq, cand.right.seq, backend=backend,
                               **screen_kwargs)
        else:
            spec = not_screened("genome screen skipped by request (screen=False)")
        results.append(DesignResult(locus, pair.system, junction.sequence,
                                    [cand], pair, rmsk=[], specificity=spec,
                                    pair_folding=_folding([pair], check_folding)))
    return results


#: Which reference an RNA design is screened against, and why.
#:
#: TRANSCRIPTOME ONLY, by default. Two reasons, and the second is the one that
#: decides it.
#:
#: 1. The transcriptome is the population of molecules actually present. An RNA
#:    probe competes with every other transcript in the cell, which is precisely
#:    what a transcriptome index represents and what a genome index does not: a
#:    gene's expressed isoforms are separate molecules at separate abundances,
#:    and a genome collapses them into one locus.
#: 2. A footprint may legitimately span an exon-exon junction -- often the point,
#:    since it is what makes a probe isoform-specific. Such a footprint does not
#:    map contiguously to the genome at all. Screened against a genome its two
#:    halves land in different exons, thousands of nt apart, so the real
#:    on-target site is invisible and the output is actively misleading.
#:
#: Genomic DNA is in the sample too, but a standard RNA-FISH protocol has no
#: denaturation step, so that DNA stays double-stranded and a single-stranded
#: probe cannot invade it. If your protocol *does* denature -- a combined
#: DNA/RNA experiment, for instance -- then the genome matters and you should
#: screen against both: pass `also_genome_backend`. Doing so is not the default
#: because for the usual protocol it adds a reference the probe never meets, and
#: junction-spanning designs would be flagged for a geometry that is correct.
RNA_SCREEN_POLICY = "transcriptome-only by default; see RNA_SCREEN_POLICY"


@dataclass
class RnaRanking:
    """How `design_rna` ranks a transcript's windows, kept for tiling and previews."""

    ranked: list[Candidate]               # shortlist first, then every other window
    costs: dict[int, float]               # tiling cost by offset
    access: dict[int, AccessibilityResult]
    shortlist: list[Candidate]            # the windows accessibility re-ranked


def rank_rna_windows(seq: str, target: RnaTarget, top: int = 5,
                     accessibility_top: int = 25, check_accessibility: bool = True,
                     accessibility_weight: float = DEFAULT_ACCESSIBILITY_WEIGHT,
                     nucleation_nt: int = DEFAULT_NUCLEATION_NT) -> RnaRanking:
    """Every window of a transcript, ranked as `design_rna` tiles from them."""
    # Every window is scored; accessibility is evaluated for a shortlist of the
    # best, wider than requested so it has candidates to choose between: the
    # most accessible window is often not the best-scoring one on GC and Tm
    # alone, and re-ranking a list of one achieves nothing.
    every = scan_transcript(seq, target.accession, top=len(seq),
                            scan_start=target.scan_start, scan_end=target.scan_end,
                            exon_boundaries=target.exon_boundaries)
    if not every:
        raise ValueError(f"no valid 51-nt window found in {target.accession}")
    cands = every[:max(top, accessibility_top)]

    # Accessibility is RNA-only and is applied HERE, not in `design._score`.
    # That keeps `_score` byte-comparable with every design made before this
    # existed -- adding a term there would retroactively reorder genomic
    # designs that have already been ordered. See accessibility.py on why any
    # threshold is uncalibrated; this is a ranking input, never a reject.
    access: dict[int, AccessibilityResult] = {}
    if check_accessibility:
        for cand in cands:
            access[cand.offset] = site_accessibility(
                seq, cand.offset, nucleation_nt=nucleation_nt)
        cands = sorted(
            cands,
            key=lambda c: c.penalty + accessibility_penalty(
                access[c.offset], weight=accessibility_weight),
        )
    # Tiled from every window, the shortlist first in its accessibility order
    # and the rest after it on GC/Tm alone -- so greedy takes exactly what it
    # took from the shortlist whenever the shortlist holds every pair, and only
    # reaches past it when it does not. (Tiling from the shortlist alone placed
    # 3 of 4 on a 420 nt sequence where 4 fit.) A window outside the shortlist
    # has no accessibility yet; scored at the worst the shortlist saw, it
    # cannot outrank an evaluated one by not having been evaluated.
    shortlist = cands
    ranked = shortlist + every[len(shortlist):]
    costs = {c.offset: c.penalty + (accessibility_penalty(
                 access[c.offset], weight=accessibility_weight)
                 if c.offset in access else 0.0)
             for c in shortlist}
    unknown = max((costs[c.offset] - c.penalty for c in shortlist), default=0.0)
    for c in every[len(shortlist):]:
        costs[c.offset] = c.penalty + unknown
    return RnaRanking(ranked, costs, access, shortlist)


def design_rna(target: RnaTarget, backend=None, screen: bool = True,
               top: int = 5, transcript_seq: str | None = None,
               also_genome_backend=None, check_accessibility: bool = True,
               accessibility_top: int = 25,
               accessibility_weight: float = DEFAULT_ACCESSIBILITY_WEIGHT,
               nucleation_nt: int = DEFAULT_NUCLEATION_NT,
               n_pairs: int = 1, min_spacing: int = DEFAULT_MIN_SPACING_NT,
               check_folding: bool = True,
               **screen_kwargs) -> DesignResult:
    """Design a split-probe pair against a transcript.

    `backend` should index a TRANSCRIPTOME -- see `RNA_SCREEN_POLICY` for why,
    and for when to also pass `also_genome_backend`.

    `n_pairs` tiles several non-overlapping pairs along the transcript. The
    published guidance for RNA is roughly 10-15 pairs per transcript at medium
    to low abundance and 1-5 for a highly abundant one, so the right count
    depends on the target's expression and is left to the caller. Each pair is
    screened and has its accessibility evaluated separately -- the whole point
    of tiling an mRNA is that structure varies along it, so a shared number
    would defeat the exercise.

    Target-site accessibility is evaluated for each candidate and folded into
    the ranking -- see `accessibility`. It never rejects a window, and when
    RNAplfold is absent every candidate scores 0 for it, so the ordering is
    exactly what it was before accessibility existed.

    The screen is given `target_gene`, so the probe's own gene is excluded
    instead of being reported as an off-target in each of its isoforms. Note the
    co-localisation window is in transcript coordinates here: two half-sites 50
    nt apart in a spliced mRNA may be far apart in the genome, and the spliced
    distance is the one that matters, because the spliced molecule is what the
    probe binds.
    """
    system = get_system(target.system)
    seq = transcript_seq if transcript_seq is not None else fetch_transcript(
        target.accession)
    ranking = rank_rna_windows(
        seq, target, top=top, accessibility_top=accessibility_top,
        check_accessibility=check_accessibility,
        accessibility_weight=accessibility_weight, nucleation_nt=nucleation_nt)
    ranked, costs, access = ranking.ranked, ranking.costs, ranking.access
    shortlist = ranking.shortlist
    cands = shortlist[:top]
    selected, repacked_from = tile(ranked, n_pairs, min_spacing,
                                   cost=lambda c: costs[c.offset])
    # Evaluated now for any pair taken from past the shortlist, so every
    # ordered pair still reports its accessibility.
    if check_accessibility:
        for cand in selected:
            if cand.offset not in access:
                access[cand.offset] = site_accessibility(
                    seq, cand.offset, nucleation_nt=nucleation_nt)
    pairs = build_tiled_pairs(target.name, target.gene, system, selected)
    locus = Locus(name=target.name, gene=target.gene,
                  description=(target.description
                               or f"RNA target on {target.accession}"),
                  interval=selected[0].interval, system=system.name)

    screens, accesses = [], []
    for cand in selected:
        if screen:
            spec = screen_pair(cand.left.seq, cand.right.seq, backend=backend,
                               target_gene=target.gene, **screen_kwargs)
            if also_genome_backend is not None:
                genome_kwargs = {k: v for k, v in screen_kwargs.items()
                                 if k != "expression"}
                genome = screen_pair(cand.left.seq, cand.right.seq,
                                     backend=also_genome_backend, **genome_kwargs)
                # Kept as a note rather than merged: the two screens answer
                # different questions against different coordinate spaces, and
                # averaging their verdicts would obscure both.
                spec.reason = (spec.reason + " | additional genome screen: "
                               + genome.summary()).strip(" |")
        else:
            spec = not_screened("genome screen skipped by request (screen=False)")
        screens.append(spec)
        # Already computed above for every ranked candidate when requested.
        accesses.append(access.get(
            cand.offset, not_evaluated("accessibility check not requested")))

    return DesignResult(locus, system, seq, cands, pairs[0], rmsk=[],
                        specificity=screens[0], accessibility=accesses[0],
                        pairs=pairs, pair_screens=screens,
                        pair_accessibility=accesses,
                        pair_folding=_folding(pairs, check_folding),
                        tiling_pool=ranked, tiling_costs=costs,
                        repacked_from=repacked_from, packing_solved=n_pairs > 1)
