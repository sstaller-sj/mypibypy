"""The shared troubleshooting layer: facts about a design, and previews of changes.

These are the answers the chat's model used to work out for itself, so they are
checked here, without a model: the right numbers, the right reason, and nothing
shaped like a sequence.
"""
import random
import re
from dataclasses import replace

import pytest

from mypibypy.agent import DesignParameters
from mypibypy.design import build_tiled_pairs, scan, tile
from mypibypy.diagnosis import (
    PREVIEW_CAVEAT,
    TOPICS,
    diagnose,
    placement_facts,
    placement_reason,
    preview_change,
    try_changes,
)
from mypibypy.loci import Locus
from mypibypy.pipeline import DesignResult
from mypibypy.sequence import GenomicInterval
from mypibypy.specificity import not_screened
from mypibypy.systems import systems

SEQUENCE_SHAPED = re.compile(r"[ACGTUacgtu]{12,}")
#: A registry key, which is what a target's `system` names.
KEY = sorted(systems())[0]


@pytest.fixture(scope="module")
def system():
    reg = systems()
    assert reg, "no amplifier systems configured"
    return list(reg.values())[0]


def make_result(system, n_pairs: int, spacing: int, length: int = 400,
                chrom: str = "chr1", seed: int = 3):
    """A result designed the way `design_locus` does: one ranking, `design.tile`."""
    rng = random.Random(seed)
    seq = "".join(rng.choice("ACGT") for _ in range(length))
    iv = GenomicInterval("hg38", chrom, 1000, 1000 + length - 1)
    ranked = scan(seq, iv, top=len(seq))
    selected, repacked_from = tile(ranked, n_pairs, spacing)
    pairs = build_tiled_pairs("t", "GENE", system, selected)
    screens = [not_screened("screen skipped in test") for _ in pairs]
    return DesignResult(Locus("t", "GENE", "test", iv, system.name), system, seq,
                        ranked[:5], pairs[0], [], screens[0], pairs=pairs,
                        pair_screens=screens, repacked_from=repacked_from,
                        packing_solved=n_pairs > 1)


def params_for(system, n_pairs, spacing, **kw):
    return DesignParameters(system=system.name, n_pairs=n_pairs,
                            min_spacing=spacing, **kw)


@pytest.fixture
def short(system):
    """Asked for more pairs than wide spacing lets fit."""
    params = params_for(system, 6, 40)
    result = make_result(system, params.n_pairs, params.min_spacing)
    assert result.n_pairs < params.n_pairs, "fixture should fall short"
    return result, params


# --- diagnose -------------------------------------------------------------------

def test_every_fact_has_a_known_topic(short):
    result, params = short
    assert {f.topic for f in diagnose([result], params)} <= set(TOPICS)


def test_placement_fact_is_the_placement_reason(short):
    result, params = short
    facts = [f.text for f in diagnose([result], params) if f.topic == "placement"]
    assert " ".join(facts) == placement_reason(result, params)
    assert facts[0] == f"Placed {result.n_pairs} of 6 requested."


def test_unscreened_pairs_are_said_with_their_reason(short):
    result, params = short
    screen = next(f.text for f in diagnose([result], params) if f.topic == "screen")
    assert "NOT screened" in screen and "screen skipped in test" in screen


def test_an_unenforced_window_is_never_a_rejection(short):
    result, params = short
    windows = next(f.text for f in diagnose([result], params) if f.topic == "windows")
    assert "NOTHING was rejected" in windows


def test_an_enforced_window_reports_passing_windows(system):
    params = params_for(system, 1, 3, enforce_window=True)
    result = make_result(system, 1, 3)
    windows = next(f.text for f in diagnose([result], params) if f.topic == "windows")
    assert "windows pass" in windows and "NOTHING" not in windows


# --- try_changes -----------------------------------------------------------------

def test_nothing_to_try_when_every_pair_was_placed(system):
    params = params_for(system, 2, 3)
    result = make_result(system, 2, 3)
    assert result.n_pairs == 2
    assert try_changes([result], params) == []


def test_a_fusion_has_nothing_to_tile(short):
    result, params = short
    assert try_changes([result, result], params) == []


def test_previews_offer_smaller_spacings_and_fewer_pairs(short):
    result, params = short
    previews = try_changes([result], params)
    spacings = [p.params.min_spacing for p in previews if p.change == "min_spacing"]
    assert spacings and all(s < params.min_spacing for s in spacings)
    assert 0 in spacings
    fewer = [p for p in previews if p.change == "n_pairs"]
    assert [(p.params.n_pairs, p.pairs_placed) for p in fewer] == \
        [(result.n_pairs, result.n_pairs)]


def test_a_spacing_preview_matches_designing_with_it(system, short):
    """The preview's count is what the pipeline's own tiling gives at that spacing."""
    result, params = short
    for p in try_changes([result], params):
        if p.change == "min_spacing":
            redesigned = make_result(system, p.params.n_pairs, p.params.min_spacing)
            assert p.pairs_placed == redesigned.n_pairs
            assert f"{p.pairs_placed} of {params.n_pairs} placed" in p.text


def test_a_transcript_without_its_shortlist_is_not_previewed(system):
    """An RNA design tiles from a shortlist; without it (a result built by hand)
    a preview from every window could be wrong -- one said "5 of 6" where
    designing placed 3 -- so none is given."""
    params = params_for(system, 6, 40)
    result = make_result(system, 6, 40, chrom="transcript")
    assert try_changes([result], params) == []
    assert preview_change([result], params, params_for(system, 6, 10)) is None


def _clustered(seed: int, length: int = 420) -> str:
    """Random sequence in blocks of varying GC, so good windows cluster.

    Uniform random sequence has good windows everywhere, and greedy then fills
    any request; real targets are uneven, which is when it falls short.
    """
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < length:
        gc = rng.choice((0.5, 0.75, 0.85))
        out += [rng.choice("GC") if rng.random() < gc else rng.choice("AT")
                for _ in range(rng.randint(30, 90))]
    return "".join(out[:length])


#: A synthetic 420 nt target that behaves like the one a user hit in the chat:
#: asked for 4 pairs at 50 nt spacing, greedy placed 3 though 4 fit end to end.
#: (Generated, not copied: target sequences are the lab's and stay out of the
#: repository.)
REPORTED = _clustered(1)


def design_reported(n_pairs, spacing, seq=REPORTED, **kw):
    from mypibypy.loci import RnaTarget
    from mypibypy.pipeline import design_rna

    return design_rna(RnaTarget("t", "GENE", "pasted-sequence", KEY), screen=False,
                      transcript_seq=seq, n_pairs=n_pairs, min_spacing=spacing,
                      check_folding=False, **kw)


def test_the_reported_shortfall_is_re_packed_and_said(system):
    """Greedy places 3 of 4 at 50 nt on the 420 nt target; 4 fit."""
    result = design_reported(4, 50)
    assert (result.n_pairs, result.repacked_from) == (4, 3)
    facts = placement_facts(result, params_for(system, 4, 50))
    assert facts[0] == "Placed 4 of 4 requested."
    assert "fit only 3, so the pairs were re-laid out" in facts[1]


def test_an_rna_preview_matches_designing_with_it(system):
    """Previewed from the pool and costs the pipeline kept, it is the pipeline's answer."""
    params = params_for(system, 6, 50)
    result = design_reported(6, 50)
    assert result.n_pairs < 6 and result.tiling_pool
    previews = try_changes([result], params)
    assert previews
    for p in previews:
        assert p.pairs_placed == design_reported(p.params.n_pairs,
                                                 p.params.min_spacing).n_pairs


#: The reported sequence with its middle 120 nt unusable: only the two arms
#: hold windows, so no layout fits more than one pair per arm.
GAPPED = REPORTED[:150] + "N" * 120 + REPORTED[270:]


def test_a_solved_shortfall_says_no_layout_fits_more(system):
    """Placement is solved, so a shortfall is a bound -- and the facts still say
    what each gap needs, which the user had to be told (105 nt free is not
    room for a pair between two others at 50 nt spacing; 151 is)."""
    result = design_reported(4, 50, seq=GAPPED, check_accessibility=False)
    assert result.packing_solved and result.n_pairs < 4
    reason = placement_reason(result, params_for(system, 4, 50))
    assert "needs 151 nt free (50 + 51 + 50)" in reason
    assert "nt free, needs 101" in reason
    assert f"No layout of the target's usable windows fits more than {result.n_pairs}" in reason
    assert "placed greedily" not in reason


def test_a_roomy_gap_with_no_window_is_said_as_such(system):
    """If a gap is big enough but nothing was placed there, the reason is that
    no candidate window lies in it -- not that the gap is too small."""
    params = params_for(system, 4, 40)
    result = make_result(system, 4, 40)
    # Keep only one pair: there is now room beside it that nothing fills. By
    # hand, so placement was not solved -- the case this wording is for.
    result.pairs = result.pairs[:1]
    result.repacked_from, result.packing_solved = 0, False
    reason = placement_reason(result, params)
    assert "has room, but no candidate window lies there" in reason
    assert "no gap the placed ones left is big enough" not in reason


def test_no_preview_unless_it_reproduces_the_run(system, short):
    """A result the preview cannot reproduce (here: designed at another
    spacing than `params` says) gets no preview rather than a wrong one."""
    result, params = short
    assert try_changes([result], replace(params, min_spacing=0)) == []


def test_previews_change_only_what_they_name(short):
    result, params = short
    for p in try_changes([result], params):
        changed = {k for k, v in p.params.to_dict().items()
                   if v != getattr(params, k)}
        assert changed == {p.change}


def test_nothing_shaped_like_a_sequence_comes_out(short):
    result, params = short
    texts = [f.text for f in diagnose([result], params)]
    texts += [p.text for p in try_changes([result], params)] + [PREVIEW_CAVEAT]
    for text in texts:
        assert not SEQUENCE_SHAPED.search(text), text
