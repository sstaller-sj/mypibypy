"""The chat's answers, written by the software, one per topic.

`agent.topics` lists the kinds of question the chat answers. The model picks
the topic; this module writes the answer from the design itself -- the same
objects the results table and the order workbook are drawn from -- so a number
in a reply is the number in the design, and a reason is one the pipeline
worked out (`mypibypy.diagnosis`), not one a model inferred.

That is also why none of these goes through `reply_check`: there is no model
text in them to check. The only reply that still does is topic "other", which
`answer` declines (returns None) so the caller falls back to the model's note.

Each answer carries a few follow-ups: messages the user could send next, shown
as buttons. They are ordinary chat messages and go through the same path as
typed ones, so a follow-up can ask for a design ("make it 4 pairs") as well as
another question.

No Streamlit and no model call here, so every template is tested directly
(`tests/test_answers.py`). No sequence either: an answer names pairs, numbers
and genes, never bases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ..design import FOOTPRINT
from ..diagnosis import (
    NOT_PREVIEWABLE,
    PREVIEW_CAVEAT,
    diagnose,
    preview_change,
    screening,
    try_changes,
)
from ..fold import FOLDING_IS_UNCALIBRATED
from .actions import BOUNDS, DesignParameters, clamp
from .context import SCREEN_UNSET, ScreenStatus, _details, compare_pairs, pairs_named
from .intent import UNCHANGED_INT
from .topics import NONE, OTHER, TOPIC_KEYS


@dataclass(frozen=True)
class Asked:
    """What a template may draw on for one turn.

    `run` is the last `session.Run` on the current target, or None when nothing
    has been designed on it. `n_pairs` and `min_spacing` are the numbers the
    model read out of a what-if question, at their sentinels otherwise.
    """

    message: str
    params: DesignParameters
    run: object | None = None
    screen: ScreenStatus = SCREEN_UNSET
    target: str = ""
    system_keys: tuple[str, ...] = ()
    n_pairs: int = UNCHANGED_INT
    min_spacing: int = UNCHANGED_INT
    #: The run's scored windows, if the caller has them cached.
    windows: list | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    follow_ups: tuple[str, ...] = ()


# --- small helpers ----------------------------------------------------------

@dataclass(frozen=True)
class _Pair:
    number: int
    result: object
    pair: object
    screen: object
    access: object
    fold: object

    @property
    def penalty(self) -> float:
        """What the pipeline ranked this pair by; lower is better."""
        from ..accessibility import accessibility_penalty

        return self.pair.candidate.penalty + accessibility_penalty(self.access)


def _pairs(run) -> list[_Pair]:
    out = []
    for r in run.results:
        for i, (pair, screen, access) in enumerate(r.rows()):
            out.append(_Pair(len(out) + 1, r, pair, screen, access, r.folding_at(i)))
    return out


def _join(items: list[str]) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _named(numbers: list[int]) -> str:
    """"pair 2" / "pairs 1 and 3"."""
    return ("pair " if len(numbers) == 1 else "pairs ") + _join(numbers)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _is_genomic(result) -> bool:
    from ..probe_map import _is_genomic as genomic

    return genomic(result)


def _index_for(asked: Asked) -> str:
    """A configured index name that suits the target, or ""."""
    names = asked.screen.available
    if not names:
        return ""
    want = "(genome" if (asked.run and _is_genomic(asked.run.results[0])) else "(transcriptome"
    fitting = [n for n in names if want in n]
    return (fitting or names)[0].split(" ")[0]


def _ask_for(what_if) -> str:
    """A previewed change, as the message that would design it."""
    p = what_if.params
    if what_if.change == "min_spacing":
        return f"space them {p.min_spacing} nt apart"
    if what_if.change == "n_pairs":
        return f"make it {p.n_pairs} pairs"
    return f"make it {p.n_pairs} pairs at {p.min_spacing} nt spacing"


def _no_design(asked: Asked) -> Answer:
    if asked.target:
        return Answer(
            f"Nothing has been designed on `{asked.target}` yet, so there is no "
            f"result to explain. Ask for a design first.",
            ("design 3 pairs", "how does this work?"))
    return Answer(
        "There is no target yet, so there is no design to explain. Give me a "
        "region like `chr7:5530000-5531000`, an accession like `NM_000546.6`, "
        "or paste a sequence (51 nt or more), and say how many pairs you want.",
        ("how does this work?", "what can you do?"))


# --- about this design --------------------------------------------------------

def _placement(asked: Asked) -> Answer:
    run = asked.run
    facts = diagnose(run.results, run.params)
    lines = [" ".join(f.text for f in facts if f.topic == "placement")]
    if run.params.enforce_window:
        lines.append(" ".join(f.text for f in facts if f.topic == "windows"))
    previews = try_changes(run.results, run.params, asked.windows)
    if previews:
        lines.append("What would place more:\n" + _bullets([p.text for p in previews])
                     + f"\n\n_{PREVIEW_CAVEAT}_")
        return Answer("\n\n".join(lines),
                      (*[_ask_for(p) for p in previews[:2]],
                       "how many pairs can this hold?"))
    return Answer("\n\n".join(lines),
                  ("how many pairs can this hold?", "which pair is the weakest?"))


def _signal_sites(screen) -> list:
    """Off-target sites that could give a spot: both halves bound, near enough,
    oriented right, and predicted to hold through the washes."""
    return [s for s in screen.pair_sites if s.survives_wash and s.signal_capable]


def _screen_line(p: _Pair) -> str:
    from ..ui import worst_offtarget_cell

    s = p.screen
    capable = _signal_sites(s)
    line = (f"Pair {p.number} ({s.verdict}): {len(s.pair_sites)} off-target "
            f"site{'s' if len(s.pair_sites) != 1 else ''} with both halves "
            f"nearby, {len(capable)} able to give signal")
    if s.offtarget_genes:
        line += f"; gene{'s' if len(s.offtarget_genes) != 1 else ''} "
        line += _join(s.offtarget_genes[:6])
        if len(s.offtarget_genes) > 6:
            line += f" and {len(s.offtarget_genes) - 6} more"
    worst = str(worst_offtarget_cell(s))
    if worst and worst not in ("-", "nan"):
        line += f"; worst off-target P(bound) {worst}"
    if not s.pair_sites:
        line += (f"; {len(s.surviving_single_half_hits)} lone half-hit(s) that "
                 f"would hold through the washes lowered its score")
    return line + "."


def _screen(asked: Asked) -> Answer:
    run = asked.run
    pairs = _pairs(run)
    fact = next(f.text for f in diagnose(run.results, run.params) if f.topic == "screen")
    lines = [fact]
    screened = [p for p in pairs if p.screen.screened]
    if screened:
        lines.append(f"Screened against {run.screen}." if run.screen
                     and run.screen != "off" else "")
        by_verdict: dict[str, list[int]] = {}
        for p in screened:
            by_verdict.setdefault(p.screen.verdict, []).append(p.number)
        lines.append("Verdicts: " + "; ".join(
            f"{_named(ns)} {v}" for v, ns in by_verdict.items()) + ".")
        lines.extend(_screen_line(p) for p in screened if p.screen.verdict != "pass")
    follow: tuple[str, ...] = ("how does the screen work?", "which pair is the weakest?")
    if len(screened) < len(pairs):
        name = _index_for(asked)
        if name:
            lines.append(f"To screen this design, ask me to *screen against {name}*.")
            follow = (f"screen against {name}", "how does the screen work?")
        else:
            lines.append("No usable specificity index is configured. Add a bowtie "
                         "v1 index to `indexes.local.json`, or enter its path "
                         "under *Settings*; then ask me to screen against it.")
            follow = ("how do I add an index?", "how does the screen work?")
    return Answer("\n\n".join(line for line in lines if line), follow)


def _folding(asked: Asked) -> Answer:
    pairs = _pairs(asked.run)
    done = [p for p in pairs if p.fold.evaluated]
    if not done:
        reason = (pairs[0].fold.reason or "no reason recorded").rstrip(". ")
        return Answer(f"Probe self-folding was not evaluated: {reason}.",
                      ("is this design ok?",))
    by_label: dict[str, list[int]] = {}
    for p in done:
        by_label.setdefault(p.fold.limiting.label, []).append(p.number)
    lines = [_cap("; ".join(f"{_named(ns)} {'is' if len(ns) == 1 else 'are'} {label}"
                            for label, ns in by_label.items()) + "."),
             "That is the more folded of each pair's two primaries: how much of "
             "its pad or its arm is paired in the oligo's own fold (open below "
             "50%, partly folded below 65%, folded above). A tied-up arm cannot "
             "present its half of the P2 site.",
             _cap(FOLDING_IS_UNCALIBRATED) + "."]
    worst = max(done, key=lambda p: p.fold.worst_paired)
    return Answer("\n\n".join(lines),
                  (f"tell me about pair {worst.number}", "is this design ok?"))


def _accessibility(asked: Asked) -> Answer:
    pairs = _pairs(asked.run)
    done = [p for p in pairs if p.access.evaluated]
    if not done:
        reason = (pairs[0].access.reason or "no reason recorded").rstrip(". ")
        return Answer(f"Target-site accessibility was not evaluated: {reason}.",
                      ("is this design ok?",))
    lines = ["; ".join(f"pair {p.number} {p.access.pair_nucleation_p:.3f}"
                       for p in done)]
    lines[0] = "Accessibility: " + lines[0] + "."
    lines.append(
        f"The number is the chance that a stretch of "
        f"{done[0].access.nucleation_nt} nt in the pad is unpaired in the "
        f"target's own predicted "
        f"structure, so a probe can start binding there; higher is more open. "
        f"A pair takes its worse pad, since both halves must bind, and a less "
        f"accessible pair ranks lower.")
    lowest = min(done, key=lambda p: p.access.pair_nucleation_p)
    return Answer("\n\n".join(lines),
                  (f"tell me about pair {lowest.number}", "which pair is the weakest?"))


#: What each QC flag means, keyed by the word a flag contains.
FLAG_MEANINGS = {
    "homopolymer": ("`homopolymerN`: a run of N identical bases in that pad. "
                    "Flagged at 5 or more, and adds 1.0 to the ranking penalty."),
    "dinuc": ("`dinucN`: a perfect dinucleotide repeat N bases long in that pad "
              "(ATATATAT is 8). Flagged at 8 or more, and adds 1.0 to the "
              "ranking penalty."),
    "gc": ("`gcN%`: that pad's GC content is outside 30-70%. Reported only; GC "
           "already counts in the ranking penalty."),
    "spans exon junction": ("`spans exon junction`: the footprint crosses an "
                            "exon boundary of the transcript. Reported, not "
                            "penalised."),
}


def _qc_flags(asked: Asked) -> Answer:
    flagged = [p for p in _pairs(asked.run) if p.pair.candidate.flags]
    if flagged:
        lines = ["; ".join(f"pair {p.number}: {', '.join(p.pair.candidate.flags)}"
                           for p in flagged)]
        lines[0] = _cap(lines[0]) + "."
        present = " ".join(f for p in flagged for f in p.pair.candidate.flags)
        kinds = [text for key, text in FLAG_MEANINGS.items() if key in present]
    else:
        lines = ["No placed pair carries a QC flag."]
        kinds = list(FLAG_MEANINGS.values())
        lines.append("The flags a pair can carry:")
    lines.append(_bullets(kinds))
    lines.append("`L:` and `R:` say which pad, left or right. A flag never "
                 "removes a pair; it lowers its rank or is reported.")
    return Answer("\n\n".join(lines),
                  ((f"tell me about pair {flagged[0].number}",) if flagged else ())
                  + ("what is penalty?",))


def _repeats(asked: Asked) -> Answer:
    lines = []
    for r in asked.run.results:
        if _is_genomic(r):
            n = len(r.rmsk)
            lines.append(
                f"{n} RepeatMasker annotation{'s' if n != 1 else ''} overlap"
                f"{'s' if n == 1 else ''} the target and "
                f"{'was' if n == 1 else 'were'} soft-masked, so no pad sits in "
                f"repetitive DNA." if n else
                "No RepeatMasker annotation overlaps the target, so nothing "
                "was masked.")
        else:
            lines.append("Repeat masking applies to genomic regions only: a "
                         "transcript or a pasted sequence has no RepeatMasker "
                         "track, so nothing was masked.")
    return Answer(" ".join(dict.fromkeys(lines)),
                  ("why did I get this many pairs?", "is this design ok?"))


def _summary(asked: Asked) -> Answer:
    run = asked.run
    facts = [f.text for f in diagnose(run.results, run.params)]
    _, unscreened, _ = screening(run.results)
    lines = [f"Run #{run.number} on `{asked.target or run.target_label}`:",
             _bullets(facts)]
    lines.append(
        "The order workbook can be downloaded as it is. "
        + ("Pairs that were not screened are marked NOT SCREENED in it, and "
           "nothing establishes that they are single-copy."
           if unscreened else
           "Every pair in it was screened; read any verdict other than *pass* "
           "before ordering."))
    name = _index_for(asked)
    return Answer("\n\n".join(lines),
                  ((f"screen against {name}",) if unscreened and name else ())
                  + ("which pair is the weakest?", "how do I get the oligos?"))


# --- about a pair ---------------------------------------------------------------

#: (words in the message, which line to lead with) for `_pair_detail`.
_FOCUS = (
    (r"\btm\b|melting|temperature", "tm"),
    (r"\bgc\b", "gc"),
    (r"where|position|location|footprint|coordinat|\bsit\b|\bsits\b", "where"),
    (r"penalt|rank|score", "penalty"),
    (r"screen|specific|off.?target|verdict|review|fail|pass", "screen"),
    (r"fold", "fold"),
    (r"access|open\b", "access"),
    (r"flag|\bqc\b|homopolymer|dinuc", "flags"),
)


def _pair_lines(detail, p: _Pair) -> dict[str, str]:
    n = detail.pair
    lines = {
        "tm": (f"Pair {n}'s Tm is {detail.tm_left:g} C for the left pad and "
               f"{detail.tm_right:g} C for the right pad."),
        "gc": (f"Pair {n}'s GC is {detail.gc_left:g} for the left pad and "
               f"{detail.gc_right:g} for the right pad."),
        "where": f"Pair {n} sits at `{detail.footprint}`.",
        "penalty": (f"Pair {n}'s ranking penalty is {detail.penalty:g}; lower "
                    f"is better."),
        "screen": (_screen_line(p) if p.screen.screened else
                   f"Pair {n} was NOT SCREENED: "
                   f"{(p.screen.reason or 'no reason recorded').rstrip('. ')}."),
        "fold": f"Pair {n}'s self-folding: {detail.self_folding}.",
        "access": f"Pair {n}'s accessibility: {detail.accessibility}.",
        "flags": f"Pair {n}'s QC flags: {detail.qc_flags}.",
    }
    return lines


def _pair_detail(asked: Asked) -> Answer:
    pairs = _pairs(asked.run)
    numbers = pairs_named(asked.message)
    if not numbers and len(pairs) == 1:
        numbers = [1]
    if not numbers:
        return Answer(f"Which pair? This design has {len(pairs)}.",
                      tuple(f"tell me about pair {i}" for i in range(1, min(len(pairs), 3) + 1)))
    missing = [n for n in numbers if not 1 <= n <= len(pairs)]
    numbers = [n for n in numbers if 1 <= n <= len(pairs)]
    out = []
    if missing:
        out.append(f"There is no {_named(missing)}: this design has "
                   f"{len(pairs)} pair{'s' if len(pairs) != 1 else ''}.")
    wanted = [key for pattern, key in _FOCUS
              if re.search(pattern, asked.message, re.IGNORECASE)]
    for detail in _details(asked.run, numbers):
        lines = _pair_lines(detail, pairs[detail.pair - 1])
        lead = [lines[k] for k in wanted]
        rest = [lines[k] for k in lines if k not in wanted]
        if lead:
            out.append(" ".join(lead))
            out.append("The rest of pair " + str(detail.pair) + ":\n" + _bullets(rest))
        else:
            out.append(_bullets(list(lines.values())))
    others = [p.number for p in pairs if p.number not in numbers]
    follow = ((f"compare pair {numbers[0]} and pair {others[0]}",)
              if numbers and others else ())
    return Answer("\n\n".join(out), follow + ("which pair is the weakest?",))


def _compare_pairs(asked: Asked) -> Answer:
    pairs = _pairs(asked.run)
    numbers = [n for n in pairs_named(asked.message) if 1 <= n <= len(pairs)]
    if len(pairs) < 2:
        return Answer("This design has one pair, so there is nothing to compare "
                      "it with.", ("tell me about pair 1", "could I fit 3 pairs?"))
    if len(numbers) < 2:
        return Answer(f"Which two pairs? This design has {len(pairs)}.",
                      ("compare pair 1 and pair 2", "which pair is the weakest?"))
    a, b = _details(asked.run, numbers[:2])
    lines = compare_pairs(a, b)
    better, worse = (a, b) if a.penalty <= b.penalty else (b, a)
    head = (f"Pair {a.pair} and pair {b.pair} rank the same." if a.penalty == b.penalty
            else f"Pair {better.pair} ranks better than pair {worse.pair}: its "
                 f"penalty is {better.penalty:g} against {worse.penalty:g}, and "
                 f"lower is better.")
    out = [head, _bullets([_cap(line) + "." for line in lines])]
    verdicts = {pairs[d.pair - 1].screen.verdict for d in (a, b)}
    if verdicts != {"pass"}:
        out.append("The ranking does not include the screen: "
                   + "; ".join(f"pair {d.pair} is {d.specificity}" for d in (a, b)) + ".")
    return Answer("\n\n".join(out),
                  (f"tell me about pair {worse.pair}", "what is penalty?"))


def _best_worst(asked: Asked) -> Answer:
    pairs = _pairs(asked.run)
    if len(pairs) < 2:
        return Answer("This design has one pair, so it is both the best and the "
                      "weakest.", ("tell me about pair 1", "could I fit 3 pairs?"))
    ranked = sorted(pairs, key=lambda p: p.penalty)
    best, worst = ranked[0], ranked[-1]
    lines = [f"Best ranked: pair {best.number} (penalty {best.penalty:.3g}). "
             f"Weakest: pair {worst.number} (penalty {worst.penalty:.3g}). "
             f"Lower is better.",
             "The penalty is how far a pair's GC is from 50% and how unequal "
             "its two pads' Tm are, plus accessibility for an RNA target. It "
             "does not include the screen."]
    not_pass = [p for p in pairs if p.screen.screened and p.screen.verdict != "pass"]
    unscreened = [p.number for p in pairs if not p.screen.screened]
    if not_pass:
        lines.append("On the screen: " + "; ".join(
            f"pair {p.number} is {p.screen.verdict}" for p in not_pass) + ".")
    elif len(unscreened) == len(pairs):
        lines.append("No pair was screened, so the screen has nothing to add "
                     "to that ranking.")
    elif unscreened:
        lines.append(f"{_cap(_named(unscreened))} "
                     f"{'was' if len(unscreened) == 1 else 'were'} not screened; "
                     f"the rest pass.")
    else:
        lines.append("Every pair passes the screen.")
    return Answer("\n\n".join(lines),
                  (f"compare pair {best.number} and pair {worst.number}",
                   f"tell me about pair {worst.number}"))


# --- what if ----------------------------------------------------------------------

def _cannot_preview() -> Answer:
    return Answer(f"I cannot preview a change to this design: {NOT_PREVIEWABLE}. "
                  f"Ask for the design instead and it will be run.",
                  ("is this design ok?",))


def _what_if(asked: Asked) -> Answer:
    run = asked.run
    new = run.params
    if asked.n_pairs != UNCHANGED_INT:
        new = replace(new, n_pairs=asked.n_pairs)
    if asked.min_spacing != UNCHANGED_INT:
        new = replace(new, min_spacing=asked.min_spacing)
    if new == run.params:
        return _get_more(asked)
    new, clamps = clamp(new, asked.system_keys)
    preview = preview_change(run.results, run.params, new, asked.windows)
    if preview is None:
        return _cannot_preview()
    lines = [preview.text]
    lines.extend(f"({c})" for c in clamps)
    if preview.pairs_placed < preview.pairs_requested:
        lines.append(f"That is {preview.pairs_requested - preview.pairs_placed} "
                     f"short of what it asks for.")
    lines.append(f"_{PREVIEW_CAVEAT}_")
    short = preview.pairs_placed < preview.pairs_requested
    return Answer("\n\n".join(lines),
                  ("how many pairs can this hold?", "how do I get more pairs?")
                  if short else
                  (_ask_for(preview), "how many pairs can this hold?"))


def _capacity(asked: Asked) -> Answer:
    run = asked.run
    top = int(BOUNDS["n_pairs"][1])
    spacing = run.params.min_spacing
    at = preview_change(run.results, run.params,
                        replace(run.params, n_pairs=top), asked.windows)
    if at is None:
        if len(run.results) != 1:
            return _cannot_preview()
        length = len(run.results[0].region_seq)
        fit = max(0, (length + spacing) // (FOOTPRINT + spacing))
        return Answer(
            f"I cannot preview this design, but by arithmetic the {length} nt "
            f"target holds at most {fit} pairs at {spacing} nt minimum spacing "
            f"(each needs a {FOOTPRINT} nt footprint), and fewer if some "
            f"positions have no usable window.", ("is this design ok?",))

    def said(n: int) -> str:
        return f"{top} or more ({top} is the most one design tiles)" if n >= top else str(n)

    lines = [f"At {spacing} nt minimum spacing the target's usable windows "
             f"hold {said(at.pairs_placed)} pairs. This design has "
             f"{sum(r.n_pairs for r in run.results)}."]
    follow: list[str] = []
    if at.pairs_placed > run.params.n_pairs:
        follow.append(f"make it {min(at.pairs_placed, top)} pairs")
    if spacing > 0:
        touching = preview_change(run.results, run.params,
                                  replace(run.params, n_pairs=top, min_spacing=0),
                                  asked.windows)
        if touching is not None and touching.pairs_placed > at.pairs_placed:
            lines.append(f"With the footprints allowed to touch (0 nt spacing) "
                         f"they hold {said(touching.pairs_placed)}.")
            follow.append(f"what if the spacing were {spacing // 2}?")
    lines.append(f"_{PREVIEW_CAVEAT}_")
    return Answer("\n\n".join(lines), (*follow, "which pair is the weakest?"))


def _get_more(asked: Asked) -> Answer:
    run = asked.run
    previews = try_changes(run.results, run.params, asked.windows)
    if previews:
        return Answer("The design is short of what was asked. What would place "
                      "more:\n\n" + _bullets([p.text for p in previews])
                      + f"\n\n_{PREVIEW_CAVEAT}_",
                      (*[_ask_for(p) for p in previews[:2]],
                       "why did I get fewer pairs?"))
    placed = sum(r.n_pairs for r in run.results)
    capacity = _capacity(asked)
    return Answer(f"All {placed} requested pair{'s' if placed != 1 else ''} "
                  f"{'was' if placed == 1 else 'were'} placed, so nothing is "
                  f"holding the design back: ask for more and it will tile "
                  f"them.\n\n" + capacity.text, capacity.follow_ups)


# --- about the method (fixed text, from the README and the Help page) -------------

ABOUT_PIFISH = (
    "piFISH labels a target with **split primaries**. P1L and P1R bind two "
    "adjacent 25 nt stretches of the target with a 1 nt gap between them: a "
    "51 nt footprint. Each carries a constant arm, and the two arms meet at "
    "the gap and together form the landing site for the P2 secondary, so P2 "
    "binds only where both halves are bound side by side. That coincidence is "
    "where the specificity comes from.\n\n"
    "P2 carries sites for P3, and P3 for the fluorophore-labelled P4, which "
    "amplifies the signal. The amplifier channel (cy3, cy5, ...) fixes the "
    "arms, P2, P3 and P4, so only P1L and P1R are new oligos for each target. "
    "Several pairs tiled along a target add signal.\n\n"
    "Method: Tao et al., *Nat Commun* 14:443 (2023).")

ABOUT_SCREENING = (
    "Each pair's two pads are aligned against the chosen index with bowtie v1, "
    "reporting every hit within 3 mismatches.\n\n"
    "The screen is **pair-aware**. A hit for one half alone cannot recruit P2, "
    "so it cannot make a spot. What counts is a site where both halves land "
    "near each other, close enough and oriented to rebuild the P2 site, with "
    "both duplexes predicted to hold through the washes. Each such site is "
    "priced as a P(bound).\n\n"
    "- **pass**: no off-target site with both halves nearby.\n"
    "- **review**: such a site exists but is not predicted to give signal, or "
    "lone half-hits lowered the score.\n"
    "- **fail**: at least one off-target site could give signal.\n"
    "- **NOT SCREENED**: no screen ran. It is not a pass, and the workbook "
    "says so.\n\n"
    "A genomic region or a fusion is screened against a genome; an RNA target "
    "against a transcriptome, where sites are counted per gene and hits to the "
    "target's own gene are not off-targets.")

#: (words that ask for it, the term, what it means), for `_glossary`.
GLOSSARY = (
    (r"penalt|rank|score", "Penalty",
     "the score a window is ranked by; lower is better. It adds how far each "
     "pad's GC and their mean are from 50%, the Tm difference between the two "
     "pads divided by 20, and 1.0 for each homopolymer or dinucleotide-repeat "
     "flag. For an RNA target with accessibility evaluated, an accessibility "
     "penalty is added."),
    (r"p\(?bound|bound|occupan", "P(bound)",
     "the equilibrium fraction bound at the limiting step, given for the "
     "target and for the worst off-target site that could give signal. A "
     "ranking number, not an absolute prediction."),
    (r"\btm\b|melting", "Tm",
     "the predicted melting temperature of each 25 nt pad on the target, "
     "nearest-neighbour, in the system's hybridization buffer. Matched Tm in "
     "the two pads lets them bind together."),
    (r"window|\bgc\b|enforce", "GC/Tm window",
     "the GC and Tm range the lab aims for. It is reported for context: the "
     "pipeline ranks windows by distance from it and rejects on it only when "
     "you ask to *enforce the GC/Tm window*."),
    (r"spacing|apart", "Minimum spacing",
     "the fewest bases allowed between two neighbouring footprints."),
    (r"footprint|pad|gap", "Footprint",
     "the 51 nt a pair covers on the target: a 25 nt left pad, a 1 nt gap, a "
     "25 nt right pad."),
    (r"verdict|pass|review|fail|not screened", "Verdict",
     "the screen's call on a pair: pass, review or fail, or NOT SCREENED when "
     "no screen ran. Ask *how does the screen work?* for what each means."),
    (r"fold", "Self-folding",
     "how much of each primary is paired in its own fold. A tied-up arm cannot "
     "present its half of the P2 site. Its cutoffs are uncalibrated."),
    (r"access", "Accessibility",
     "for an RNA target, the chance the binding site is unpaired in the "
     "transcript's own structure; higher is more open. Not evaluated for DNA, "
     "which the protocol denatures."),
    (r"express|abundan|tpm", "Expression weighting",
     "scaling each off-target gene by its abundance in the imaged cell type, "
     "when an expression table is supplied. Uncalibrated."),
)

TARGET_TYPES = (
    "- **Genomic region**: given as coordinates. Repeats are soft-masked from "
    "RepeatMasker, so no pad sits in repetitive DNA, and it is screened "
    "against a genome.\n"
    "- **RNA transcript**: given as an accession. Screened against a "
    "transcriptome, and target-site accessibility is evaluated when "
    "RNAplfold is installed.\n"
    "- **Pasted sequence**: designed in the pasted coordinates, so it has no "
    "genome position. With a screen on, the target itself may be reported as "
    "an off-target hit, because the screen does not know its gene.\n"
    "- **Fusion junction**: one position, with the gap on the breakpoint, so "
    "tiling does not apply. Both strands are designed. Fusions come from the "
    "Form or the Batch template, not the chat.")

PROBE_MAP = (
    "- **Ordered pairs**: each is a 51 nt footprint. Click one for its "
    "close-up.\n"
    "- **Previous run** (grey): the pairs of the last run on the same target, "
    "so a change shows as a shift.\n"
    "- **Best windows**: the 40 best-scoring windows, stacked into lanes where "
    "they overlap. Green ones were ordered.\n"
    "- **Penalty**: the score of every window along the target; lower is "
    "better, and the axis is flipped so good windows are high. For RNA the "
    "pipeline also re-ranks by accessibility, so the order of pairs can "
    "differ from this track.\n"
    "- **Close-up**: the target 5'→3' on top, with P1L and P1R bound beneath. "
    "Both arms hang off at the gap, where P2 lands only when both halves are "
    "bound. Probe bases are shaded red by how often they are paired in the "
    "probe's own fold, when folding was computed.")

HOW_ORDER = (
    "Press **Download order workbook (.xlsx)** under the design on the right. "
    "It holds the P1L and P1R sequences for the run that is selected, named "
    "`P1L_<gene>_<name>`; pick another run from the run menu to download that "
    "one instead.\n\n"
    "The workbook says in writing which pairs were not screened, and whether "
    "placeholder amplifier systems were in use (those are not orderable). I "
    "never see or write a sequence: they exist only in that file.")

CAN_DO = (
    "I can:\n"
    "- **change the design**: the pair count, the spacing, the channel, the "
    "specificity screen, and whether the GC/Tm window is enforced;\n"
    "- **explain it**: why it placed the pairs it did, what the screen found, "
    "any one pair, or how two pairs differ;\n"
    "- **preview a change** without designing it: *could I fit 10?*;\n"
    "- **answer questions about the method and the tool**: how piFISH probes "
    "work, what a number means, how to order.")

NEEDS_TARGET = ("Give me a target to start: a region (`chr7:5530000-5531000`), "
                "an accession (`NM_000546.6`) or a pasted sequence.")


def after_design(asked: Asked) -> tuple[str, ...]:
    """What to offer next once `asked.run` has just been designed."""
    run = asked.run
    pairs = _pairs(run)
    out = []
    name = _index_for(asked)
    if name and any(not p.screen.screened for p in pairs):
        out.append(f"screen against {name}")
    if len(pairs) < run.params.n_pairs * len(run.results):
        out.append("why did I get fewer pairs?")
    out.append("is this design ok?")
    if len(pairs) > 1:
        out.append("which pair is the weakest?")
    return tuple(out[:3])


def starters(asked: Asked) -> tuple[str, ...]:
    """What to offer when there is nothing more specific to suggest."""
    return _starters(asked)


def _starters(asked: Asked) -> tuple[str, ...]:
    if asked.run is not None:
        return ("is this design ok?", "which pair is the weakest?",
                "how do I get more pairs?")
    if asked.target:
        return ("design 3 pairs", "how does this work?")
    return ("how does this work?", "what are the kinds of target?")


def _glossary(asked: Asked) -> Answer:
    hits = [(term, text) for pattern, term, text in GLOSSARY
            if re.search(pattern, asked.message, re.IGNORECASE)]
    if not hits:
        return Answer("The terms I can explain: "
                      + _join([term for _, term, _ in GLOSSARY])
                      + ". Which one?",
                      ("what is penalty?", "what is P(bound)?"))
    text = "\n\n".join(f"**{term}**: {meaning}" for term, meaning in hits[:3])
    return Answer(text, ("how does the screen work?",) + _starters(asked)[:1])


def _how_setup(asked: Asked) -> Answer:
    available = asked.screen.available
    lines = [
        "- **A target**: give it to me in a message (a region, an accession or "
        "a pasted sequence), or use the Form. For several at once, use Batch "
        "and its template. Targets the command line and the MCP server use by "
        "name live in `loci.local.json`.",
        "- **A specificity index**: build a bowtie v1 index of the genome or "
        "transcriptome (`bowtie-build`), then name it in `indexes.local.json` "
        "with its kind and path, or enter the path under *Settings*. "
        + (f"Configured now: {_join(available)}." if available
           else "None is configured now.")
        + " A target should be screened against its own species' index.",
        "- **A channel**: amplifier systems are read from `systems.local.json`; "
        "add the arms, P2, P3 and P4 there. "
        + (f"Configured now: {_join(list(asked.system_keys))}."
           if asked.system_keys else ""),
    ]
    return Answer(_bullets([line[2:] for line in lines]),
                  ("how does the screen work?", "what are the kinds of target?"))


def _fixed(text: str, follow: tuple[str, ...]):
    return lambda asked: Answer(text, follow)


def _can_do(asked: Asked) -> str:
    return CAN_DO if asked.target else CAN_DO + "\n\n" + NEEDS_TARGET


def _greeting(asked: Asked) -> Answer:
    return Answer(_can_do(asked), _starters(asked))


def _out_of_scope(asked: Asked) -> Answer:
    return Answer("That is outside what I can answer here: I only know this "
                  "design and this tool.\n\n" + _can_do(asked), _starters(asked))


#: Topics that explain the current design, and so need one.
NEEDS_DESIGN = frozenset({"placement", "screen", "folding", "accessibility",
                          "qc_flags", "repeats", "summary", "pair_detail",
                          "compare_pairs", "best_worst", "what_if", "capacity",
                          "get_more"})

_TEMPLATES = {
    "placement": _placement,
    "screen": _screen,
    "folding": _folding,
    "accessibility": _accessibility,
    "qc_flags": _qc_flags,
    "repeats": _repeats,
    "summary": _summary,
    "pair_detail": _pair_detail,
    "compare_pairs": _compare_pairs,
    "best_worst": _best_worst,
    "what_if": _what_if,
    "capacity": _capacity,
    "get_more": _get_more,
    "about_pifish": _fixed(ABOUT_PIFISH, ("how does the screen work?",
                                          "what are the kinds of target?")),
    "about_screening": _fixed(ABOUT_SCREENING, ("what is P(bound)?",
                                                "how do I add an index?")),
    "glossary": _glossary,
    "target_types": _fixed(TARGET_TYPES, ("how does this work?",
                                          "how does the screen work?")),
    "probe_map": _fixed(PROBE_MAP, ("what is penalty?", "how do I get the oligos?")),
    "how_order": _fixed(HOW_ORDER, ("is this design ok?",)),
    "how_setup": _how_setup,
    "greeting": _greeting,
    "out_of_scope": _out_of_scope,
}

assert set(_TEMPLATES) == set(TOPIC_KEYS), "every topic needs exactly one template"


def answer(topic: str, asked: Asked) -> Answer | None:
    """The software's answer for `topic`, or None when it has none.

    None for "other" and "none" (and anything unknown): the caller then shows
    the model's own note, checked, as it did before topics existed.
    """
    if topic in (OTHER, NONE) or topic not in _TEMPLATES:
        return None
    if topic in NEEDS_DESIGN and asked.run is None:
        return _no_design(asked)
    return _TEMPLATES[topic](asked)
