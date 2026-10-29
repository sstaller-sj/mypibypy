"""What the chat model is shown about the current design, as a typed schema.

The design state lives in `session.DesignSession`. This is the *view* of it the
model reads on each chat turn -- pushed into the request, not fetched by the
model -- and the one place that decides what that view may contain:

* **Every turn:** the target's label, every design setting, and a summary of
  the last run on this target (pairs placed vs requested, how the windows fare
  against the GC/Tm window, screening, the worst off-target, the most folded
  primary). That is what the model's choices actually turn on: whether a
  follow-up changes a setting, what would place more, why nothing passed.
  The sentences in it come from `mypibypy.diagnosis`, the layer meant to serve
  the MCP server and the results panel too, so a fact is worked out once.
* **On demand:** per-pair detail, only for the pairs the user's message names
  ("why is pair 3 worse than pair 2?"). Twelve pairs of numbers on every turn
  would be noise -- and a small model given a table tends to misquote it.

NO SEQUENCE CAN BE CARRIED. Every field is a label, a number, an enum-like word
or a gene name; the target is its label ("pasted sequence (420 nt)"), never its
bases. The models forbid extra fields, and `tests/test_chat_context.py` checks
the schema and a serialized context for anything shaped like a sequence.

Each turn is a separate model call, so this snapshot is how the *design*
continues: the model cannot misremember a target or act on a setting that has
since changed. The last few chat lines go alongside it (`intent.chat_request`),
redacted, only so a follow-up question ("no, for the screen") has something to
refer to -- they never stand in for the state held here.
"""
from __future__ import annotations

import math
import re

from pydantic import BaseModel, ConfigDict

from ..design import FOOTPRINT
from ..diagnosis import (
    diagnose,
    flagged_pairs,
    most_folded,
    placement_reason,  # noqa: F401 -- re-exported; the chat tests import it here
    screening,
    tm_vs_window,
    try_changes,
    windows_text,
)
from .actions import DesignParameters

#: "pair 3", "pairs 2 and 4", "#3", "p3" -- the pairs a message asks about.
_PAIR_RE = re.compile(r"(?:\bpairs?\s*#?|#|\bp)(\d{1,2}(?:\s*(?:,|and|&|vs\.?|or)"
                      r"\s*#?\d{1,2})*)\b", re.IGNORECASE)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Settings(_Strict):
    """Every design parameter, current values. Mirrors `DesignParameters`."""

    system: str
    n_pairs: int
    min_spacing: int
    gc_min: float
    gc_max: float
    tm_min: float
    tm_max: float
    enforce_window: bool
    top: int
    max_mismatches: int
    accept_single_arm_hits: bool
    widen_by_nt: int

    @classmethod
    def from_params(cls, params: DesignParameters) -> Settings:
        return cls(**{name: getattr(params, name) for name in cls.model_fields})


class RunSummary(_Strict):
    """How the last design on this target came out. Numbers and words only."""

    run_number: int
    pairs_requested: int
    pairs_placed: int
    limiting_constraint: str
    pairs_screened: int
    pairs_not_screened: int
    #: Highest off-target P(bound) over screened pairs; None if none screened
    #: or none priced.
    worst_offtarget_p_bound: float | None
    #: The most folded primary's label ("open", "partly folded", ...), or
    #: "not evaluated".
    most_folded: str
    pairs_with_qc_flags: int


class PairDetail(_Strict):
    """One ordered pair, as the results table shows it. No sequence."""

    pair: int
    footprint: str
    tm_left: float
    tm_right: float
    gc_left: float
    gc_right: float
    accessibility: str
    specificity: str
    on_target_p_bound: str
    worst_offtarget_p_bound: str
    self_folding: str
    offtarget_genes: str
    qc_flags: str
    #: The score the pipeline ranked this window by; lower is better. What
    #: "worse" means to the design itself: the GC/Tm penalty plus, for an RNA
    #: target with accessibility evaluated, the accessibility penalty at the
    #: pipeline's default weight -- the base score alone could rank the pair
    #: the other way round.
    penalty: float


class WindowRow(_Strict):
    """One of the top-ranked windows kept from the scan. No sequence.

    Only the best few windows are kept (the rest of the scan is not retained),
    so this table cannot answer "what about the window near nt 100" unless that
    window ranked in the top few -- `ChatContext.windows_note` says so.
    """

    rank: int
    #: 1-based nt range within the target, e.g. "96-146".
    target_position: str
    tm_left: float
    tm_right: float
    gc_left: float
    gc_right: float
    penalty: float
    #: Both pads against the Tm window, in words ("both pads within 55-62 C",
    #: "left pad 52.3 C is 2.7 below 55"). Done here so the model need not
    #: compare numbers.
    tm_vs_window: str
    #: "placed as pair N", or why it was not placed.
    status: str


#: How the screen is turned on, said in the data so the model can answer "how do
#: I turn the screen on" -- without it, it pointed the user at pasting a
#: target, the only instruction it had.
HOW_TO_SCREEN = ("Set `screen` to a name in `available`; or the user adds an "
                 "index to indexes.local.json, or enters a bowtie v1 index "
                 "path in the sidebar under Settings > 'bowtie index "
                 "(optional)'. The model never writes a path.")


class ScreenStatus(_Strict):
    """The specificity screen: on or off, against what, and the named choices.

    `against` is a name or "off" -- never a path, which stays in the app.
    """

    on: bool
    against: str
    #: "name (kind, description)" for each usable configured index.
    available: list[str] = []
    how_to_turn_on: str = HOW_TO_SCREEN


SCREEN_UNSET = ScreenStatus(on=False, against="off")


WINDOWS_NOTE = ("`windows` lists only the top-ranked windows kept from the scan; "
                "other windows (e.g. one at a given nt position) were scored but "
                "not kept, so say you cannot see them rather than guessing. "
                "Positions are 1-based within the target.")


class ChatContext(_Strict):
    target: str | None
    specificity_screen: ScreenStatus = SCREEN_UNSET
    settings: Settings
    last_run: RunSummary | None = None
    #: Only the pairs the message names; empty otherwise.
    pair_details: list[PairDetail] = []
    #: Every numeric difference between the first two named pairs, worked out
    #: by the software. A small model handed two numbers called 70.4 "higher"
    #: than 73.8; handed the comparison, it only has to repeat it.
    pair_comparison: list[str] = []
    #: The top-ranked windows of the last run, with where each stands.
    windows: list[WindowRow] = []
    windows_note: str = ""
    #: `diagnosis.diagnose` on the last run: placement and why any pairs are
    #: missing, the GC/Tm window, screening, folding. Sentences to repeat --
    #: the model guessed reasons ("the sequence was too short") when left to
    #: infer them.
    facts: list[str] = []
    #: `diagnosis.try_changes`: what a smaller spacing or fewer pairs would
    #: place, previewed by the software, when the last run fell short.
    what_if: list[str] = []

    def render(self) -> str:
        """The JSON the model reads, compact and with unset parts left out."""
        return self.model_dump_json(exclude_none=True, indent=1)


def pairs_named(message: str) -> list[int]:
    """Pair numbers the message refers to, in order, without repeats."""
    found: list[int] = []
    for match in _PAIR_RE.finditer(message):
        for number in re.findall(r"\d{1,2}", match.group(1)):
            if int(number) not in found:
                found.append(int(number))
    return found


def _window_status(cand, result, spacing: int) -> str:
    placed = {p.candidate.offset: i + 1
              for i, p in enumerate(sorted(result.pairs,
                                           key=lambda p: p.candidate.offset))}
    if cand.offset in placed:
        return f"placed as pair {placed[cand.offset]}"
    for offset, number in placed.items():
        if abs(cand.offset - offset) < FOOTPRINT + spacing:
            return (f"not placed: overlaps or is within {spacing} nt of "
                    f"pair {number}")
    return "not placed: the requested number of pairs was already reached"


def _windows(run, params: DesignParameters) -> list[WindowRow]:
    rows = []
    for r in run.results:
        for rank, c in enumerate(r.candidates, 1):
            rows.append(WindowRow(
                rank=rank, target_position=f"{c.offset + 1}-{c.offset + FOOTPRINT}",
                tm_left=round(c.left.tm, 1), tm_right=round(c.right.tm, 1),
                gc_left=round(c.left.gc, 3), gc_right=round(c.right.gc, 3),
                penalty=round(c.penalty, 3),
                tm_vs_window=tm_vs_window(c, params),
                status=_window_status(c, r, run.params.min_spacing)))
    return rows


def _summary(run) -> RunSummary:
    results = run.results
    screened, not_screened, worst = screening(results)
    return RunSummary(
        run_number=run.number,
        pairs_requested=run.params.n_pairs * len(results),
        pairs_placed=sum(r.n_pairs for r in results),
        limiting_constraint=windows_text(results, run.params)[1],
        pairs_screened=screened,
        pairs_not_screened=not_screened,
        worst_offtarget_p_bound=worst,
        most_folded=most_folded(results),
        pairs_with_qc_flags=flagged_pairs(results),
    )


def _details(run, numbers: list[int]) -> list[PairDetail]:
    from ..ui import candidate_rows

    rows = [row for r in run.results for row in candidate_rows(r)]
    from ..accessibility import accessibility_penalty

    pairs = [pair for r in run.results for pair in r.pairs]
    access = [a for r in run.results for _, _, a in r.rows()]
    out = []
    for number in numbers:
        if not 1 <= number <= len(rows):
            continue
        row = rows[number - 1]
        out.append(PairDetail(
            pair=number, footprint=row["footprint"],
            tm_left=row["Tm L"], tm_right=row["Tm R"],
            gc_left=row["GC L"], gc_right=row["GC R"],
            accessibility=str(row["accessibility"]),
            specificity=str(row["specificity"]),
            on_target_p_bound=str(row["on-target P(bound)"]),
            worst_offtarget_p_bound=str(row["worst off-target P(bound)"]),
            self_folding=str(row["self-folding"]),
            offtarget_genes=str(row["off-target genes"]),
            qc_flags=str(row["QC flags"]),
            penalty=round(pairs[number - 1].candidate.penalty
                          + accessibility_penalty(access[number - 1]), 3),
        ))
    return out


#: (field, label, what a lower value means) for `compare_pairs`.
_COMPARED = (("penalty", "ranking penalty", "better ranked"),
             ("tm_left", "left-pad Tm", ""), ("tm_right", "right-pad Tm", ""),
             ("gc_left", "left-pad GC", ""), ("gc_right", "right-pad GC", ""),
             ("accessibility", "accessibility", "less accessible"))


def compare_pairs(a: PairDetail, b: PairDetail) -> list[str]:
    """Plain statements of how pair `a` differs from pair `b`, number by number."""
    out = []
    for name, label, lower_means in _COMPARED:
        try:
            x, y = float(getattr(a, name)), float(getattr(b, name))
        except ValueError:      # "not evaluated" and the like
            continue
        if math.isclose(x, y, abs_tol=1e-9):
            out.append(f"pair {a.pair} and pair {b.pair} have the same {label} ({x:g})")
            continue
        word = "lower" if x < y else "higher"
        line = (f"pair {a.pair} {label} {x:g} is {abs(x - y):.3g} {word} than "
                f"pair {b.pair} ({y:g})")
        if lower_means:
            line += f"; lower is {lower_means}"
        out.append(line)
    return out


def build_chat_context(message: str, params: DesignParameters,
                       target_label: str = "", run=None, *,
                       screen: ScreenStatus = SCREEN_UNSET,
                       windows=None) -> ChatContext:
    """The context for one chat turn.

    `run` is the last `session.Run` on this target, or None. The app decides
    that "on this target" -- a run on another target would describe the wrong
    design. `windows` is that run's scored windows, if the caller has them
    cached; otherwise the previews re-score them.
    """
    if run is None:
        return ChatContext(target=target_label or None, specificity_screen=screen,
                           settings=Settings.from_params(params))
    details = _details(run, pairs_named(message))
    return ChatContext(
        target=target_label or None,
        specificity_screen=screen,
        settings=Settings.from_params(params),
        last_run=_summary(run),
        pair_details=details,
        pair_comparison=(compare_pairs(details[0], details[1])
                         if len(details) >= 2 else []),
        windows=_windows(run, params),
        windows_note=WINDOWS_NOTE,
        facts=[f.text for f in diagnose(run.results, run.params)],
        what_if=[w.text for w in try_changes(run.results, run.params, windows)],
    )


#: Said before the facts when the model's answer could not be checked.
FACTS_FALLBACK = ("I could not give an answer I could check against the design, "
                  "so here is what the software found:")


def facts_reply(context: ChatContext) -> str:
    """The facts and previews as a reply, for when the model's reply is refused."""
    lines = [*context.facts, *context.what_if]
    if not lines:
        return ("I could not give an answer I could check, and there is no design "
                "on this target yet to report on.")
    return FACTS_FALLBACK + "\n\n" + "\n".join(f"- {line}" for line in lines)
