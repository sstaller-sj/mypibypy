"""The kinds of question the chat answers, as a closed list.

A small local model is reliable at one thing in a conversation: picking which
of a few things was asked. It is not reliable at composing the answer -- handed
two numbers it called 70.4 "higher" than 73.8 -- so the answer to each of these
is written by the software (`agent.answers`) from the design's own data, and
the model's part is choosing the topic.

This module holds only the list, so both `intent` (the schema and the model's
guide) and `answers` (the templates) can import it without importing each
other. `tests/test_answers.py` checks that every topic here has a template.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The topic of a turn that is not a question (a change, or unclear).
NONE = "none"
#: A question about the design that no topic fits. The model answers it in
#: `note` from the facts it was given, and the reply is checked (`reply_check`).
OTHER = "other"


@dataclass(frozen=True)
class Topic:
    key: str
    group: str        # "design" | "pair" | "what_if" | "method" | "tool"
    ask: str          # what the user is asking, as the model's guide says it
    example: str


TOPICS: tuple[Topic, ...] = (
    Topic("placement", "design",
          "why fewer pairs were placed than requested, or what limited the design",
          "why did I only get 3?"),
    Topic("screen", "design",
          "whether the pairs were screened and against what, what off-targets "
          "were found, or why no screen ran",
          "did you check off-targets?"),
    Topic("folding", "design",
          "the probes' own self-folding",
          "is partly folded a problem?"),
    Topic("accessibility", "design",
          "target-site accessibility (RNA structure at the binding site)",
          "is the site open?"),
    Topic("qc_flags", "design",
          "QC flags on the pairs, or what a flag means",
          "what does homopolymer5 mean?"),
    Topic("repeats", "design",
          "repeat masking of the target",
          "what was masked?"),
    Topic("summary", "design",
          "how the design came out overall, or whether it is ready to order",
          "is this design ok?"),
    Topic("pair_detail", "pair",
          "ONE named pair: its Tm, GC, position, verdict, folding or flags",
          "what's the Tm of pair 2?"),
    Topic("compare_pairs", "pair",
          "how TWO named pairs differ, or why one ranks below the other",
          "why is pair 2 worse than pair 1?"),
    Topic("best_worst", "pair",
          "which pair is the best or the weakest, with no pair named",
          "which pair is the weakest?"),
    Topic("what_if", "what_if",
          "what a different pair count or spacing WOULD place, asked as a "
          "question (\"would N fit\", \"how many would I get if ...\") rather "
          "than a request; put the number asked about in n_pairs or "
          "min_spacing",
          "could I fit 10 pairs?"),
    Topic("capacity", "what_if",
          "the most pairs the target can hold",
          "how many pairs can this hold?"),
    Topic("get_more", "what_if",
          "how to get more pairs, or what to change",
          "how do I get more pairs?"),
    Topic("about_pifish", "method",
          "what piFISH split probes are and how the amplification works",
          "how does this work?"),
    Topic("about_screening", "method",
          "how the specificity screen works, or what NOT SCREENED means",
          "why doesn't a half-hit matter?"),
    Topic("glossary", "method",
          "what a number or term means: penalty, P(bound), Tm, the GC/Tm "
          "window, spacing, footprint, verdict",
          "what is penalty?"),
    Topic("target_types", "method",
          "the kinds of target (genomic region, RNA transcript, pasted "
          "sequence, fusion junction), whether one can be designed, and how "
          "they differ",
          "genomic versus RNA?"),
    Topic("probe_map", "method",
          "how to read the probe map or a pair's close-up",
          "what are the grey rows?"),
    Topic("how_order", "tool",
          "how to order or download the oligos",
          "how do I get the oligos?"),
    Topic("how_setup", "tool",
          "how to add or set up a target, a specificity index, or an "
          "amplifier channel (dye, fluorophore), or how to screen another "
          "genome",
          "how do I screen mouse?"),
    Topic("greeting", "tool",
          "a greeting, thanks, or asking what the chat can do",
          "hello"),
    Topic("out_of_scope", "tool",
          "anything that is not about this design or this tool",
          "which antibody should I use?"),
)

TOPIC_KEYS: tuple[str, ...] = tuple(t.key for t in TOPICS)
#: Every value the schema's `topic` field may take.
ALL_KEYS: tuple[str, ...] = (*TOPIC_KEYS, OTHER, NONE)


def guide() -> str:
    """The topic list as the model reads it, one line per topic."""
    lines = [f'- "{t.key}": {t.ask}. e.g. "{t.example}"' for t in TOPICS]
    lines.append(f'- "{OTHER}": a question about this design that none of the '
                 f"above fits. Only then do you answer it yourself, in `note`.")
    lines.append(f'- "{NONE}": the message is not a question (intent "change" '
                 f'or "unclear").')
    return "\n".join(lines)
