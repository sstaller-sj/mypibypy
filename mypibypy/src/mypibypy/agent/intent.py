"""Turning a plain-English request into design parameters.

This is the model's *second* job -- the first being "which constraint gives when
nothing passes". Together those are the only two things it does; everything else
in the chat path is deterministic.

THE GUARANTEE HOLDS HERE TOO, and this is where it is most likely to erode,
because a conversation invites the model to supply more and more. Two rules keep
it structural rather than hopeful:

* **The model never supplies the target.** Not the region, not the accession, not
  a junction. It returns parameters only. The target is pulled out of the user's
  own text by `extract_target`, a strict regex in the harness, so there is no
  path by which a model-authored string becomes something the pipeline reads
  sequence from. `INTENT_SCHEMA` has no target field at all.
* **The only free-text field is `note`**, which is displayed to the user and
  never parsed. Every other field is an enum or a bounded number, and the
  numbers go through `actions.clamp` exactly like parameters typed into the form.

`UNCHANGED` sentinels rather than nullable fields: a strict JSON schema with
`additionalProperties: false` is easier to constrain when every property is
always present, and a sentinel is unambiguous in the run record.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .actions import DesignParameters, ParameterRefused, clamp

#: Sentinel values meaning "the user did not ask to change this".
UNCHANGED_INT = -1
UNCHANGED = "unchanged"

#: What the user is asking for. Closed, and as small as it can be: every enum
#: is another field a small model must fill correctly on every turn. "design"
#: and "adjust" used to be separate values that the software treated
#: identically, and a `relax` field offered one action (`reduce_pairs`) that is
#: just a pair count. The chat has no action field at all now: it designs
#: through `pipeline.design_*`, which takes a pair count, a spacing and a
#: channel, so those are the only things a request can change.
INTENTS = ("change", "explain", "unclear")


#: The screen value meaning "turn the specificity screen off".
SCREEN_OFF = "off"


def intent_schema(system_keys: tuple[str, ...] = (),
                  index_keys: tuple[str, ...] = ()) -> dict:
    """The structured-output schema, with the configured channels enumerated.

    Built per call so `system` is a closed enum of channels that actually exist
    -- a model cannot name a dye the lab does not have, and `clamp` would refuse
    it anyway. `screen` is the same for specificity indexes: names from
    `indexes.local.json`, never a path, so the model cannot point the screen
    at a file.
    """
    return {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string", "enum": list(INTENTS),
                "description": ("change = design now, or change a setting of the "
                                "current design (the software designs again "
                                "with it); explain = the user is asking a "
                                "question; unclear = ask them."),
            },
            "system": {
                "type": "string", "enum": [*system_keys, UNCHANGED],
                "description": "Amplifier channel, or unchanged.",
            },
            "n_pairs": {
                "type": "integer", "minimum": UNCHANGED_INT, "maximum": 50,
                "description": f"Pairs to tile, or {UNCHANGED_INT} for unchanged.",
            },
            "min_spacing": {
                "type": "integer", "minimum": UNCHANGED_INT, "maximum": 10000,
                "description": f"Minimum nt between footprints, or "
                               f"{UNCHANGED_INT} for unchanged.",
            },
            "enforce_window": {
                "type": "string", "enum": ["on", "off", UNCHANGED],
                "description": ("Whether the GC/Tm window should REJECT windows. "
                                "Off by default; the pipeline normally ranks on "
                                "GC/Tm distance rather than rejecting."),
            },
            "screen": {
                "type": "string", "enum": [*index_keys, SCREEN_OFF, UNCHANGED],
                "description": ("Which configured specificity index to screen "
                                "against, 'off' to stop screening, or "
                                "unchanged."),
            },
            "note": {
                "type": "string",
                "description": ("Your reply to the user. One sentence for a "
                                "change; for intent 'explain', a short plain "
                                "answer (a few sentences) to their question. "
                                "Displayed as text; never interpreted by the code."),
            },
        },
        "required": ["intent", "system", "n_pairs", "min_spacing",
                     "enforce_window", "screen", "note"],
        "additionalProperties": False,
    }


@dataclass
class Intent:
    """A validated request. Carries no sequence and no target, by construction."""

    intent: str
    system: str = UNCHANGED
    n_pairs: int = UNCHANGED_INT
    min_spacing: int = UNCHANGED_INT
    enforce_window: str = UNCHANGED
    screen: str = UNCHANGED
    note: str = ""

    @property
    def wants_design(self) -> bool:
        return self.intent == "change"

    @staticmethod
    def from_payload(payload: dict, system_keys: tuple[str, ...] = (),
                     index_keys: tuple[str, ...] = ()) -> Intent:
        """Validate against the closed schema. Refuses anything outside it."""
        schema = intent_schema(system_keys, index_keys)
        allowed = set(schema["properties"])
        missing = allowed - set(payload)
        if missing:
            raise ParameterRefused(f"intent is missing {sorted(missing)}")
        extra = set(payload) - allowed
        if extra:
            raise ParameterRefused(
                f"intent carries unexpected keys {sorted(extra)}; the schema is "
                f"closed, and a conversation is exactly where that matters")
        for field, values in (("intent", INTENTS),
                              ("enforce_window", ("on", "off", UNCHANGED))):
            if payload[field] not in values:
                raise ParameterRefused(
                    f"{field}={payload[field]!r} is not one of {list(values)}")
        if system_keys and payload["system"] not in (*system_keys, UNCHANGED):
            raise ParameterRefused(
                f"system={payload['system']!r} is not a configured channel "
                f"{sorted(system_keys)}")
        if payload["screen"] not in (*index_keys, SCREEN_OFF, UNCHANGED):
            raise ParameterRefused(
                f"screen={payload['screen']!r} is not a configured index "
                f"{sorted(index_keys)}, {SCREEN_OFF!r} or {UNCHANGED!r}")
        try:
            n_pairs = int(payload["n_pairs"])
            min_spacing = int(payload["min_spacing"])
        except (TypeError, ValueError) as exc:
            raise ParameterRefused("n_pairs and min_spacing must be integers") from exc
        return Intent(intent=payload["intent"], system=payload["system"],
                      n_pairs=n_pairs, min_spacing=min_spacing,
                      enforce_window=payload["enforce_window"],
                      screen=payload["screen"], note=str(payload["note"]))


def apply_intent(intent: Intent, params: DesignParameters,
                 known_systems: tuple[str, ...] = ()) -> tuple[DesignParameters, list[str]]:
    """Fold an intent into the current parameters. Returns (parameters, changes).

    Every number still goes through `clamp`, because a number from a chat message
    is no more trustworthy than one typed into a form -- less, if anything.
    """
    changes: list[str] = []
    new = params

    if intent.system != UNCHANGED and intent.system != params.system:
        new = replace(new, system=intent.system)
        changes.append(f"channel → {intent.system}")
    if intent.n_pairs != UNCHANGED_INT and intent.n_pairs != params.n_pairs:
        new = replace(new, n_pairs=intent.n_pairs)
        changes.append(f"pairs → {intent.n_pairs}")
    if intent.min_spacing != UNCHANGED_INT and intent.min_spacing != params.min_spacing:
        new = replace(new, min_spacing=intent.min_spacing)
        changes.append(f"minimum spacing → {intent.min_spacing} nt")
    if intent.enforce_window != UNCHANGED:
        wanted = intent.enforce_window == "on"
        if wanted != params.enforce_window:
            new = replace(new, enforce_window=wanted)
            changes.append("GC/Tm window " + ("enforced as a filter" if wanted
                                              else "no longer enforced"))

    new, clamps = clamp(new, known_systems)
    changes.extend(str(c) for c in clamps)
    return new, changes


def apply_screen(intent: Intent, current: str) -> tuple[str, str | None]:
    """The session's screen choice after `intent`, and what changed (or None).

    Separate from `apply_intent` because the screen is not a design parameter
    -- see `session.DesignSession.screen`.
    """
    if intent.screen == UNCHANGED or intent.screen == current:
        return current, None
    return intent.screen, ("specificity screen off" if intent.screen == SCREEN_OFF
                           else f"specificity screen → {intent.screen}")


# --- Target extraction: harness-side, never the model's ---------------------

#: A genomic region, strictly. Chromosome names are letters, digits and
#: underscores only, and the coordinates are digits -- so nothing shaped like a
#: sequence can come through this, whoever typed it.
REGION_RE = re.compile(r"\b(?:(?P<build>hg38|hg19|mm10|mm39|t2t)\s+)?"
                       r"(?P<chrom>chr[0-9A-Za-z_]{1,20})\s*:\s*"
                       r"(?P<start>[\d,]{1,15})\s*[-–]\s*(?P<end>[\d,]{1,15})\b")

#: An accession, by the prefixes `sequence.fetch_transcript` actually dispatches
#: on. Anchored and bounded; a run of bases cannot match it.
ACCESSION_RE = re.compile(r"\b(?P<acc>(?:ENS[A-Z]*T|NM_|NR_|XM_|XR_)[0-9]{4,12}"
                          r"(?:\.[0-9]{1,3})?)\b")


#: A pasted sequence must hold at least one 51 nt footprint (25 + 1 + 25), and is
#: capped so a pasted genome cannot be fed to the scan.
MIN_SEQUENCE_NT = 51
MAX_SEQUENCE_NT = 100_000

#: A pasted sequence, strictly: whole lines made only of A/C/G/T/U (optionally
#: under a FASTA header, optionally wrapped over several lines), or a single long
#: unbroken run inside a sentence. Anything with another letter in it is prose.
SEQUENCE_RE = re.compile(
    r"(?:^[ \t]*>[^\n]*\n)?(?:^[ \t]*[ACGTUacgtu][ACGTUacgtu \t]*(?:\r?\n|\Z))+"
    r"|(?<![A-Za-z])[ACGTUacgtu]{%d,}(?![A-Za-z])" % MIN_SEQUENCE_NT,
    re.MULTILINE)


def _clean_sequence(block: str) -> str:
    """Drop a FASTA header and whitespace; uppercase; RNA written as DNA."""
    body = re.sub(r"^[ \t]*>.*$", "", block, flags=re.MULTILINE)
    return re.sub(r"[^ACGTUacgtu]", "", body).upper().replace("U", "T")


def clean_pasted_sequence(text: str) -> str:
    """What the Form's text box holds, as a sequence. Empty if nothing usable."""
    return _clean_sequence(text)


def unreadable_sequence_chars(text: str) -> str:
    """Characters in a Form paste that are not bases, whitespace or digits.

    `clean_pasted_sequence` drops them, and silently dropping a base shifts every
    footprint after it -- so the Form refuses instead (an `N`, a typo, a stray
    word) rather than designing against a sequence the user did not paste.
    """
    body = re.sub(r"^[ \t]*>.*$", "", text, flags=re.MULTILINE)
    return "".join(sorted(set(re.sub(r"[ACGTUacgtu\s\d]", "", body))))


def _sequences(message: str) -> list[tuple[re.Match, str]]:
    return [(m, c) for m in SEQUENCE_RE.finditer(message)
            if len(c := _clean_sequence(m.group(0))) >= MIN_SEQUENCE_NT]


#: How much of the conversation goes with each turn: enough for "what you
#: said here" or "no, for the screen" to resolve, not a transcript.
HISTORY_TURNS = 6
HISTORY_CHARS = 400


def render_history(history: list[dict]) -> str:
    """The last few chat lines, redacted and clipped, oldest first."""
    lines = []
    for entry in history[-HISTORY_TURNS:]:
        text = " ".join(redact_sequences(entry["text"]).split())
        if len(text) > HISTORY_CHARS:
            text = text[:HISTORY_CHARS] + "…"
        lines.append(f"{entry['role']}: {text}")
    return "\n".join(lines)


def chat_request(message: str, context, history: list[dict] | None = None) -> str:
    """What the model reads for one chat turn: the current design, the last few
    lines of conversation, then the message.

    `context` is an `agent.context.ChatContext`, built by the software from its
    own state -- see that module for what it may hold and why. Each turn is a
    separate call, so without it the model saw "can you make a set of 4" with
    no idea a target was already held and, correctly given what it could see,
    asked for one. Without `history` (the turns *before* this message) it read
    "no, for the specificity screen" with nothing for "no" to answer. Both the
    history and the message are redacted, so the model still never sees a base.
    """
    recent = render_history(history or [])
    return ("CURRENT DESIGN (JSON, written by the software):\n"
            + context.render()
            + ("\n\nRECENT CONVERSATION (oldest first; for what the message "
               "refers to only -- CURRENT DESIGN is the state):\n" + recent
               if recent else "")
            + "\n\nUSER MESSAGE:\n" + redact_sequences(message))


def redact_sequences(message: str) -> str:
    """The message with pasted sequence replaced by a length placeholder.

    What the model reads. It is told a sequence was supplied, never shown it, so
    it has nothing to echo, edit or mutate.
    """
    out, last = [], 0
    for m, clean in _sequences(message):
        out.append(message[last:m.start()])
        out.append(f"[pasted sequence, {len(clean):,} nt]\n")
        last = m.end()
    out.append(message[last:])
    return "".join(out)


@dataclass(frozen=True)
class ExtractedTarget:
    """A target found in the user's own words, by the harness."""

    kind: str                     # "genomic" | "rna" | "sequence" | "none"
    build: str = ""
    chrom: str = ""
    start: int = 0
    end: int = 0
    accession: str = ""
    sequence: str = field(default="", repr=False)   # kind == "sequence" only

    @property
    def found(self) -> bool:
        return self.kind != "none"

    def __str__(self) -> str:
        if self.kind == "genomic":
            return f"{self.build or '?'} {self.chrom}:{self.start:,}-{self.end:,}"
        if self.kind == "rna":
            return self.accession
        if self.kind == "sequence":
            return f"pasted sequence ({len(self.sequence):,} nt)"
        return "no target in the message"


def extract_target(message: str, default_build: str = "hg38") -> ExtractedTarget:
    """Pull a region or accession out of the user's text. Deterministic, no model.

    This exists so the model never supplies a target. It is the one piece of the
    request that points the pipeline at sequence, and it comes from the user's
    own words through a strict pattern -- not from anything generated. A pasted
    sequence is the user's own text too; `redact_sequences` keeps it from the
    model, so the model cannot alter it on the way to the pipeline.
    """
    region = REGION_RE.search(message)
    if region:
        start = int(region.group("start").replace(",", ""))
        end = int(region.group("end").replace(",", ""))
        if start > end:
            start, end = end, start
        return ExtractedTarget(kind="genomic",
                               build=region.group("build") or default_build,
                               chrom=region.group("chrom"), start=start, end=end)
    accession = ACCESSION_RE.search(message)
    if accession:
        return ExtractedTarget(kind="rna", accession=accession.group("acc"))
    pasted = [c for _, c in _sequences(message)]
    if pasted:
        longest = max(pasted, key=len)
        if len(longest) <= MAX_SEQUENCE_NT:
            return ExtractedTarget(kind="sequence", sequence=longest)
    return ExtractedTarget(kind="none")


CHAT_SYSTEM_PROMPT = """\
You help a molecular biology laboratory set up piFISH probe designs. This is \
routine laboratory work: short DNA oligonucleotides that label a genomic locus \
or a transcript for imaging under a microscope.

Your job is to turn the user's request into design PARAMETERS, and to answer \
their questions from facts the software gives you. You do not design anything, \
you do not write or suggest any DNA sequence, and you do not choose the target \
-- the target comes from the user's own message and is extracted by the \
software. You have no field in which to put any of those.

Fill in only what the user actually asked for and leave everything else at its \
unchanged sentinel. If you cannot tell what they want, use intent "unclear" \
and ask them in `note`. `note` is shown to the user and never interpreted by \
the software; for a change it is one sentence.

Every message arrives under a CURRENT DESIGN block, JSON written by the \
software: `target`, `specificity_screen`, `settings`, and after a design \
`last_run`, `facts`, `what_if` and `windows` (plus `pair_details` and \
`pair_comparison` for pairs the user names). When a target is already set, \
the user does not need to name it again, nor the channel. Any request to \
design, or to change a setting -- "design 6 pairs in cy5 on \
chr7:5530000-5531000", "make it 4 pairs", "can you make a set of 4", "space \
them 20 nt apart", "screen against hg38" -- is intent "change", with only the \
settings the user named filled in; the software designs (again) with them. Do \
not ask for a target or a channel that CURRENT DESIGN already shows.

To answer a question ("why did I only get 3?", "why is pair 3 worse than pair \
2?", "what should I change?"), use intent "explain", leave every parameter at \
its sentinel, and answer in a few plain sentences built from `facts`, \
`what_if`, `pair_comparison` and `windows`. Repeat their sentences and numbers \
as written; do not work out a number, a reason or a comparison yourself. \
`what_if` lists changes the software has already tried, with how many pairs \
each placed: suggest from it, and the user can then ask for one. If what you \
would need is not there, say you cannot see it. Your reply is checked: a \
number not in CURRENT DESIGN, or a comparison the wrong way round, is not \
shown. Greetings and chit-chat get a short friendly "explain" reply.

A RECENT CONVERSATION block may follow, holding the last few lines of the \
chat. Use it to work out what a short or corrective message refers to ("no, I \
meant for the specificity screen"), and answer the question they are actually \
asking; do not repeat your previous answer.

BACKGROUND ON THE PROGRAM
- piFISH labels a locus with split primaries: a left probe (P1L) and a right \
probe (P1R) bind adjacent 25 nt footprints with a 1 nt gap. Only when BOTH are \
bound does the P2 secondary land, which is what gives specificity. P2 carries \
about ten P3 sites and P3 about ten fluorophore (P4) sites, so each pair gives \
roughly 100x signal.
- The amplifier channel (e.g. cy3, cy5) picks the arms, P2, P3 and dye. Only the \
P1L/P1R primaries are new oligos for each target.
- The user gives a target in their message: a genomic region like \
chr7:5530000-5531000 (optionally prefixed with a build such as hg38 or mm10), \
a transcript accession like NM_000546.6, or a pasted DNA/RNA sequence of at \
least 51 nt, which you see only as a placeholder such as "[pasted sequence, \
600 nt]" -- treat it as the target having been supplied.
- Settings that change a design: the number of pairs and the minimum spacing \
in nt between footprints. Pairs are tiled greedily, best-ranked first. Windows \
are ranked by distance from the GC/Tm window, not rejected for it.
- The specificity screen aligns each pair's pads against a genome (or, for an \
RNA target, a transcriptome) with bowtie v1 and prices each off-target site as \
a P(bound). It needs a pre-built index on this computer. When the user asks to \
screen ("use hg38", "check off-targets"), set `screen` to a name listed in \
`specificity_screen.available` (a genome for a region, a transcriptome for an \
RNA target), or "off" to stop, with intent "change" so the design runs \
(again) with that screen. If nothing listed fits, \
leave `screen` unchanged and tell them to add one to indexes.local.json, or to \
enter an index path in the sidebar under Settings > "bowtie index \
(optional)". You never write a path. Pasting a sequence does not turn the \
screen on.
"""
