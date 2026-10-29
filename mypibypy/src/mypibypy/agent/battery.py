"""Scripted chat turns that exercise the harness end to end, scored.

Ten things a user might type, run through the path `app.py` runs -- the context
the model is shown, the redaction, the schema, the validation, the reply check,
and the design again when a setting changed -- against a synthetic target, and
scored against the intent fields each turn should produce. It answers two
questions without touching lab data: is the harness working on this machine,
and can this model do the job. `scripts/benchmark_chat.py` is the command;
`scripts/validate_agent.py` does the same for the decision path.

NOTHING HERE PRINTS A SEQUENCE. The target is a seeded random sequence; the
report carries only its length, and the model sees only the placeholder it
sees in the app. Any run of bases that did turn up in a reply is scrubbed
before it is printed.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field

from ..loci import RnaTarget
from ..pipeline import design_rna
from ..probe_map import scored_windows
from ..session import DesignSession
from .actions import Action, DesignParameters, ParameterRefused
from .client import SAMPLING, OllamaDecisionClient
from .context import ScreenStatus, build_chat_context
from .intent import (
    CHAT_SYSTEM_PROMPT,
    UNCHANGED,
    UNCHANGED_INT,
    Intent,
    apply_intent,
    apply_screen,
    chat_request,
    extract_target,
    intent_schema,
)
from .loop import Budget, DesignAgent
from .reply_check import check_reply

SEQUENCE_NT = 600
SEED = 7
#: Index names the battery says are configured, so "screen against hg38" can be
#: scored on any machine. Nothing is screened here: there is no backend.
INDEX_KEYS = ("hg38", "gencode-v50")
INDEX_KINDS = ("genome", "transcriptome")

#: Anything that looks like oligo-length DNA, scrubbed from printed text.
DNA_RUN = re.compile(r"[ACGTUacgtu]{15,}")


def synthetic_sequence(n: int = SEQUENCE_NT, seed: int = SEED) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def scrub(text: str) -> str:
    return DNA_RUN.sub("[bases removed]", text)


@dataclass(frozen=True)
class Turn:
    """One message and the intent fields it should come back with.

    `expect` maps a schema field to the value wanted, or to a set of
    acceptable values. `apply` is False for the turns that name another
    target: they are scored only, so the battery stays on its synthetic
    sequence and fetches nothing.
    """

    message: str
    expect: dict
    apply: bool = True


def turns(system_keys: tuple[str, ...]) -> list[Turn]:
    """The battery, on the configured channels. `{placed}` is filled in at run
    time with the pairs the current design placed."""
    first = system_keys[0]
    other = system_keys[1] if len(system_keys) > 1 else first
    return [
        Turn(f"design 3 pairs in {other} for mm10 chr5:142906255-142906754",
             {"intent": "change", "n_pairs": 3,
              **({"system": other} if other != first else {})}, apply=False),
        Turn("give me 12 pairs at 20 nt spacing on chr7:5530000-5531000",
             {"intent": "change", "n_pairs": 12, "min_spacing": 20}, apply=False),
        Turn("hello", {"intent": {"explain", "unclear"}}),
        Turn("make it 6 pairs", {"intent": "change", "n_pairs": 6}),
        Turn("why did I only get {placed}?", {"intent": "explain"}),
        Turn("why is pair 2 worse than pair 1?", {"intent": "explain"}),
        Turn("screen against hg38", {"intent": "change", "screen": "hg38"}),
        # Not a setting the chat can change: anything but an invented change.
        Turn("too few passed, loosen the Tm",
             {"intent": {"change", "explain", "unclear"}, "n_pairs": UNCHANGED_INT,
              "min_spacing": UNCHANGED_INT, "system": UNCHANGED}),
        Turn("enforce the GC/Tm window", {"intent": "change", "enforce_window": "on"}),
        Turn("what's the Tm of pair 2?", {"intent": "explain"}),
    ]


@dataclass
class TurnResult:
    message: str
    seconds: float = 0.0
    error: str = ""                 # transport or decoding error, verbatim
    refused: str = ""               # the payload did not fit the closed schema
    note: str = ""
    #: The fields the model set, sentinels left out.
    fields: dict = field(default_factory=dict)
    #: (field, wanted, got, ok) per expectation.
    checks: list = field(default_factory=list)
    #: `reply_check` on an explain reply; None when the turn was not one.
    reply_problems: list[str] | None = None
    #: What the software changed on a change turn, in its own words.
    applied: list[str] = field(default_factory=list)
    designed: str = ""              # "6 pair(s) in 0.8 s" when it designed again

    @property
    def ok(self) -> bool:
        return (not self.error and not self.refused
                and all(ok for _, _, _, ok in self.checks))

    def lines(self) -> list[str]:
        out = [f"> {scrub(self.message)!r}  [{self.seconds:.1f} s]"]
        if self.error:
            out.append(f"    ERROR: {self.error}")
            return out
        if self.refused:
            out.append(f"    REFUSED: {self.refused}")
            return out
        out.append("    set: " + (" ".join(f"{k}={v}" for k, v in self.fields.items())
                                  or "nothing"))
        out.append(f"    note: {scrub(' '.join(self.note.split()))[:240]}")
        verdict = " ".join(f"{k}={'OK' if ok else f'WRONG({got!r})'}"
                           for k, _, got, ok in self.checks)
        out.append(f"    expected: {verdict} => {'PASS' if self.ok else 'FAIL'}")
        if self.reply_problems is not None:
            out.append("    reply check: " + ("clean" if not self.reply_problems
                                              else "; ".join(self.reply_problems)))
        if self.applied:
            out.append("    applied: " + "; ".join(self.applied)
                       + (f"; designed again: {self.designed}" if self.designed else ""))
        return out

    def to_dict(self) -> dict:
        return {"message": scrub(self.message), "seconds": round(self.seconds, 2),
                "error": self.error, "refused": self.refused,
                "note": scrub(self.note), "fields": self.fields,
                "checks": [{"field": k, "wanted": sorted(w) if isinstance(w, (set, frozenset)) else w,
                            "got": g, "ok": ok} for k, w, g, ok in self.checks],
                "ok": self.ok, "reply_problems": self.reply_problems,
                "applied": self.applied, "designed": self.designed}


@dataclass
class BatteryReport:
    label: str
    model: str
    sequence_nt: int
    placed: int                     # pairs the opening design placed
    design_seconds: float
    turns: list[TurnResult] = field(default_factory=list)
    designs: int = 1                # designs made, the opening one included

    @property
    def total(self) -> int:
        return len(self.turns)

    @property
    def passed(self) -> int:
        return sum(t.ok for t in self.turns)

    @property
    def model_seconds(self) -> float:
        return sum(t.seconds for t in self.turns)

    @property
    def unreachable(self) -> bool:
        """Every turn failed before the model answered: nothing was measured."""
        return bool(self.turns) and all(t.error for t in self.turns)

    @property
    def reply_check_flags(self) -> int:
        return sum(1 for t in self.turns if t.reply_problems)

    def lines(self) -> list[str]:
        out = [f"== chat battery: {self.label} ==",
               f"synthetic target {self.sequence_nt:,} nt; opening design placed "
               f"{self.placed} pair(s) in {self.design_seconds:.1f} s"]
        for t in self.turns:
            out.append("")
            out.extend(t.lines())
        per_turn = self.model_seconds / self.total if self.total else 0.0
        out.append("")
        out.append(f"==> {self.passed}/{self.total} turns as expected; model time "
                   f"{self.model_seconds:.0f} s total, {per_turn:.1f} s/turn; "
                   f"{self.reply_check_flags} explain repl"
                   f"{'y' if self.reply_check_flags == 1 else 'ies'} flagged by the "
                   f"reply check; {self.designs} design(s) made")
        return out

    def to_dict(self) -> dict:
        return {"label": self.label, "model": self.model,
                "sequence_nt": self.sequence_nt, "placed": self.placed,
                "design_seconds": round(self.design_seconds, 2),
                "passed": self.passed, "total": self.total,
                "model_seconds": round(self.model_seconds, 1),
                "reply_check_flags": self.reply_check_flags, "designs": self.designs,
                "turns": [t.to_dict() for t in self.turns]}


def _design(s: DesignSession, params: DesignParameters, seq: str):
    """The chat's pasted-sequence design, as `app.design_for_target` runs it,
    with no screen. Returns (run, seconds)."""
    started = time.time()
    s.update(params=params)
    result = design_rna(RnaTarget("target", "GENE", "pasted-sequence", params.system),
                        backend=None, screen=False, transcript_seq=seq,
                        n_pairs=params.n_pairs, min_spacing=params.min_spacing)
    run = s.record(source="chat", results=[result], note="mypibypy chat battery",
                   target_label=str(s.target), mode="chat", screen="off")
    return run, time.time() - started


def _screen_status(s: DesignSession, index_keys) -> ScreenStatus:
    return ScreenStatus(on=False, against=s.screen or "off",
                        available=[f"{k} ({kind})"
                                   for k, kind in zip(index_keys, INDEX_KINDS)])


def run_turn(client, s: DesignSession, turn: Turn, history: list[dict],
             system_keys: tuple[str, ...], index_keys=INDEX_KEYS,
             seq: str = "") -> TurnResult:
    """One chat turn, as `app.chat_editor` runs it, without Streamlit."""
    run = s.current
    message = turn.message.format(
        placed=sum(r.n_pairs for r in run.results) if run else 0)
    result = TurnResult(message=message)

    found = extract_target(message)
    current = found if found.found else s.target
    on_target = (run if run and current and run.target_label == str(current)
                 else None)
    context = build_chat_context(
        message, s.params, str(current) if current else "", on_target,
        screen=_screen_status(s, index_keys),
        windows=(scored_windows(on_target.results[0])
                 if on_target and len(on_target.results) == 1 else None))
    request = chat_request(message, context, history)
    history.append({"role": "user", "text": message})

    payload, error, seconds = client.structured(
        CHAT_SYSTEM_PROMPT, request, intent_schema(system_keys, index_keys))
    result.seconds = seconds
    if error or payload is None:
        result.error = error or "no payload"
        history.append({"role": "assistant", "text": "(no reply)"})
        return result
    try:
        intent = Intent.from_payload(payload, system_keys, index_keys)
    except ParameterRefused as exc:
        result.refused = str(exc)
        history.append({"role": "assistant", "text": "(refused)"})
        return result

    result.note = intent.note
    result.fields = {k: v for k, v in payload.items()
                     if k != "note" and v not in (UNCHANGED, UNCHANGED_INT)}
    for key, want in turn.expect.items():
        got = payload.get(key)
        ok = got in want if isinstance(want, (set, frozenset)) else got == want
        result.checks.append((key, want, got, ok))
    if intent.intent == "explain":
        result.reply_problems = check_reply(
            intent.note, CHAT_SYSTEM_PROMPT + "\n" + request,
            enforce_window=s.params.enforce_window)

    if intent.wants_design and turn.apply:
        new_params, changes = apply_intent(intent, s.params, system_keys)
        s.screen, screen_change = apply_screen(intent, s.screen)
        if screen_change:
            changes.append(screen_change + " (nothing is screened in the battery)")
        result.applied = changes
        if new_params != s.params:
            # As the app: a setting the design uses moved, so design again.
            run, took = _design(s, new_params, seq)
            result.designed = f"{sum(r.n_pairs for r in run.results)} pair(s) in {took:.1f} s"
    history.append({"role": "assistant", "text": intent.note})
    return result


def run_battery(client, *, system_keys: tuple[str, ...], label: str = "",
                n_pairs: int = 4, min_spacing: int = 50, seed: int = SEED,
                sequence_nt: int = SEQUENCE_NT, index_keys=INDEX_KEYS,
                battery: list[Turn] | None = None) -> BatteryReport:
    """Design the synthetic target, then run every turn against that session."""
    seq = synthetic_sequence(sequence_nt, seed)
    s = DesignSession(params=DesignParameters(system=system_keys[0], n_pairs=n_pairs,
                                              min_spacing=min_spacing))
    s.target = extract_target(f"design for {seq} in {system_keys[0]}")
    run, took = _design(s, s.params, seq)
    report = BatteryReport(label=label or getattr(client, "name", "model"),
                           model=getattr(client, "name", "?"),
                           sequence_nt=len(seq),
                           placed=sum(r.n_pairs for r in run.results),
                           design_seconds=took)
    history: list[dict] = []
    for turn in battery if battery is not None else turns(system_keys):
        result = run_turn(client, s, turn, history, system_keys, index_keys, seq)
        report.turns.append(result)
    report.designs = len(s.runs)
    return report


# --- the Form's agent loop over the same target --------------------------------

@dataclass
class LoopDemo:
    status: str
    message: str
    model_calls: int
    seconds: float
    events: list[str]

    def lines(self) -> list[str]:
        out = [f"== agent loop over the synthetic design ==",
               f"status {self.status} in {self.seconds:.1f} s; model calls "
               f"{self.model_calls}", f"message: {scrub(self.message)}"]
        out.extend(f"  - {scrub(e)}" for e in self.events)
        return out

    def to_dict(self) -> dict:
        return {"status": self.status, "message": scrub(self.message),
                "model_calls": self.model_calls, "seconds": round(self.seconds, 2),
                "events": [scrub(e) for e in self.events]}


def run_loop(client, *, system_keys: tuple[str, ...], seq: str | None = None,
             n_pairs: int = 4, min_spacing: int = 50,
             enforce_window: bool = False) -> LoopDemo:
    """The Form's agent loop, as `app.run_agent_loop` drives it: the pipeline
    really runs again when the loop changes the pair count or channel, and
    only those two actions are offered. With no screen, the classifier flags
    the design itself and the model is never called -- which is the point."""
    seq = seq or synthetic_sequence()
    params = DesignParameters(system=system_keys[0], n_pairs=n_pairs,
                              min_spacing=min_spacing, enforce_window=enforce_window)

    def design_fn(p: DesignParameters):
        result = design_rna(RnaTarget("target", "GENE", "pasted-sequence", p.system),
                            backend=None, screen=False, transcript_seq=seq,
                            n_pairs=p.n_pairs, min_spacing=p.min_spacing)
        return result, scored_windows(result)

    agent = DesignAgent(design_fn, client=client,
                        budget=Budget(max_turns=4, max_relaxations=3),
                        known_systems=tuple(system_keys),
                        allowed_actions=(Action.REDUCE_PAIRS, Action.SWITCH_SYSTEM))
    started = time.time()
    out = agent.run(params, target_label="synthetic sequence", request="chat battery")
    kinds = {"model_decision", "deterministic_decision", "budget", "error",
             "refusal", "outcome"}
    return LoopDemo(out.status, out.message, out.record.model_calls,
                    time.time() - started,
                    [f"[{e.kind}] {e.summary}" for e in out.record.events
                     if e.kind in kinds])


# --- sampler A/B ---------------------------------------------------------------

class ModelfileSamplerClient(OllamaDecisionClient):
    """The same client with `client.SAMPLING` left out, so the sampler is
    whatever the model's Modelfile says -- what the harness ran with before the
    sampler was pinned. For comparing the two, nothing else."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name += " (modelfile sampler)"

    def request_body(self, system: str, evidence: str,
                     schema: dict | None = None) -> dict:
        body = super().request_body(system, evidence, schema)
        body["options"] = {k: v for k, v in body["options"].items()
                           if k not in SAMPLING or k == "temperature"}
        return body
