"""The durable run record: what the loop did, on what evidence, and why.

A probe ordered today gets questioned in six months. "The agent chose it" is not
an answer; "turn 3 relaxed the Tm window from 55-62 to 52-65 because only 7 of
250 windows survived, and here is the screen verdict for the pair that came out"
is. So every tool call, every clamp, and every model decision is appended here
with the evidence it was made on.

Append-only JSONL, for the same reasons as `outcomes`: appending never rewrites
earlier lines, a corrupt line costs one event rather than the file, and it stays
readable by hand years from now. The default file is gitignored -- a run record
quotes designed sequences.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOCAL_FILENAME = "agent_runs.local.jsonl"
ENV_VAR = "MYPIBYPY_AGENT_RUNS"

#: Event kinds. A closed set so a reader can rely on them.
TOOL_CALL = "tool_call"
MODEL_DECISION = "model_decision"
DETERMINISTIC_DECISION = "deterministic_decision"
PARAMETER_CLAMP = "parameter_clamp"
BUDGET = "budget"
REFUSAL = "refusal"
ERROR = "error"
OUTCOME = "outcome"
TRANSITION = "transition"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Event:
    """One thing that happened, with enough context to explain it later."""

    kind: str
    turn: int
    summary: str
    detail: dict = field(default_factory=dict)
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunRecord:
    """Everything about one agent run.

    `events` is the audit trail. `parameters_history` is kept separately
    because "what were the parameters at turn 3" is the question most often
    asked of a record, and reconstructing it by replaying events is work a
    reader should not have to do.
    """

    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    target: str = ""
    request: str = ""
    model: str = ""
    started_at: str = field(default_factory=_now)
    finished_at: str = ""
    status: str = "running"          # running | succeeded | failed | flagged
    events: list[Event] = field(default_factory=list)
    parameters_history: list[dict] = field(default_factory=list)
    turns_used: int = 0
    relaxations_used: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # -- recording ---------------------------------------------------------
    def log(self, kind: str, turn: int, summary: str, **detail) -> Event:
        event = Event(kind=kind, turn=turn, summary=summary, detail=detail)
        self.events.append(event)
        return event

    def log_parameters(self, turn: int, params, note: str = "") -> None:
        self.parameters_history.append(
            {"turn": turn, "note": note, **params.to_dict()})

    # -- reading -----------------------------------------------------------
    def of_kind(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    @property
    def decisions(self) -> list[Event]:
        """Every decision, model-made or deterministic, in order."""
        return [e for e in self.events
                if e.kind in (MODEL_DECISION, DETERMINISTIC_DECISION)]

    @property
    def consulted_model(self) -> bool:
        """Did this run need the model at all?

        Worth asking directly: most runs should not. The deterministic path
        handles clear cases, and a run that reached the model on a
        straightforward design is a sign the classifier has drifted.
        """
        return self.model_calls > 0

    def explain(self) -> str:
        """The run as prose, for a human asking why this probe.

        Deliberately plain and chronological. This is what gets pasted into a
        lab notebook or an email six months later.
        """
        lines = [
            f"Run {self.run_id} on {self.target or '(unnamed target)'}: "
            f"{self.status}, {self.turns_used} turn(s), "
            f"{self.relaxations_used} relaxation(s), "
            f"{self.model_calls} model call(s)."
        ]
        if self.request:
            lines.append(f"  Request: {self.request}")
        for event in self.events:
            lines.append(f"  [turn {event.turn}] {event.kind}: {event.summary}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "target": self.target,
            "request": self.request,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "turns_used": self.turns_used,
            "relaxations_used": self.relaxations_used,
            "model_calls": self.model_calls,
            "consulted_model": self.consulted_model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "parameters_history": self.parameters_history,
            "events": [e.to_dict() for e in self.events],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> RunRecord:
        record = RunRecord(
            run_id=d.get("run_id", ""), target=d.get("target", ""),
            request=d.get("request", ""), model=d.get("model", ""),
            started_at=d.get("started_at", ""), finished_at=d.get("finished_at", ""),
            status=d.get("status", "unknown"),
            parameters_history=list(d.get("parameters_history", [])),
            turns_used=int(d.get("turns_used", 0)),
            relaxations_used=int(d.get("relaxations_used", 0)),
            model_calls=int(d.get("model_calls", 0)),
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
        )
        record.events = [Event(**e) for e in d.get("events", [])]
        return record


def resolve_store_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.cwd() / LOCAL_FILENAME


class RunStore:
    """Append-only store of `RunRecord`s. One JSON object per line."""

    def __init__(self, path: str | Path | None = None):
        self.path = resolve_store_path(path)

    def append(self, record: RunRecord) -> RunRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(record.to_json() + "\n")
        return record

    def all(self) -> list[RunRecord]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                out.append(RunRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{self.path}: unreadable run record: {exc}") from exc
        return out

    def get(self, run_id: str) -> RunRecord | None:
        # Later writes supersede earlier ones for the same id, so scan backwards.
        for record in reversed(self.all()):
            if record.run_id == run_id:
                return record
        return None
