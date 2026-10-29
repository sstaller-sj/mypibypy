"""The MCP server's design store: what a design *was*, never what it reads.

`mcp_server.Session` keeps every design it makes in memory, results and all. An
MCP client restarts the server often -- each new conversation, each launch of
the app -- and a design made in an earlier conversation is one a user may want
to order now. So each design's description is saved here as it changes: the
target id, the `DesignParameters`, the screen flag, the pad assignment, the run
id, the warnings, the stop reason, and the run's position in the state machine
(state, turn, relaxations). The results are not. The pipeline is deterministic,
so a later server rebuilds them by running it again for the same target and
parameters, and proves it got the same design back against the digest saved
with the entry.

That digest is the one thing here derived from the oligos. It is a SHA-256 over
each pair's name, footprint, both primaries and screen verdict; it cannot be
read or turned back into a sequence, and it exists so that a loci entry, a
reference or a screen configuration that changed between conversations refuses
the reload instead of quietly attaching an earlier conversation's acceptance to
a different design.

Append-only JSONL, like `agent.record`: a later line for the same design id
supersedes an earlier one, so a run's history of states is kept and nothing is
rewritten. The default file is gitignored (`*.local.jsonl`): it names the lab's
targets, though it carries no sequence, and `test_mcp_server.py` checks that.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .agent import record as rec

LOCAL_FILENAME = "mcp_designs.local.jsonl"
ENV_VAR = "MYPIBYPY_MCP_DESIGNS"


def resolve_store_path(path: str | Path | None = None) -> Path:
    """`path`, else ``$MYPIBYPY_MCP_DESIGNS``, else beside the run record.

    "Beside the run record" rather than beside the package: the two files
    describe the same runs, and wherever the run record was sent (the checkout
    by `configure_environment`, a temporary directory by the tests) the design
    store belongs with it.
    """
    if path:
        return Path(path).expanduser()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return rec.resolve_store_path().parent / LOCAL_FILENAME


def fingerprint(results) -> str:
    """A digest of what a design would order and how each pair screened.

    Over every pair of every result: its two names, its footprint, both
    primaries and the screen verdict (`pass` / `review` / `fail` /
    `not screened`). The verdict is in it on purpose: a design accepted as
    unscreened and one that now screens as failing are not the same evidence,
    even on the same oligos, and the saved state belongs to the first.
    """
    h = hashlib.sha256()
    for res in results:
        for pair, screen, _ in res.rows():
            h.update("\t".join((pair.p1l_name, pair.p1r_name,
                                str(pair.candidate.interval),
                                pair.p1l, pair.p1r, screen.verdict)).encode())
            h.update(b"\n")
    return h.hexdigest()


class DesignStore:
    """Append-only store of design descriptions. One JSON object per line."""

    def __init__(self, path: str | Path | None = None):
        self.path = resolve_store_path(path)

    def append(self, entry: dict) -> dict:
        """Write `entry` with a timestamp; returns what was written."""
        line = dict(entry, saved_at=datetime.now(timezone.utc)
                    .isoformat(timespec="seconds"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
        return line

    def all(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{self.path}: unreadable design entry: {exc}") from exc
            if not isinstance(entry, dict) or "design_id" not in entry:
                raise ValueError(f"{self.path}: a line is not a design entry")
            out.append(entry)
        return out

    def get(self, design_id: str) -> dict | None:
        # Later writes supersede earlier ones for the same id, so scan backwards.
        for entry in reversed(self.all()):
            if entry["design_id"] == design_id:
                return entry
        return None

    def ids(self) -> list[str]:
        """Every design id in the store, once each, in the order first saved."""
        return list(dict.fromkeys(e["design_id"] for e in self.all()))
