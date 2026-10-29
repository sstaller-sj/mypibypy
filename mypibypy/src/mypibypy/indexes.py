"""Named specificity-screen indexes, loaded from a local file.

The screen needs a bowtie v1 index on this machine, and which one is a choice
made per design: hg38 or mm10 for a locus, a transcriptome for an RNA target.
Typing a path for every design is error-prone, and letting the chat model type
one is worse -- a misread "mm10" that became "/data/genomes/hg38" would screen
against the wrong genome and still report the result as *screened*. So the
indexes are named once here, and both the Form and the chat choose among the
names. The model gets a closed enum of them; it never sees or writes a path.

Resolution order, the same shape as `systems` and `loci`:

1. An explicit path passed to `load_indexes()`.
2. ``$MYPIBYPY_INDEXES``.
3. ``indexes.local.json`` in the current directory or any parent (gitignored).

No file is fine: the screen is then set by path (the app's sidebar, or
``$MYPIBYPY_BOWTIE_INDEX``) or not at all. Format:

    {"indexes": {
        "hg38":    {"path": "/data/genomes/hg38", "kind": "genome",
                    "description": "human GRCh38"},
        "mm10-tx": {"path": "/data/tx/mm10_gencode", "kind": "transcriptome"}
    }}

`path` is the bowtie v1 index *basename* -- the part before ``.1.ebwt``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

LOCAL_FILENAME = "indexes.local.json"
ENV_VAR = "MYPIBYPY_INDEXES"
KINDS = ("genome", "transcriptome")

#: The choice that turns the screen off for a design, whatever is configured.
OFF = "off"
#: The choice that defers to the path set outside the session (sidebar / env).
DEFAULT = ""

#: Index names become enum values the model may return, so they are kept to
#: plain identifiers: nothing a path, a sentence or a sequence could hide in.
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,31}$")

#: What bowtie-build writes for a small and a large index.
_BOWTIE_SUFFIXES = (".1.ebwt", ".1.ebwtl")


class IndexesNotConfigured(RuntimeError):
    """The indexes file exists but cannot be used."""


@dataclass(frozen=True)
class ScreenIndex:
    name: str
    path: str
    kind: str = "genome"
    description: str = ""

    @property
    def problem(self) -> str | None:
        """Why this index cannot be used, or None. Checked on disk."""
        return bowtie_index_problem(self.path)

    def __str__(self) -> str:
        return f"{self.name} ({self.kind}" + (f", {self.description})"
                                              if self.description else ")")


def bowtie_index_problem(path: str) -> str | None:
    """None if `path` is a bowtie v1 index basename with its files present."""
    base = Path(path).expanduser()
    if any(Path(str(base) + s).is_file() for s in _BOWTIE_SUFFIXES):
        return None
    return (f"no bowtie v1 index at {base} (expected {base.name}.1.ebwt next "
            f"to it; build one with bowtie-build)")


def kind_mismatch(index: ScreenIndex, wanted: str) -> str | None:
    """Why `index` must not screen a target that needs a `wanted` index, or None.

    `wanted` is "transcriptome" for an RNA target and "genome" for a locus or a
    fusion junction. One source for the sentence, so the Form, the chat and the
    MCP server refuse (or warn about) the same mismatch in the same words.
    """
    if index.kind == wanted:
        return None
    if wanted == "transcriptome":
        return (f"**{index.name}** is a {index.kind} index. A footprint that spans "
                f"a splice junction does not map contiguously to a genome, so it "
                f"can look falsely clean.")
    return (f"**{index.name}** is a {index.kind} index. A genomic footprint can "
            f"have off-target sites in introns and intergenic DNA, which a "
            f"transcriptome does not contain, so it can look falsely clean.")


def _search_upward(filename: str, start: Path | None = None) -> Path | None:
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        candidate = d / filename
        if candidate.is_file():
            return candidate
    return None


def resolve_indexes_path(path: str | Path | None = None,
                         start: Path | None = None) -> Path | None:
    """Where indexes would be loaded from, or None if there is no file."""
    if path:
        return Path(path).expanduser()
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise IndexesNotConfigured(f"${ENV_VAR} points at a missing file: {p}")
        return p
    return _search_upward(LOCAL_FILENAME, start)


def load_indexes(path: str | Path | None = None,
                 start: Path | None = None) -> dict[str, ScreenIndex]:
    """The named indexes, in file order. Empty if no file is configured.

    A malformed entry is refused, not skipped: a silently missing index would
    turn "screen against mm10" into no screen at all.
    """
    src = resolve_indexes_path(path, start)
    if src is None:
        return {}
    try:
        raw = json.loads(src.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexesNotConfigured(f"{src} could not be read: {exc}") from exc
    entries = raw.get("indexes")
    if not isinstance(entries, dict):
        raise IndexesNotConfigured(f"{src} has no 'indexes' object")
    out: dict[str, ScreenIndex] = {}
    for name, d in entries.items():
        if not _NAME_RE.match(name) or name == OFF:
            raise IndexesNotConfigured(
                f"{src}: index name {name!r} must be a short identifier "
                f"(letters, digits, . _ -) and not {OFF!r}")
        if not isinstance(d, dict) or not isinstance(d.get("path"), str) \
                or not d["path"].strip():
            raise IndexesNotConfigured(f"{src}: index {name!r} needs a 'path'")
        kind = d.get("kind", "genome")
        if kind not in KINDS:
            raise IndexesNotConfigured(
                f"{src}: index {name!r} kind {kind!r} is not one of {list(KINDS)}")
        out[name] = ScreenIndex(name=name, path=d["path"].strip(), kind=kind,
                                description=str(d.get("description", "")))
    return out


def resolve_screen(choice: str, indexes: dict[str, ScreenIndex],
                   fallback_path: str = "") -> tuple[str, str]:
    """(bowtie basename to screen against, label for it). Empty path = no screen.

    `choice` is a session's screen setting: a configured name, `OFF`, or
    `DEFAULT`, which uses `fallback_path` (the sidebar box), then
    ``$MYPIBYPY_BOWTIE_INDEX``. A name that is no longer configured screens
    nothing rather than falling back -- the user asked for that index, not
    for whichever one happens to be the default.
    """
    from .specificity import ENV_BOWTIE_INDEX

    if choice == OFF:
        return "", "off"
    if choice in indexes:
        return indexes[choice].path, choice
    if choice != DEFAULT:
        return "", f"{choice} (not configured)"
    path = fallback_path.strip() or os.environ.get(ENV_BOWTIE_INDEX, "").strip()
    return path, (path or "off")
