"""The one design state both UI editors share.

The Form and the Chat used to keep separate state -- the form its last run, the
chat its own parameters, target and results -- so a tweak in one was invisible
to the other and there were two download buttons for two different things.
`DesignSession` is now the single source: either editor changes `params` and
`target`, every design is recorded as a `Run`, and the design panel shows
whichever run is selected.

No Streamlit here, so it can be tested directly. The app keeps one instance in
`st.session_state`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

from .agent import DesignParameters
from .agent.intent import ExtractedTarget

#: Parameters the panel reports as changed between runs. The rest of
#: `DesignParameters` (GC/Tm window, mismatches, ...) is shown only when it
#: differs, to keep the summary short.
HEADLINE = ("system", "n_pairs", "min_spacing")


@dataclass
class Run:
    """One press of Design, or one chat turn that designed. Kept, not replaced."""

    number: int
    source: str                     # "form" | "chat"
    results: list                   # list[DesignResult]; two for a fusion
    note: str                       # goes into the workbook
    params: DesignParameters
    target_label: str
    mode: str = ""                  # the form's target type, or "chat"
    agent: object | None = None     # the agent's RunOutcome, when the loop ran
    screen: str = ""                # what it was screened against, or "off"

    @property
    def label(self) -> str:
        pairs = sum(r.n_pairs for r in self.results)
        return (f"#{self.number} · {self.source} · {self.target_label} · "
                f"{pairs} pair{'s' if pairs != 1 else ''}")


@dataclass
class DesignSession:
    params: DesignParameters = field(default_factory=DesignParameters)
    target: ExtractedTarget | None = None
    runs: list[Run] = field(default_factory=list)
    selected: int = -1              # index into `runs`; -1 is the latest
    #: The specificity-screen choice: a name from `indexes`, `indexes.OFF`,
    #: or `indexes.DEFAULT` (the sidebar path / $MYPIBYPY_BOWTIE_INDEX). Not a
    #: `DesignParameters` field: the agent loop may vary those, and which
    #: genome to screen against is never the loop's to change.
    screen: str = ""
    #: Bumped whenever something other than the form changes `params` or
    #: `target`. The form keys its widgets on it, so they re-read their
    #: defaults from the session instead of keeping stale values.
    revision: int = 0

    # --- editing ------------------------------------------------------------

    def update(self, params: DesignParameters | None = None,
               target: ExtractedTarget | None = None, *,
               from_form: bool = False) -> None:
        """Change the shared parameters and/or target.

        The form passes `from_form=True`: its widgets already hold the new
        values, and resetting them would discard what the user is typing.
        """
        if params is not None:
            self.params = params
        if target is not None:
            self.target = target
        if not from_form:
            self.revision += 1

    def record(self, *, source: str, results: list, note: str,
               target_label: str, mode: str = "", agent=None,
               screen: str = "") -> Run:
        """Keep a finished design and select it.

        `screen` is the label of what was actually screened against, which
        the caller resolved -- not the session's choice, which may be DEFAULT.
        """
        run = Run(number=len(self.runs) + 1, source=source, results=results,
                  note=note, params=replace(self.params),
                  target_label=target_label, mode=mode, agent=agent,
                  screen=screen)
        self.runs.append(run)
        self.selected = -1
        return run

    # --- reading ------------------------------------------------------------

    @property
    def current(self) -> Run | None:
        if not self.runs:
            return None
        index = self.selected if -len(self.runs) <= self.selected < len(self.runs) else -1
        return self.runs[index]

    def previous(self, run: Run) -> Run | None:
        """The run before `run` on the same target, for showing what changed."""
        for earlier in reversed(self.runs[:run.number - 1]):
            if earlier.target_label == run.target_label:
                return earlier
        return None

    def select(self, number: int) -> None:
        self.selected = number - 1 if number < len(self.runs) else -1


def changed_params(before: DesignParameters | None,
                   after: DesignParameters) -> dict[str, tuple]:
    """{name: (old, new)} for every parameter that differs."""
    if before is None:
        return {}
    return {f.name: (getattr(before, f.name), getattr(after, f.name))
            for f in fields(DesignParameters)
            if getattr(before, f.name) != getattr(after, f.name)}


def target_from_form(mode: str, *, build: str = "", chrom: str = "",
                     start: int = 0, end: int = 0, accession: str = "",
                     sequence: str = "") -> ExtractedTarget | None:
    """The form's target in the chat's terms, so the chat can keep working on it.

    A fusion junction has no `ExtractedTarget` form -- it is chosen from the
    loci file, never typed -- so it leaves the chat's target alone.
    """
    if mode == "Genomic region":
        return ExtractedTarget("genomic", build=build, chrom=chrom,
                               start=start, end=end)
    if mode == "RNA transcript":
        return ExtractedTarget("rna", accession=accession)
    if mode == "Pasted sequence":
        return ExtractedTarget("sequence", sequence=sequence)
    return None
