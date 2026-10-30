"""Streamlit front end for mypibypy.  Run with:  streamlit run app.py

Deliberately small. This UI is a form, a chat, a table and a download, and the
measured runtimes make Streamlit's rerun model fine -- a single-pair hg38 design
is about 2 s end to end and six tiled pairs about 4 s, so there is no job queue,
no progress streaming and no async machinery here. The alternative considered
was the React path (React + MUI + Vite, a FastAPI backend, auth, two repos,
thousands of lines); for this surface it would be more framework than feature.

LAYOUT. One Workspace page and a Help page. The workspace has two editors on
the left -- Chat and Form -- and the current design on the right. Both editors
change the same `DesignSession` (parameters, target, kept runs), so a tweak in
the chat shows up in the form and in the design panel, and there is one
download for whichever run is selected. Long explanations live on Help so the
workspace stays readable.

CACHING AND SEQUENCE DATA. `st.cache_data` is memory-only here on purpose --
`persist="disk"` would write real target sequence into Streamlit's cache
directory, and real sequences stay out of anything that could be committed or
synced. The same goes for the workbook: it is offered as a download from memory
rather than written into the repo.
"""
from __future__ import annotations

import base64
import os
import re
import time
from dataclasses import replace
from pathlib import Path

import streamlit as st

# `systems`/`loci`/`indexes` search the working directory and its PARENTS, which works for
# the CLI (run from the package directory) but not for a UI that may be launched
# from the repository root. Point them at the files next to this script when
# nothing else is configured, so the app behaves the same either way. The sidebar
# always shows which file was actually loaded, so this can never be ambiguous.
_HERE = Path(__file__).resolve().parent
for _var, _name in (("MYPIBYPY_SYSTEMS", "systems.local.json"),
                    ("MYPIBYPY_LOCI", "loci.local.json"),
                    ("MYPIBYPY_INDEXES", "indexes.local.json")):
    if not os.environ.get(_var) and (_HERE / _name).is_file():
        os.environ[_var] = str(_HERE / _name)
# The agent's run record has no file until the first run, so it is pointed here
# unconditionally: `RunStore()` then writes beside this script (gitignored), as
# the MCP server does. Without it the Form's agent runs were never kept.
if not os.environ.get("MYPIBYPY_AGENT_RUNS"):
    os.environ["MYPIBYPY_AGENT_RUNS"] = str(_HERE / "agent_runs.local.jsonl")

from mypibypy.agent import (
    Action,
    Budget,
    DesignAgent,
    DesignParameters,
    RunStore,
    build_client,
)
from mypibypy.agent.client import OllamaDecisionClient
from mypibypy.agent.intent import (
    CHAT_SYSTEM_PROMPT,
    Intent,
    chat_request,
    apply_intent,
    apply_screen,
    extract_target,
    intent_schema,
    MAX_SEQUENCE_NT,
    MIN_SEQUENCE_NT,
    clean_pasted_sequence,
    unreadable_sequence_chars,
)
from mypibypy.agent.context import ScreenStatus, build_chat_context, facts_reply
from mypibypy.agent.reply_check import checked_note, retry_request
from mypibypy.agent.tools import window_stats
from mypibypy.diagnosis import PREVIEW_CAVEAT, try_changes, windows_text
from mypibypy.design import DEFAULT_MIN_SPACING_NT
from mypibypy.indexes import (
    DEFAULT as SCREEN_DEFAULT,
    OFF as SCREEN_OFF,
    IndexesNotConfigured,
    kind_mismatch,
    load_indexes,
    resolve_screen,
)
from mypibypy.loci import Locus, load_fusions
from mypibypy.pipeline import design_fusion, design_locus, design_rna
from mypibypy.probe_map import map_data, pair_closeup_svg, region_charts, scored_windows
from mypibypy.sequence import GenomicInterval, fetch_sequence, fetch_transcript
from mypibypy.session import HEADLINE, DesignSession, changed_params, target_from_form
from mypibypy.specificity import (
    ENV_BOWTIE_INDEX,
    Backend,
    SpecificityNotConfigured,
    resolve_backend,
)
from mypibypy.systems import is_example, resolve_systems_path, systems
from mypibypy.expression import ENV_EXPRESSION, load_expression
from mypibypy.ui import (
    candidate_rows,
    expression_status,
    folding_details,
    no_change_reply,
    unscreened_pairs,
    unscreened_reason,
    workbook_bytes,
)

from app_help import render_help

ICON_PATH = Path(__file__).parent / "assets" / "icon.png"
st.set_page_config(page_title="mypibypy", page_icon=str(ICON_PATH), layout="wide")

#: Where to point the user when a design was not screened.
NO_SCREEN_HINT = ("Pick an index under *Screen against* in the Form, ask the "
                  "chat to screen it (e.g. *screen against hg38*), or set a "
                  "path under Settings.")

def no_screen_hint(screened_against: str, result) -> str:
    """What to do about an unscreened result. If an index was chosen and the
    screen still did not run, saying "pick an index" would be wrong -- say why.
    `screened_against` carries the reason when the screen could not start."""
    if screened_against and screened_against != "off":
        if ": " in screened_against:
            return f"The screen could not start — {screened_against}."
        return (f"The screen against **{screened_against}** did not run: "
                f"{unscreened_reason(result)}")
    return NO_SCREEN_HINT


TARGET_TYPES = ["Genomic region", "RNA transcript", "Pasted sequence",
                "Fusion junction"]
#: The chat's target kinds, mapped onto the form's target types.
KIND_TO_TYPE = {"genomic": "Genomic region", "rna": "RNA transcript",
                "sequence": "Pasted sequence"}
#: Columns of the compact table in the design panel; the rest are one click away.
KEY_COLUMNS = ["#", "footprint", "Tm L", "Tm R", "specificity",
               "worst off-target P(bound)", "self-folding"]
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# --- cached resources ------------------------------------------------------

@st.cache_resource
def registry() -> dict:
    """The amplifier registry. A resource, not data: it is loaded once and held."""
    return systems()


@st.cache_resource
def screening_backend(index: str) -> tuple[Backend | None, str]:
    """(bowtie handle, why not). Resource-cached so the index is resolved once."""
    if not index:
        return None, ""
    try:
        return resolve_backend(bowtie_index=index), ""
    except SpecificityNotConfigured as exc:
        return None, str(exc)


@st.cache_resource
def screen_indexes() -> tuple[dict, str]:
    """(named indexes, load error). Loaded once; a bad file is shown, not raised."""
    try:
        return load_indexes(), ""
    except IndexesNotConfigured as exc:
        return {}, str(exc)


def usable_indexes() -> list[str]:
    """Names whose bowtie files are actually on disk -- all the chat may pick."""
    return [name for name, idx in screen_indexes()[0].items() if idx.problem is None]


def screen_for(choice: str):
    """(backend or None, label) for a session's screen choice.

    The label is what the run records and the user reads: a name, a path for
    the sidebar/env fallback, "off", or -- when an index was chosen but the
    screen cannot start (bowtie missing, say) -- the name and the reason, so
    nobody is told to "pick an index" they already picked.
    """
    path, label = resolve_screen(choice, screen_indexes()[0], index_path)
    if not path:
        return None, ("off" if label == "off" else f"{label}: not available")
    backend, problem = screening_backend(path)
    return backend, (label if backend is not None else f"{label}: {problem}")


def screen_is_off(label: str) -> bool:
    return label == "off"


def screen_status(choice: str) -> ScreenStatus:
    """What the chat model is told about the screen. Names only, no paths."""
    backend, label = screen_for(choice)
    named = screen_indexes()[0]
    return ScreenStatus(
        on=backend is not None,
        # A sidebar/env path is not a name the model may echo back; say what
        # it is instead.
        against=(label if backend is None or label in named
                 else "a path set in the sidebar"),
        available=[str(named[n]) for n in usable_indexes()])


def screen_label(choice: str) -> str:
    """The Form dropdown's text for a choice."""
    if choice == SCREEN_OFF:
        return "Off — no screen"
    if choice == SCREEN_DEFAULT:
        path, _ = resolve_screen(SCREEN_DEFAULT, {}, index_path)
        return f"Default — {path}" if path else "Default — off (no path set)"
    return str(screen_indexes()[0][choice])


@st.cache_data(show_spinner="Fetching genomic sequence…")
def cached_region(build: str, chrom: str, start: int, end: int) -> str:
    """Memory-only. Never persist=\"disk\": that writes target sequence to disk."""
    return fetch_sequence(GenomicInterval(build, chrom, start, end))


@st.cache_resource(show_spinner="Loading expression table…")
def cached_expression(path: str):
    """A resource: the parsed table is held, and it carries no sequence."""
    return load_expression(path)


@st.cache_data(show_spinner="Fetching transcript…")
def cached_transcript(accession: str) -> str:
    return fetch_transcript(accession)


@st.cache_data(ttl=30, show_spinner=False)
def ollama_health(host: str, model: str) -> tuple[str, bool]:
    """(one line on the local model, whether it is usable). Two GETs, 1.5 s each.

    The sidebar used to say "Model: ollama" whether or not the server was up,
    the model pulled, or loaded; the first chat message then failed two minutes
    later. Cached for 30 s so a rerun does not ask again.
    """
    import json
    import urllib.error
    import urllib.request

    def get(path: str) -> dict:
        with urllib.request.urlopen(f"{host}{path}", timeout=1.5) as resp:
            return json.load(resp)

    def names(payload: dict) -> set[str]:
        return {m.get("name") or m.get("model") or "" for m in payload.get("models", [])}

    wanted = {model, f"{model}:latest"}
    try:
        pulled = names(get("/api/tags"))
    except (urllib.error.URLError, OSError, ValueError):
        return f"Ollama is not reachable at {host} (`ollama serve`)", False
    if not wanted & pulled:
        return f"{model} is not pulled (`ollama pull {model}`)", False
    try:
        running = [m for m in get("/api/ps").get("models", [])
                   if (m.get("name") or m.get("model")) in wanted]
    except (urllib.error.URLError, OSError, ValueError):
        running = []
    if running:
        return f"{model} loaded, {running[0].get('size', 0) / 1e9:.1f} GB", True
    return f"{model} pulled, not loaded (the first call loads it, ~10 s)", True


# --- helpers ---------------------------------------------------------------

def run_design(fn, *args, **kwargs):
    """Call a design function and turn its failures into readable messages.

    A design can legitimately fail -- a region that is entirely repeat-masked has
    no valid window, an accession can be wrong, a fetch can be blocked by a TLS
    proxy. None of those is a bug, and none should reach the user as a
    traceback.
    """
    from mypibypy.design import FusionGeometryError
    from mypibypy.loci import LociNotConfigured
    from mypibypy.sequence import TranscriptFetchError
    try:
        return fn(*args, **kwargs)
    except (ValueError, FusionGeometryError) as exc:
        st.error(f"**No design could be produced.** {exc}")
    except TranscriptFetchError as exc:
        st.error(f"**Could not fetch the transcript.** {exc}")
    except LociNotConfigured as exc:
        st.error(f"**Configuration problem.** {exc}")
    except OSError as exc:
        st.error(f"**Could not reach a reference service.** {exc}")
    return None


def session() -> DesignSession:
    """The one design state both editors share. Created on first use."""
    state = st.session_state
    if "design" not in state:
        state["design"] = DesignSession(
            params=DesignParameters(system=sorted(registry())[0]))
    return state["design"]


def windows_for(run, i: int) -> list:
    """Every scored window of result `i` of `run`, computed once per run.

    The re-scan is the slow part of the map, and a click on the map reruns the
    script; recomputing it on every click made selecting a pair sluggish.
    """
    cache = st.session_state.setdefault("map_windows", {})
    key = (run.number, i)
    if key not in cache:
        cache[key] = scored_windows(run.results[i])
    return cache[key]


def design_for_target(target, params: DesignParameters, backend):
    """Run the deterministic design for an extracted target. No model involved."""
    from mypibypy.loci import RnaTarget

    if target.kind == "genomic":
        locus = Locus("target", "GENE", "from chat",
                      GenomicInterval(target.build, target.chrom,
                                      target.start, target.end),
                      params.system)
        return run_design(design_locus, locus, backend=backend,
                          screen=backend is not None, n_pairs=params.n_pairs,
                          min_spacing=params.min_spacing)
    if target.kind == "rna":
        rna = RnaTarget("target", "GENE", target.accession, params.system)
        seq = run_design(cached_transcript, target.accession)
        if seq is None:
            return None
        return run_design(design_rna, rna, backend=backend,
                          screen=backend is not None, transcript_seq=seq,
                          n_pairs=params.n_pairs, min_spacing=params.min_spacing)
    if target.kind == "sequence":
        rna = RnaTarget("target", "GENE", "pasted-sequence", params.system)
        return run_design(design_rna, rna, backend=backend,
                          screen=backend is not None,
                          transcript_seq=target.sequence,
                          n_pairs=params.n_pairs, min_spacing=params.min_spacing)
    return None


def show_run_record(record) -> None:
    """Why, not just what. The decisions and the evidence behind them."""
    cols = st.columns(4)
    cols[0].metric("Turns", record.turns_used)
    cols[1].metric("Relaxations", record.relaxations_used)
    cols[2].metric("Model calls", record.model_calls)
    cols[3].metric("Status", record.status)
    if not record.consulted_model:
        st.caption("Settled deterministically — the model was not consulted. "
                   "That is the expected path for a clear case.")
    for event in record.events:
        icon = {"model_decision": "🤖", "deterministic_decision": "⚙️",
                "parameter_clamp": "📏", "budget": "⏳", "refusal": "🚫",
                "error": "⚠️", "outcome": "🏁"}.get(event.kind, "•")
        st.write(f"{icon} **turn {event.turn}** · `{event.kind}` — {event.summary}")
    with st.expander("Full record (JSON)"):
        st.json(record.to_dict())


# --- chat editor -----------------------------------------------------------

def chat_editor(s: DesignSession, client_kind: str,
                system_keys: list[str]) -> None:
    """Converse with the pipeline.

    The model does exactly two things here: turn a request into parameters, and
    pick which constraint gives when nothing passes. Everything else -- the
    target, the design, the screens, the workbook -- is deterministic.

    THE TARGET IS NEVER THE MODEL'S. It is pulled from the user's own words by
    `extract_target`, a strict pattern in the harness. The model has no field in
    which to supply a region, an accession or a sequence, which is what keeps
    the no-model-authored-sequences guarantee intact once a conversation is
    involved.

    The design panel is drawn after this returns, so it already shows what this
    message changed -- no second rerun is needed to bring it up to date.
    """
    history = st.session_state.setdefault("chat", [])
    log = st.container(height=520)
    with log:
        if not history:
            st.caption(
                "Try: *design six pairs for this enhancer in cy5* · *why did I "
                "only get 3?* · *give me 12 pairs at 20 nt spacing on "
                "chr7:5530000-5531000*")
        for entry in history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["text"])

    # A turn takes 10-30 s on this hardware. A second message sent meanwhile
    # would queue behind it and both would run, so the box is disabled until
    # the turn is over.
    message = st.chat_input("Ask for a design, or adjust one…",
                            submit_mode="disable")
    if not message:
        return

    def say(role: str, text: str) -> None:
        history.append({"role": role, "text": text})
        with log, st.chat_message(role):
            st.markdown(text)

    say("user", message)

    client = build_client(client_kind)
    if client is None:
        say("assistant", "No decision model is selected, so I cannot interpret "
                         "that. Pick **ollama** under *Settings* in the sidebar, "
                         "or use the Form.")
        return

    # The reply is one assistant message. The steps drawn inside it show what
    # the turn is doing and how long each part took -- reading the request,
    # designing, checking the answer -- because the model alone is 7-16 s here
    # and a single spinner over all of it read as a hang. History keeps the
    # final text only.
    reply_box = log.chat_message("assistant")

    def finish(text: str) -> None:
        history.append({"role": "assistant", "text": text})
        reply_box.markdown(text)

    # 1. The target comes from the user's words, by pattern, not from the model.
    found = extract_target(message)

    # 2. The model's first job: request -> parameters. It is told what the
    # session already holds, so a follow-up need not repeat the target.
    current = found if found.found else s.target
    last = s.runs[-1] if s.runs else None
    on_target = last if last and current and last.target_label == str(current) else None
    context = build_chat_context(
        message, s.params, str(current) if current else "", on_target,
        screen=screen_status(s.screen),
        windows=(windows_for(on_target, 0)
                 if on_target and len(on_target.results) == 1 else None))
    index_keys = tuple(usable_indexes())
    # history[-1] is this message, already sent as USER MESSAGE.
    request = chat_request(message, context, history[:-1])
    schema = intent_schema(tuple(system_keys), index_keys)
    reading = reply_box.status("Reading the request", type="step")
    payload, error, seconds = client.structured(CHAT_SYSTEM_PROMPT, request, schema)
    if error or payload is None:
        reading.update(label="Could not read the request", state="error")
        finish(f"I could not read that request. `{error}`")
        return
    try:
        intent = Intent.from_payload(payload, tuple(system_keys), index_keys)
    except Exception as exc:
        reading.update(label="The model's reply was refused", state="error")
        finish(f"That reply did not fit the allowed shape and was refused: `{exc}`")
        return
    reading.update(label=f"Read the request as *{intent.intent}* ({seconds:.1f} s)",
                   state="complete")

    # The note is checked against what the model was shown before anyone sees
    # it (`reply_check`). An answer gets one retry and then the software's own
    # facts; a comment on a design is simply dropped -- the parameters it came
    # with are validated separately and do not depend on it.
    grounds = CHAT_SYSTEM_PROMPT + "\n" + request

    def ask_again(problems: list[str]) -> str | None:
        checking = reply_box.status(f"Checking the answer: {problems[0]}",
                                    type="step")
        again, err, again_s = client.structured(
            CHAT_SYSTEM_PROMPT, retry_request(request, problems), schema)
        try:
            retried = Intent.from_payload(again, tuple(system_keys), index_keys) \
                if again is not None and not err else None
        except Exception:
            retried = None
        ok = retried is not None and retried.intent == intent.intent
        checking.update(label=(f"Asked again ({again_s:.1f} s)" if ok
                               else "Could not get a checkable answer"),
                        state="complete" if ok else "error")
        return retried.note if ok else None

    if intent.intent == "explain":
        note, settled = checked_note(
            intent.note, grounds, enforce_window=s.params.enforce_window,
            ask_again=ask_again, fallback=facts_reply(context))
    else:
        note, settled = checked_note(
            intent.note, grounds, enforce_window=s.params.enforce_window,
            ask_again=lambda problems: None, fallback="")

    new_params, changes = apply_intent(intent, s.params, tuple(system_keys))
    s.screen, screen_change = apply_screen(intent, s.screen)
    if screen_change:
        changes.append(screen_change)
    s.update(params=new_params, target=found if found.found else None)

    # The note is the MODEL'S COMMENT. Labelled as such, never treated as a
    # justification -- in trials the chosen action was right while the
    # sentence explaining it was backwards.
    if intent.intent == "explain":
        lines = [note]
    else:
        lines = [f"_Model comment ({seconds:.1f} s): {note}_"] if note else []
    if changes:
        lines.append("**Applied:** " + "; ".join(changes))

    if intent.intent == "explain":
        pass
    elif intent.intent == "unclear":
        lines.append("Nothing was changed." + ("" if s.target else
                     " Tell me the target and what you want designed."))
    elif not s.target:
        lines.append("I have the parameters but no target yet. Give me a "
                     "region like `chr7:5530000-5531000`, an accession "
                     "like `NM_000546.6`, or paste a sequence (51 nt or "
                     "more).")
    else:
        target = s.target
        backend, screened_against = screen_for(s.screen)
        if (on_target is not None and not changes
                and on_target.screen == screened_against):
            # Nothing the design uses has moved and the pipeline is
            # deterministic, so designing again would reproduce run N under a
            # model comment promising a change ("I will loosen the Tm..."):
            # the comment is not shown, and the reply says what would change it.
            lines = [no_change_reply(intent.intent, on_target.number)]
            previews = try_changes(on_target.results, on_target.params,
                                   windows_for(on_target, 0)
                                   if len(on_target.results) == 1 else None)
            if previews:
                lines.append("**What would place more:**\n"
                             + "\n".join(f"- {p.text}" for p in previews)
                             + f"\n\n_{PREVIEW_CAVEAT}_")
            finish("\n\n".join(lines))
            return
        designing = reply_box.status(f"Designing against {target}", type="step")
        started = time.time()
        with reply_box:        # a design error lands in the reply, not the page
            result = design_for_target(target, new_params, backend)
        if result is None:
            designing.update(label="No design could be produced", state="error")
        else:
            designing.update(label=f"Designed {result.n_pairs} pair(s) against "
                                   f"{target} ({time.time() - started:.1f} s)",
                             state="complete")
            run = s.record(source="chat", results=[result],
                           note="mypibypy chat session", mode="chat",
                           target_label=str(target), screen=screened_against)
            n_unscreened = unscreened_pairs(result)
            lines.append(f"**{result.n_pairs} pair(s)** designed against "
                         f"`{target}` — run #{run.number}, shown on the right.")
            if result.repacked_from:
                lines.append(
                    f"↔️ Placing the best-ranked windows first fit only "
                    f"{result.repacked_from}, so the pairs were **re-laid out** "
                    f"across the target to fit {result.n_pairs}. That uses some "
                    f"lower-ranked windows — compare the penalties in the table.")
            if n_unscreened:
                lines.append(
                    f"⚠️ **No specificity screen ran** for {n_unscreened} of "
                    f"{result.n_pairs} pair(s) — nothing here establishes "
                    f"single-copy status. "
                    + no_screen_hint(screened_against, result))
            else:
                lines.append(f"Screened against **{screened_against}**.")
            lines.append(f"_Windows: {windows_text([result], new_params)[0]}._")
            previews = try_changes([result], new_params, windows_for(run, 0))
            if previews:
                lines.append("**What would place more:**\n"
                             + "\n".join(f"- {p.text}" for p in previews)
                             + f"\n\n_{PREVIEW_CAVEAT}_")

    finish("\n\n".join(lines))


# --- form editor -----------------------------------------------------------

def run_agent_loop(results: list, params: DesignParameters, redesign,
                   client_kind: str, system_keys: list[str], note: str):
    """The Form's design through the agent loop, with the pipeline really re-run.

    `redesign(p)` is the same pipeline call the Design button made, with the
    loop's parameters, so a change the model asks for is applied -- for the two
    it can apply here. The pipeline takes a pair count, a spacing and a channel
    and ranks on GC/Tm rather than rejecting, so relax_tm, relax_gc,
    widen_region and accept_single_arm_hits would design the identical thing
    again: the loop is told so (`allowed_actions`) and stops for a human if the
    model picks one, instead of re-asking until the turn budget failed a good
    design. A fusion junction is one fixed footprint with its channels in the
    loci file, so it gets the terminal actions only (`redesign` is None).

    A design nobody screened is flagged by the classifier itself, without a
    model call; that is the common case on a machine with no index configured.
    """
    first = results[0]
    fusion = redesign is None

    def design_fn(p: DesignParameters):
        # Turn 0 is the design the button just made; the pipeline runs again
        # only once the loop has changed something.
        result = first if fusion or p == params else redesign(p)
        if result is None:
            return None, []
        return result, (list(result.candidates) if fusion else scored_windows(result))

    allowed = () if fusion else (Action.REDUCE_PAIRS, Action.SWITCH_SYSTEM)
    # Kept in the same run record as the MCP server's runs ($MYPIBYPY_AGENT_RUNS,
    # pointed beside this script at the top of the file).
    agent = DesignAgent(design_fn, client=build_client(client_kind),
                        budget=Budget(max_turns=4, max_relaxations=3),
                        known_systems=tuple(system_keys), store=RunStore(),
                        allowed_actions=allowed)
    with st.spinner("Agent loop: classifying the evidence…" if fusion else
                    "Agent loop: classifying the evidence, and designing again "
                    "if the model asks for fewer pairs or the other channel…"):
        return agent.run(params, target_label=note or "UI run",
                         request="designed from the UI")


def form_editor(s: DesignSession, system_keys: list[str],
                use_agent_default: bool, client_kind: str) -> None:
    """Fill in a target and settings, then press Design.

    Widgets are keyed on `s.revision`, which the chat bumps whenever it changes
    the parameters or the target. A new key makes Streamlit build the widget
    afresh from its default -- and the defaults come from the session -- so the
    form always opens on what the chat last set, instead of on stale values.

    No `st.form`: nothing designs until the button is pressed anyway, and a
    form would freeze the live checks under the pasted-sequence box (length,
    unreadable characters) until submit.
    """
    rev = s.revision
    params, target = s.params, s.target
    kind_type = KIND_TO_TYPE.get(target.kind) if target else None

    mode = st.segmented_control("Target type", TARGET_TYPES, required=True,
                                default=kind_type or "Genomic region",
                                key=f"f_mode_{rev}")
    system_key = st.selectbox(
        "Amplifier system", system_keys,
        index=system_keys.index(params.system) if params.system in system_keys else 0,
        format_func=lambda k: f"{k} ({registry()[k].fluorophore})",
        key=f"f_system_{rev}")

    if mode == "Fusion junction":
        n_pairs, min_spacing = 1, DEFAULT_MIN_SPACING_NT
        st.caption("A junction is one position, so tiling does not apply.")
    else:
        c1, c2 = st.columns(2)
        n_pairs = c1.number_input(
            "Pairs to tile", 1, 50, int(params.n_pairs), key=f"f_pairs_{rev}",
            help="Adjacent probe pairs along the target; more pairs, more signal.")
        min_spacing = c2.number_input(
            "Minimum spacing (nt)", 0, 5000, int(params.min_spacing),
            key=f"f_spacing_{rev}", help="Gap kept between neighbouring pairs.")

    screen_options = [SCREEN_DEFAULT, *usable_indexes(), SCREEN_OFF]
    screen_choice = st.selectbox(
        "Screen against", screen_options,
        index=screen_options.index(s.screen) if s.screen in screen_options else 0,
        format_func=screen_label, key=f"f_screen_{rev}",
        help="The bowtie v1 index each pair is screened against for "
             "off-targets. Name indexes in indexes.local.json; Default uses "
             "the path under Settings in the sidebar.")
    backend, screened_against = screen_for(screen_choice)

    with st.expander("Advanced"):
        use_agent = st.checkbox(
            "Run through the agent loop", value=use_agent_default, key="f_agent",
            help="Adds the deterministic classifier and, for genuinely "
                 "conflicting signals, a model decision. Clear cases are "
                 "settled without a model, and a design nobody screened is "
                 "flagged for you without one. If the model asks for fewer "
                 "pairs or the other channel, the design is run again with "
                 "that change (up to 4 designs, 3 relaxations). The GC/Tm "
                 "window is not relaxed here: the pipeline ranks on it rather "
                 "than rejecting, so that would change nothing.")

    new_params = replace(params, system=system_key, n_pairs=int(n_pairs),
                         min_spacing=int(min_spacing))
    results, note, label, form_target = None, "", "", None
    redesign = None         # the same design again with the loop's parameters

    if mode == "Genomic region":
        g = target if target and target.kind == "genomic" else None
        col1, col2 = st.columns([2, 3])
        build = col1.text_input("Genome build", g.build if g else "hg38",
                                key=f"f_build_{rev}")
        region = col2.text_input(
            "Region (chrom:start-end)",
            f"{g.chrom}:{g.start}-{g.end}" if g else "",
            placeholder="chr5:142906255-142906754", key=f"f_region_{rev}")
        with st.expander("Labels (optional)"):
            lc1, lc2 = st.columns(2)
            gene = lc1.text_input("Gene label", "GENE", key="f_gene_label")
            name = lc2.text_input("Target name", "target", key="f_target_name")
        if st.button("Design probes", type="primary", disabled=not region):
            try:
                chrom, span = region.split(":")
                start, end = (int(x.replace(",", "")) for x in span.split("-"))
            except ValueError:
                st.error("Region must look like chr5:142906255-142906754")
                return
            locus = Locus(name, gene, "designed in the UI",
                          GenomicInterval(build, chrom, start, end), system_key)
            with st.spinner("Designing…"):
                one = run_design(design_locus, locus, backend=backend,
                                 screen=backend is not None, n_pairs=int(n_pairs),
                                 min_spacing=int(min_spacing))
            if one is not None:
                results, label = [one], f"{build} {chrom}:{start:,}-{end:,}"
                note = f"mypibypy UI: {locus.interval}, system {system_key}."
                form_target = target_from_form(mode, build=build, chrom=chrom,
                                               start=start, end=end)

                def redesign(p: DesignParameters, locus=locus):
                    return run_design(design_locus, replace(locus, system=p.system),
                                      backend=backend, screen=backend is not None,
                                      n_pairs=p.n_pairs, min_spacing=p.min_spacing)

    elif mode == "RNA transcript":
        r = target if target and target.kind == "rna" else None
        c1, c2 = st.columns(2)
        accession = c1.text_input("Accession", r.accession if r else "",
                                  placeholder="NM_000546.6 or ENST…",
                                  key=f"f_accession_{rev}")
        gene = c2.text_input("Gene symbol", "", placeholder="TP53", key="f_rna_gene",
                             help="Used to exclude on-target hits from the screen.")
        expression_path = st.text_input(
            "Expression table (optional)", os.environ.get(ENV_EXPRESSION, ""),
            placeholder="TPM table for the imaged cell type (.tsv)",
            key="f_expression",
            help="Weights each off-target gene by its abundance. Leave blank and "
                 "the results will say weighting was not evaluated.")
        st.caption("Screen an RNA target against a **transcriptome** index — "
                   "see *Help*.")
        chosen = screen_indexes()[0].get(screen_choice)
        mismatch = kind_mismatch(chosen, "transcriptome") if chosen else None
        if mismatch:
            st.warning(mismatch)
        if st.button("Design probes", type="primary",
                     disabled=not (accession and gene)):
            from mypibypy.loci import RnaTarget
            rna = RnaTarget("target", gene, accession.strip(), system_key)
            expression, one = None, None
            if expression_path.strip():
                try:
                    expression = cached_expression(expression_path.strip())
                except (OSError, ValueError) as exc:
                    st.error(f"**Could not load the expression table.** {exc}")
                    expression = False
            if expression is not False:
                with st.spinner("Fetching and designing…"):
                    seq = run_design(cached_transcript, accession.strip())
                    one = run_design(design_rna, rna, backend=backend,
                                     screen=backend is not None, transcript_seq=seq,
                                     n_pairs=int(n_pairs),
                                     min_spacing=int(min_spacing),
                                     expression=expression) if seq else None
            if one is not None:
                results, label = [one], accession.strip()
                note = (f"mypibypy UI: RNA target {accession}, "
                        f"system {system_key}.")
                form_target = target_from_form(mode, accession=accession.strip())

                def redesign(p: DesignParameters, rna=rna, seq=seq,
                             expression=expression):
                    return run_design(design_rna, replace(rna, system=p.system),
                                      backend=backend, screen=backend is not None,
                                      transcript_seq=seq, n_pairs=p.n_pairs,
                                      min_spacing=p.min_spacing,
                                      expression=expression)

    elif mode == "Pasted sequence":
        p = target if target and target.kind == "sequence" else None
        pasted = st.text_area(
            "Sequence (DNA or RNA, FASTA header optional)",
            p.sequence if p else "", height=140, key=f"f_pasted_{rev}",
            placeholder="Paste at least 51 nt, e.g. a transcript or an enhancer.")
        cleaned = clean_pasted_sequence(pasted)
        st.caption((f"{len(cleaned):,} nt read" if cleaned else "Nothing pasted yet.")
                   + " · no genome position; see *Help* on self-hits.")
        bad = unreadable_sequence_chars(pasted)
        if bad:
            st.error(f"Only A, C, G, T and U are accepted; found `{bad}`. Fix or "
                     "remove those characters — they are not dropped silently, "
                     "because that would shift the sequence.")
        too_short = bool(cleaned) and len(cleaned) < MIN_SEQUENCE_NT
        too_long = len(cleaned) > MAX_SEQUENCE_NT
        if too_short:
            st.error(f"Need at least {MIN_SEQUENCE_NT} nt; got {len(cleaned)}.")
        if too_long:
            st.error(f"Longer than {MAX_SEQUENCE_NT:,} nt; paste a shorter region.")
        if st.button("Design probes", type="primary",
                     disabled=not cleaned or too_short or too_long or bool(bad)):
            from mypibypy.loci import RnaTarget
            rna = RnaTarget("target", "GENE", "pasted-sequence", system_key)
            with st.spinner("Designing…"):
                one = run_design(design_rna, rna, backend=backend,
                                 screen=backend is not None, transcript_seq=cleaned,
                                 n_pairs=int(n_pairs), min_spacing=int(min_spacing))
            if one is not None:
                results = [one]
                label = f"pasted sequence ({len(cleaned):,} nt)"
                note = f"mypibypy UI: {label}, system {system_key}."
                form_target = target_from_form(mode, sequence=cleaned)

                def redesign(p: DesignParameters, rna=rna, cleaned=cleaned):
                    return run_design(design_rna, replace(rna, system=p.system),
                                      backend=backend, screen=backend is not None,
                                      transcript_seq=cleaned, n_pairs=p.n_pairs,
                                      min_spacing=p.min_spacing)

    else:
        try:
            fusions = load_fusions()
        except Exception as exc:  # a missing or malformed loci file
            st.error(f"Could not load fusions: {exc}")
            fusions = []
        if not fusions:
            st.info("No fusion junctions are defined. Add a `fusions` entry to "
                    "your loci file — see `loci.example.json`.")
        else:
            chosen = st.selectbox("Junction", fusions, format_func=str,
                                  key="f_fusion")
            if st.button("Design probes", type="primary"):
                with st.spinner("Designing both strands…"):
                    both = run_design(design_fusion, chosen, backend=backend,
                                      screen=backend is not None)
                if both:
                    results, label = both, f"fusion {chosen.name}"
                    note = f"mypibypy UI: fusion junction {chosen.name}."

    if not results:
        return
    # The form's widgets already hold these values, so they are not reset.
    s.update(params=new_params, target=form_target, from_form=True)
    s.screen = screen_choice

    # The agent loop belongs to the click that produced the design, not to
    # every rerun, so it runs here and is stored with the run.
    agent_run = None
    if use_agent:
        agent_run = run_agent_loop(results, new_params, redesign, client_kind,
                                   system_keys, note)
        if agent_run.ok and agent_run.design is not None \
                and agent_run.design is not results[0]:
            # The loop reached an acceptable design by changing something
            # (fewer pairs, the other channel): the run keeps that design, and
            # the form reopens on the loop's settings. A run that failed or
            # was flagged keeps the design the button made, under the verdict.
            results = [agent_run.design, *results[1:]]
            new_params = agent_run.parameters
            s.update(params=new_params)
    s.record(source="form", results=results, note=note, target_label=label,
             mode=mode, agent=agent_run, screen=screened_against)


# --- design panel ----------------------------------------------------------

def _param_text(name: str, value) -> str:
    if name == "system":
        system = registry().get(value)
        return f"{value} ({system.fluorophore})" if system else str(value or "—")
    if name == "min_spacing":
        return f"{value} nt"
    return str(value)


def _summary_line(params: DesignParameters, before: DesignParameters | None,
                  target_label: str) -> str:
    """Channel · pairs · spacing · target, with what changed since `before`."""
    changed = changed_params(before, params)
    names = {"system": "Channel", "n_pairs": "Pairs", "min_spacing": "Spacing"}
    bits = []
    for name in HEADLINE:
        text = f"{names[name]} **{_param_text(name, getattr(params, name))}**"
        if name in changed:
            text += f" (was {_param_text(name, changed[name][0])})"
        bits.append(text)
    others = [f"{k} {v[0]} → {v[1]}" for k, v in changed.items()
              if k not in HEADLINE]
    line = " · ".join(bits) + f" · Target `{target_label}`"
    if others:
        line += "  \nAlso changed: " + "; ".join(others)
    return line


def ink_colour() -> str:
    """Text colour for drawings shown as images, which do not inherit the page's."""
    theme = getattr(st.context, "theme", None)
    return "#F1EFE8" if getattr(theme, "type", "light") == "dark" else "#2C2C2A"


def show_svg(svg: str) -> None:
    """Show an SVG as an image at its own size, scrolling sideways if wider.

    An image rather than `st.html`, which sanitises inline SVG away. Shown at
    native size because scaling a base-by-base drawing to a half-width column
    shrank the bases past legibility.
    """
    width = re.search(r'viewBox="0 0 ([\d.]+)', svg)
    data = base64.b64encode(svg.encode()).decode()
    size = f"max-width:none;width:{width.group(1)}px" if width else "width:100%"
    st.markdown(f'<div style="overflow-x:auto"><img src="data:image/svg+xml;'
                f'base64,{data}" style="{size}" alt="probe pair close-up"></div>',
                unsafe_allow_html=True)


def _selected_pair(key: str, n_pairs: int) -> int:
    """The pair clicked on the map (1-based), read before the map is redrawn.

    Reading the widget's state first, rather than the chart's return value,
    means the highlight drawn this rerun already matches the click.
    """
    event = st.session_state.get(key)
    try:
        picked = int(event["selection"]["pair"][0]["pair"])
    except (KeyError, IndexError, TypeError, ValueError):
        picked = 1
    return picked if 1 <= picked <= n_pairs else 1


def render_result(run, i: int, result, previous_run) -> None:
    """One result of a run: map, compact table, close-up, details."""
    data = map_data(result, windows_for(run, i))
    prev_data = None
    if previous_run is not None and i < len(previous_run.results):
        prev_data = map_data(previous_run.results[i], windows_for(previous_run, i))

    key = f"map_{run.number}_{i}"
    pair_no = _selected_pair(key, result.n_pairs)
    charts = region_charts(data, prev_data, pair_no)
    st.altair_chart(charts["pairs"], key=key, on_select="rerun",
                    selection_mode="pair", width="stretch",
                    alt="Ordered probe pairs along the target; click one for its close-up")
    for track, alt in (("windows", "Best-scoring windows stacked into lanes"),
                       ("scores", "Window penalty along the target, lower is better")):
        if charts[track] is not None:
            st.altair_chart(charts[track], width="stretch", key=f"{key}_{track}",
                            alt=alt)
    st.caption("Click a pair to open its close-up. "
               + ("Grey row: the previous run on this target. "
                  if prev_data and prev_data.pairs else "")
               + data.note)

    n_unscreened = unscreened_pairs(result)
    if n_unscreened:
        st.warning(f"**No specificity screen** for {n_unscreened} of "
                   f"{result.n_pairs} pair(s) — nothing here establishes "
                   f"single-copy status. {no_screen_hint(run.screen, result)}")
    elif run.screen and run.screen != "off":
        st.caption(f"Screened against **{run.screen}**.")

    rows = candidate_rows(result)
    st.dataframe([{k: row[k] for k in KEY_COLUMNS} for row in rows],
                 hide_index=True, alt="Key numbers for each designed pair")

    with st.expander(f"Pair {pair_no} close-up", expanded=True):
        show_svg(pair_closeup_svg(result, pair_no - 1, ink=ink_colour()))
        st.caption(folding_details(result)[pair_no - 1])

    with st.expander("All columns, folding and windows"):
        st.dataframe(rows, hide_index=True,
                     alt="Every column for each designed pair, including folding "
                         "and off-target genes")
        weighted, expression_text = expression_status(result)
        st.caption(expression_text if weighted else f"⚠️ {expression_text}")
        for line in folding_details(result):
            st.markdown(f"- {line}")
        stats = window_stats(list(result.candidates), run.params)
        st.caption(f"Windows scored: {stats}. P(bound), the GC/Tm window and "
                   "these scores are explained under *Help*.")


def design_panel(s: DesignSession) -> None:
    """The current design: whichever run is selected, plus any pending edits."""
    head, picker = st.columns([3, 2], vertical_alignment="bottom")
    head.subheader("Current design")
    run = s.current
    if run is None:
        st.markdown(_summary_line(s.params, None, str(s.target or "none yet")))
        st.caption("Nothing designed yet — describe a target in the chat, or "
                   "fill in the form and press **Design probes**.")
        return

    numbers = [r.number for r in reversed(s.runs)]
    chosen = picker.selectbox(
        "Run", numbers, index=numbers.index(run.number),
        format_func=lambda n: s.runs[n - 1].label, label_visibility="collapsed")
    if chosen != run.number:
        s.select(chosen)
        run = s.current

    previous = s.previous(run)
    st.markdown(_summary_line(run.params, previous.params if previous else None,
                              run.target_label))
    if run is s.runs[-1] and (changed_params(run.params, s.params)
                              or (s.target and str(s.target) != run.target_label
                                  and run.mode != "Fusion junction")):
        st.caption("✏️ Edited since this run: "
                   + _summary_line(s.params, run.params, str(s.target or "—"))
                   + " — design again to apply.")

    results = run.results
    total = sum(r.n_pairs for r in results)
    unscreened = sum(unscreened_pairs(r) for r in results)
    m1, m2, m3 = st.columns(3)
    m1.metric("Pairs designed", total)
    m2.metric("Specificity screen", "not run" if unscreened == total
              else f"{total - unscreened}/{total} screened")
    m3.metric("Channel", results[0].system.fluorophore)
    requested = run.params.n_pairs * len(results)
    if run.mode != "Fusion junction" and total < requested:
        st.caption(f"⚠️ Only {total} of the {requested} requested pairs were "
                   "placed — the tiler found no more non-overlapping windows at "
                   "this spacing among the ones it ranked.")

    if run.agent is not None:
        st.info(f"**Agent: {run.agent.status}** — {run.agent.message}")
        with st.expander("Agent run record"):
            show_run_record(run.agent.record)

    if len(results) > 1:
        tabs = st.tabs([f"{r.locus.gene} {r.locus.name}" for r in results])
        for i, (tab, result) in enumerate(zip(tabs, results)):
            with tab:
                render_result(run, i, result, previous)
    else:
        render_result(run, 0, results[0], previous)

    st.download_button(
        "⬇️ Download order workbook (.xlsx)",
        data=workbook_bytes(results, run.note),
        file_name=f"mypibypy_order_run{run.number}.xlsx", mime=XLSX,
        type="primary", key=f"download_{run.number}",
        help="Probe sequences are in the workbook. They are not written "
             "anywhere inside the repository.")


# --- sidebar: status, then settings you set once ---------------------------

reg = registry()

with st.sidebar:
    _icon_col, _title_col = st.columns([1, 3], vertical_alignment="center")
    _icon_col.image(str(ICON_PATH))
    _title_col.title("mypibypy")
    st.caption("piFISH split-probe design")
    status = st.container()
    with st.expander("Settings"):
        index_path = st.text_input(
            "bowtie index (optional)", key="index_path",
            placeholder=os.environ.get(ENV_BOWTIE_INDEX, "") or None,
            help="A bowtie v1 index basename, used when a design's *Screen "
                 "against* is Default. Leave blank to fall back to "
                 f"${ENV_BOWTIE_INDEX}, or to no screen. Named indexes go in "
                 "indexes.local.json and are picked in the Form or the chat.")
        _named, _index_error = screen_indexes()
        if _index_error:
            st.error(f"Indexes file not loaded: {_index_error}")
        for _idx in _named.values():
            if _idx.problem:
                st.warning(f"Index **{_idx.name}** unusable: {_idx.problem}")
        client_kind = st.selectbox(
            "Decision model", ["ollama", "none", "anthropic"], index=0,
            key="client_kind",
            help="Local model by default. 'none' runs the pipeline with no "
                 "model at all; the chat then cannot read requests.")
        # `loaded_from()` reports only what is currently cached, and Streamlit
        # can re-import the module on a hot reload while `registry()` stays
        # cached -- which showed "systems: None" even though the real file was
        # loaded. Resolve the path directly so the caption always names the
        # file that would be used.
        st.caption(f"Amplifier systems file: `{resolve_systems_path()}`")



def model_line() -> str:
    """The sidebar's model line: what is actually there, not just what is selected."""
    if client_kind == "none":
        return "⛔ Model: none — chat disabled"
    if client_kind != "ollama":
        return f"🤖 Model: {client_kind}"
    probe = OllamaDecisionClient()          # host and model from the environment; no call
    text, ok = ollama_health(probe.host, probe.model)
    return f"{'🤖' if ok else '⚠️'} Model: {text}"


def sidebar_status() -> None:
    """Drawn after the page, so a screen the chat just switched on shows now."""
    backend, against = screen_for(session().screen)
    with status:
        st.markdown(
            ("🧪 **Systems: synthetic examples** — not orderable" if is_example()
             else "✅ Systems: real") + "  \n"
            + ("⚠️ **Specificity screen: off** — pick an index in the Form"
               if screen_is_off(against) else
               f"⚠️ **Specificity screen: {against}**" if backend is None
               else f"✅ Specificity screen: {against}")
            + "  \n"
            + model_line())


# --- pages -------------------------------------------------------------------

def workspace() -> None:
    s = session()
    if is_example():
        st.error(f"Amplifier systems are the SYNTHETIC EXAMPLES "
                 f"({resolve_systems_path().name}). Probes designed now are "
                 "**not orderable** — see *Help*.")
    left, right = st.columns([5, 6], gap="large")
    with left:
        editor = st.segmented_control("Editor", ["Chat", "Form"], default="Chat",
                                      key="editor", label_visibility="collapsed")
        if editor == "Form":
            form_editor(s, sorted(reg), False, client_kind)
        else:
            chat_editor(s, client_kind, sorted(reg))
    # Drawn after the editor, so it already reflects whatever the editor did.
    with right:
        design_panel(s)


def help_page() -> None:
    render_help(systems_path=resolve_systems_path(), example=is_example(),
                screen_on=screen_for(session().screen)[0] is not None,
                client_kind=client_kind)


st.navigation([st.Page(workspace, title="Workspace", icon=":material/biotech:",
                       default=True),
               st.Page(help_page, title="Help", icon=":material/help:")],
              position="top").run()
sidebar_status()
