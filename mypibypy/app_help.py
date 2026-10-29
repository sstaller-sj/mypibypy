"""The Help page: how the pipeline works, what the numbers mean, setup status.

These explanations used to sit as captions under every result. They are true
and worth reading once, but repeated on every card they buried the result; the
workspace now keeps one short line and points here.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_help(*, systems_path: Path, example: bool, screen_on: bool,
                client_kind: str) -> None:
    st.title("Help")

    st.subheader("Setup")
    st.markdown(
        f"- {'🧪' if example else '✅'} **Amplifier systems** — `{systems_path}`"
        + (" holds the **synthetic examples**. Probes designed with them are "
           "not orderable: copy `systems.example.json` to `systems.local.json` "
           "and put your real arms / P2 / P3 / P4 in it (see the README)."
           if example else ".")
        + f"\n- {'✅' if screen_on else '⚠️'} **Specificity screen** — "
        + ("a bowtie index is set." if screen_on else
           "off. Name your bowtie v1 indexes in `indexes.local.json` (copy "
           "`src/mypibypy/indexes.example.json`) and pick one under *Screen "
           "against* in the Form or by asking the chat (*screen against "
           "hg38*); or enter a path under *Settings* in the sidebar. Until "
           "then every pair says NOT SCREENED and nothing establishes "
           "single-copy status.")
        + f"\n- {'⛔' if client_kind == 'none' else '🤖'} **Decision model** — "
        + (f"`{client_kind}`. The chat needs it to read requests; the form "
           "never does." if client_kind != "none" else
           "none. The form works; the chat cannot read requests."))

    st.subheader("The workspace")
    st.markdown(
        "- **Chat** and **Form** edit the same design. Change the pair count in "
        "the chat and the form opens on the new value; design from the form "
        "and the chat can keep adjusting that target.\n"
        "- **Current design** shows the selected run. Every design is kept — "
        "pick an earlier one from the run menu to compare or download it.\n"
        "- The **target is never the model's**: regions, accessions and pasted "
        "sequences are read from your own words by a fixed pattern. The model "
        "only turns a request into parameters.")
    st.markdown("**Example chat requests**")
    st.markdown(
        "- *give me 12 pairs at 20 nt spacing on chr7:5530000-5531000*\n"
        "- *design 6 pairs on NM_000546.6 in cy5*\n"
        "- *why did I only get 3?* · *why is pair 2 worse than pair 1?*\n"
        "- *screen against hg38* · *enforce the GC/Tm window*\n\n"
        "The chat changes the channel, the pair count, the spacing, the "
        "specificity screen and whether the GC/Tm window is enforced. It cannot "
        "widen the Tm or GC window itself: the pipeline ranks by distance from "
        "the window and rejects on it only when enforcement is on, so a request "
        "that changes none of these is answered with what would place more, "
        "rather than designed again.")

    st.subheader("Reading the probe map")
    st.markdown(
        "- **Ordered pairs**: each pair is a 51-nt footprint — a 25-nt left "
        "pad, a 1-nt gap, a 25-nt right pad. Click one to open its close-up.\n"
        "- **Previous run** (grey): the pairs of the last run on the same "
        "target, so a tweak shows as a shift.\n"
        "- **Best windows**: the 40 best-scoring windows, stacked into lanes "
        "where they overlap. Green ones were ordered.\n"
        "- **Penalty**: the score of every window along the target, lower is "
        "better (the axis is flipped so good windows are high). For RNA the "
        "pipeline also re-ranks by accessibility, so the order of pairs can "
        "differ from this base score.\n"
        "- **Close-up**: the target 5'→3' on top; P1L and P1R bound "
        "antiparallel beneath. `P1L = arm_left + rc(left pad)` and `P1R = "
        "rc(right pad) + arm_right`, so both arms hang off at the gap, where "
        "P2 can land only when both halves are bound. Probe bases are shaded "
        "red by how often they are paired in the probe's own folding "
        "ensemble, when folding was computed.")

    st.subheader("The numbers")
    st.markdown(
        "- **P(bound)** is the equilibrium fraction bound at the limiting "
        "step, on-target vs the worst signal-capable off-target site — a "
        "ranking number, not an absolute prediction.\n"
        "- **The GC/Tm window** is reported for context. The pipeline ranks on "
        "distance from it rather than rejecting on it, unless you ask the chat "
        "to *enforce the GC/Tm window* — at the lab's own window, zero windows "
        "at one real locus satisfy both, yet that locus has a working design.\n"
        "- **Self-folding** is how much of each primary is paired in its own "
        "fold. A tied-up arm cannot present half of P2's site.\n"
        "- **Expression weighting** scales each off-target gene by its "
        "abundance in the imaged cell type. It is uncalibrated, and the "
        "results say so when it was not evaluated.")

    st.subheader("Target types")
    st.markdown(
        "- **Genomic region**: repeats are soft-masked from RepeatMasker, so no "
        "pad is placed in repetitive DNA.\n"
        "- **RNA transcript**: screened against a **transcriptome** index — see "
        "`pipeline.RNA_SCREEN_POLICY` for why, and why a junction-spanning "
        "footprint does not map contiguously to a genome.\n"
        "- **Pasted sequence**: designed in the pasted coordinates, so there is "
        "no genome position. If a specificity index is selected, the target "
        "itself may be reported as an off-target hit, because its own gene is "
        "not known to the screen.\n"
        "- **Fusion junction**: one position — the gap sits on the breakpoint — "
        "so tiling does not apply. Both strands are designed.")
