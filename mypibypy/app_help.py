"""The Help and Tutorial pages: how the pipeline works, what the numbers mean,
setup status, and one worked example.

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
        f"- {':material/science:' if example else ':material/check_circle:'} "
        f"**Amplifier systems** — `{systems_path}`"
        + (" holds the **synthetic examples**. Probes designed with them are "
           "not orderable: copy `systems.example.json` to `systems.local.json` "
           "and put your real arms / P2 / P3 / P4 in it (see the README)."
           if example else ".")
        + f"\n- {':material/check_circle:' if screen_on else ':material/warning:'} "
        f"**Specificity screen** — "
        + ("a bowtie index is set." if screen_on else
           "off. Name your bowtie v1 indexes in `indexes.local.json` (copy "
           "`src/mypibypy/indexes.example.json`) and pick one under *Screen "
           "against* in the Form or by asking the chat (*screen against "
           "hg38*); or enter a path under *Settings* (top right). Until "
           "then every pair says NOT SCREENED and nothing establishes "
           "single-copy status.")
        + f"\n- {':material/block:' if client_kind == 'none' else ':material/smart_toy:'} "
        f"**Decision model** — "
        + (f"`{client_kind}`. The chat needs it to read requests; the form "
           "never does." if client_kind != "none" else
           "none. The form works; the chat cannot read requests."))

    st.subheader("The workspace")
    st.markdown(
        "- **Chat** and **Form** edit the same design. Change the pair count in "
        "the chat and the form opens on the new value; design from the form "
        "and the chat can keep adjusting that target.\n"
        "- **Batch** designs several targets at once. Download the template, "
        "fill in one row per target on the sheet for its type (genomic "
        "region, RNA transcript, fusion junction or pasted sequence), and "
        "upload it. The pair count, spacing and screens apply to the whole "
        "upload; genomic regions and fusions get a genome screen, transcripts "
        "a transcriptome one, and pasted sequences whichever you pick. A row "
        "that cannot be designed is listed with the reason and left out of "
        "the order, and the rest go ahead. An Excel cell holds at most "
        "32,767 characters, so paste a longer sequence into the Form.\n"
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


def render_tutorial(locus, n_pairs: int) -> bool:
    """One design from start to order, on the shipped example locus.

    Returns True when the user asks for the Form to be filled in with it.
    """
    iv = locus.interval
    region = f"{iv.chrom}:{iv.start}-{iv.end}"
    st.title("Tutorial: your first design")
    st.markdown(
        f"A worked example on the mouse *{locus.gene}* {locus.name} — "
        f"`{iv.build} {region}`, channel `{locus.system}`. *Actb* is a "
        "standard positive control: Tao et al., *Nat Commun* 14:443 (2023) "
        "imaged it with 35 pi target probes. The region is the first entry of "
        "`loci.example.json`.")
    clicked = st.button("Fill in the Form with this example", type="primary",
                        icon=":material/edit_note:", key="tutorial_fill")

    st.subheader("1. Fill in the Form")
    st.markdown(
        f"The button above opens the Workspace on the **Form** with the region, "
        f"the `{locus.system}` system and {n_pairs} pairs filled in. To type it "
        f"yourself: *Target type* **Genomic region**, *Genome build* "
        f"`{iv.build}`, *Region* `{region}`. Under *Labels (optional)*, enter "
        f"`{locus.gene}` and `{locus.name}` so the oligos are named for it.")

    st.subheader("2. Choose the specificity screen")
    st.markdown(
        "*Screen against* is the bowtie index every pair is checked against "
        f"for off-targets. For a mouse locus that should be a mouse genome "
        f"({iv.build}) index. With none, the design still runs, but every pair "
        "says **NOT SCREENED** and nothing establishes single-copy status: "
        "fine for learning the tool, not for an order.")

    st.subheader("3. Press Design probes")
    st.markdown(
        "The sequence is fetched and repeat-masked, every 51-nt window is "
        "scored, and the best non-overlapping ones are tiled into pairs — a "
        "few seconds, longer the first time a region is fetched.")

    st.subheader("4. Read the result")
    st.markdown(
        "- **The numbers at the top**: pairs placed, how many were screened, "
        "and the channel. A note says so when fewer pairs fit than you asked "
        "for.\n"
        "- **The probe map**: the ordered pairs along the region, the "
        "best-scoring windows, and the penalty along the target. Click a pair "
        "for its close-up.\n"
        "- **The table**: each pair's footprint, the Tm of each half, the "
        "screen's verdict, the worst off-target P(bound) and self-folding.\n\n"
        "*Help* explains every track and number.")

    st.subheader("5. Adjust it in the chat")
    st.markdown(
        "Switch to **Chat** and ask *make it 8 pairs* or *why did I get fewer "
        "than I asked for?*. The chat changes the same design; every design is "
        "kept in the run menu above the result, so you can compare them.")

    st.subheader("6. Download the order")
    st.markdown(
        "**Download order workbook (.xlsx)** holds the P1L/P1R sequences, "
        "named from the Form's labels (`GENE` and `target` if you left them). "
        "They are not written anywhere inside the repository.")
    return clicked
