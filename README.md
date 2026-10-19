# mypibypy

Design **piFISH** split-probe primary probes (P1L / P1R) for genomic targets and
fusion junctions, and emit an order workbook that drops straight into an
existing, validated P2 / P3 / P4 amplifier pool.

The Python package lives in [`mypibypy/`](mypibypy/).
**→ [Full documentation](mypibypy/README.md)**

```
target  ->  P1L + P1R (split primary)  ->  P2  ->  P3  ->  P4 (Cy signal)
```

A target footprint is 25 nt + a 1 nt gap + 25 nt. The two constant arms meet at
that gap and *together* form the single landing site for P2 — so P2 lands only
when both primaries are bound. That coincidence requirement is where piFISH gets
its specificity, and it is what this tool designs and checks against.

## Quick start

```bash
conda create -n pifish-tools -c conda-forge -c bioconda \
    python=3.11 blast bowtie bowtie2 samtools viennarna openpyxl pytest
conda activate pifish-tools
cd mypibypy && pip install -e . && pytest
```

```bash
python -m mypibypy batch  --loci loci.local.json -o order.xlsx
python -m mypibypy fusion --loci loci.local.json -o fusion_order.xlsx
python -m mypibypy rna    --loci loci.local.json -o rna_order.xlsx \
    --transcriptome-index /path/to/gencode/transcripts
python -m mypibypy batch  --loci loci.local.json -o order.xlsx --pairs 6

# or drive it from a browser
streamlit run mypibypy/app.py

# or from Claude Desktop / Claude Code, as MCP tools (no API key needed)
pip install -e "./mypibypy[mcp]" && python -m mypibypy mcp
```

## What is in here

| path | |
|---|---|
| [`mypibypy/`](mypibypy/) | the package, its tests, and [the real README](mypibypy/README.md) |
| [`mypibypy/src/mypibypy/`](mypibypy/src/mypibypy/) | design, screening and workbook modules |
| [`mypibypy/tests/`](mypibypy/tests/) | 436 tests; aligner-, folding- and network-backed ones skip when unavailable. No test makes an API call |
| [`mypibypy/scripts/`](mypibypy/scripts/) | threshold calibration and backend sensitivity benchmarks |

## No sequences in this repository

Amplifier systems, target loci, order workbooks and design-to-result records are
all lab-specific and often unpublished, so **none of them are committed**. Each
loads at runtime from a gitignored local file, with a synthetic placeholder
shipped so a fresh clone and the test suite still run end to end:

| what | runtime source | shipped placeholder |
|---|---|---|
| amplifier systems | `$MYPIBYPY_SYSTEMS` → `systems.local.json` | `systems.example.json` (synthetic) |
| target loci and fusions | `$MYPIBYPY_LOCI` → `loci.local.json` | `loci.example.json` (published mouse *Actb*) |
| design → result records | `$MYPIBYPY_OUTCOMES` → `outcomes.local.jsonl` | — |

Check `mypibypy.is_example()` before treating any output as orderable; the CLI
and the workbook both say so in writing when placeholders are in use.

## Two things worth knowing

**A screen that did not run never renders as a screen that passed.** Specificity
screening degrades to an explicit `NOT_SCREENED` status with a reason, which the
order workbook prints verbatim. Gate any specificity claim on
`SpecificityReport.screened`.

**Screening is pair-aware.** A genome hit for one half of a split probe cannot
recruit P2 and so cannot make a spot; two hits landing adjacent can. No
general-purpose probe tool draws that distinction, and it is the main reason this
package exists. See
[Pair-aware genome screening](mypibypy/README.md#pair-aware-genome-screening),
and [gene-level collapsing](mypibypy/README.md#screening-a-transcriptome-gene-level-collapsing)
for why a transcriptome needs more than that.

## Credit and license

Developed by Seth Staller, St. Jude Children's Research Hospital.

MIT licensed; copyright held by St. Jude Children's Research Hospital. See
[LICENSE](LICENSE), and [CITATION.cff](CITATION.cff) to cite this software.
Method: Tao *et al.*, "Multiplexed 3D analysis of immune states and niches by
pi-FISH-rainbow", *Nature Communications* **14**:443 (2023).
