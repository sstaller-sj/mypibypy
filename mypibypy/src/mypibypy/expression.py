"""How much of each off-target gene is actually in the cell.

An off-target site on a transcript at 0.1 TPM and one at 500 TPM are not the
same problem. The first is a handful of molecules per cell at most, and a spot
from it is rare; the second competes with the target in every cell. Counting
off-target genes treats them alike. This module supplies the missing number so
`specificity` can weight each off-target gene by its abundance in the cell type
being imaged.

WHERE THE NUMBERS COME FROM. A table you supply, measured in your cell type or
tissue -- bulk RNA-seq, a pseudobulked single-cell atlas, a GTEx tissue column.
Nothing is fetched or assumed: expression is cell-type specific, and a default
table would quietly be the wrong cell type for almost everyone. With no table
the weighting is NOT EVALUATED, and the screen scores exactly as it did before
this module existed.

Accepted layouts, detected from the header:

* two columns, gene then TPM, with or without a header row
* any delimited table with a gene column (``gene``, ``gene_name``,
  ``gene_symbol``, ``symbol``, ``gene_id``) and a ``tpm`` column -- RSEM's
  ``genes.results`` is one
* salmon ``quant.sf`` (``Name`` + ``TPM``): GENCODE-style transcript names are
  resolved to their gene, and isoforms are summed into one value per gene

HOW A TPM BECOMES A WEIGHT. `weight` maps TPM to 0..1 with a saturating curve,
``tpm / (tpm + half_weight_tpm)``: 0.1 TPM weighs 0.01, 10 TPM 0.5, 500 TPM
0.98. The half-weight point is a guess, exposed as a parameter. **Nothing here
is calibrated** -- no piFISH off-target spot rate has been tied to a measured
abundance -- and `EXPRESSION_IS_UNCALIBRATED` travels with every number.

A GENE MISSING FROM THE TABLE WEIGHS 1. Absent is not the same as unexpressed:
a table filtered to detected genes, a symbol renamed between annotation
releases, or a gene ID with a different version all look like a missing row.
Treating any of them as zero would turn an off-target into a clean report, so
they keep the full penalty and are listed by name.
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ENV_EXPRESSION = "MYPIBYPY_EXPRESSION"

#: TPM at which an off-target gene carries half its unweighted penalty.
DEFAULT_HALF_WEIGHT_TPM = 10.0

#: Below this TPM a surviving, signal-capable off-target gene no longer fails a
#: pair outright -- the verdict drops to ``review``, never to ``pass``.
DEFAULT_MIN_TPM = 1.0

#: Said out loud wherever a weighted number is reported.
EXPRESSION_IS_UNCALIBRATED = (
    "expression weighting is UNCALIBRATED: no piFISH off-target spot rate has "
    "been tied to a measured abundance, and the table is only as right as its "
    "match to the imaged cell type"
)

_GENE_COLUMNS = ("gene", "gene_name", "gene_symbol", "symbol", "gene_id", "geneid")
_TRANSCRIPT_COLUMNS = ("name", "transcript_id", "transcript")
_VERSION = re.compile(r"^(ENS[A-Z]*G\d+)\.\d+$", re.IGNORECASE)


def _keys(gene: str) -> list[str]:
    """Lookup keys for a gene: as given, and without an Ensembl version suffix."""
    g = gene.strip().lower()
    keys = [g]
    m = _VERSION.match(g)
    if m:
        keys.append(m.group(1).lower())
    return keys


@dataclass(frozen=True)
class ExpressionTable:
    """TPM per gene from one source, and the curve that turns TPM into weight."""

    source: str
    tpm: dict[str, float] = field(default_factory=dict)  # lowercase key -> TPM
    half_weight_tpm: float = DEFAULT_HALF_WEIGHT_TPM
    min_tpm: float = DEFAULT_MIN_TPM
    n_rows: int = 0

    def lookup(self, gene: str) -> float | None:
        """TPM for `gene`, or None when the table has no row for it."""
        if not gene:
            return None
        for k in _keys(gene):
            if k in self.tpm:
                return self.tpm[k]
        return None

    def weight(self, gene: str) -> float:
        """0..1 multiplier on an off-target penalty. 1.0 for an unknown gene."""
        tpm = self.lookup(gene)
        if tpm is None:
            return 1.0
        return tpm / (tpm + self.half_weight_tpm) if tpm > 0 else 0.0

    def below_floor(self, gene: str) -> bool:
        """Known, and expressed below `min_tpm`. Unknown genes are never below."""
        tpm = self.lookup(gene)
        return tpm is not None and tpm < self.min_tpm

    def describe(self) -> str:
        return (f"{self.source} ({self.n_rows:,} genes; weight = TPM/(TPM+"
                f"{self.half_weight_tpm:g}); floor {self.min_tpm:g} TPM)")


def _float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_expression(text: str, source: str = "<text>",
                     half_weight_tpm: float = DEFAULT_HALF_WEIGHT_TPM,
                     min_tpm: float = DEFAULT_MIN_TPM) -> ExpressionTable:
    """Parse an expression table from text. Raises ValueError if it has no usable rows."""
    from .specificity import gencode_gene

    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise ValueError(f"expression table {source} is empty")
    delim = "\t" if "\t" in lines[0] else ","
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))
    header = [h.strip().lower() for h in rows[0]]

    if len(header) >= 2 and _float(rows[0][1]) is not None:
        gene_col, tpm_col, transcripts, body = 0, 1, False, rows
    else:
        if "tpm" not in header:
            raise ValueError(f"expression table {source} has no TPM column "
                             f"(header: {rows[0]})")
        tpm_col = header.index("tpm")
        gene_col = next((header.index(c) for c in _GENE_COLUMNS if c in header), None)
        transcripts = gene_col is None
        if transcripts:
            gene_col = next((header.index(c) for c in _TRANSCRIPT_COLUMNS if c in header),
                            None)
        if gene_col is None:
            raise ValueError(f"expression table {source} has no gene or transcript "
                             f"column (header: {rows[0]})")
        body = rows[1:]

    tpm: dict[str, float] = {}
    for row in body:
        if len(row) <= max(gene_col, tpm_col):
            continue
        value = _float(row[tpm_col])
        name = row[gene_col].strip()
        if value is None or not name:
            continue
        # Transcript-level input: resolve GENCODE names to their gene and sum
        # isoforms, since off-target sites are reported per gene.
        if transcripts:
            name = gencode_gene(name) or name
        for k in _keys(name):
            tpm[k] = tpm.get(k, 0.0) + value
    if not tpm:
        raise ValueError(f"expression table {source} has no rows with a numeric TPM")
    # A versioned Ensembl ID is stored under both forms; count it once.
    n_genes = len({k for k in tpm if not _VERSION.match(k)})
    return ExpressionTable(source=source, tpm=tpm, half_weight_tpm=half_weight_tpm,
                           min_tpm=min_tpm, n_rows=n_genes)


def load_expression(path: str | Path, half_weight_tpm: float = DEFAULT_HALF_WEIGHT_TPM,
                    min_tpm: float = DEFAULT_MIN_TPM) -> ExpressionTable:
    """Load an expression table from a file. See the module docstring for layouts."""
    p = Path(path)
    return parse_expression(p.read_text(), source=str(p),
                            half_weight_tpm=half_weight_tpm, min_tpm=min_tpm)


def resolve_expression(path: str | Path | None = None, **kwargs) -> ExpressionTable | None:
    """`path`, else ``$MYPIBYPY_EXPRESSION``, else None (weighting not evaluated)."""
    path = path or os.environ.get(ENV_EXPRESSION)
    return load_expression(path, **kwargs) if path else None
