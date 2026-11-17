"""The batch template: a workbook to fill in with targets, and its reader.

The Form designs one target per press, but the lab orders several at once
(four sets on one order sheet is normal). The CLI's `batch` reads them from the
loci file; this is the same thing for someone who would rather fill in a
spreadsheet than edit JSON. Three sheets mirror the loci file's three keys --
genomic regions, RNA transcripts, fusion junctions -- with the same field names,
so a row here and an entry there say the same thing, and the loci loader's own
checks build each target. A fourth, Sequences, is the Form's pasted sequence:
bases with no coordinates or accession, designed as the Form designs them.

Excel is the hazard. It turns SEPT7 or MARCH1 into a date as it is typed, and
the cell then holds a date, not the gene. The template formats every text
column as Text so that does not happen, and the reader refuses a date in a text
column rather than guessing which gene it was.

No Streamlit here, so it can be tested directly. Nothing is written to disk:
the template is built in memory and the upload is read from memory.
"""
from __future__ import annotations

import datetime as dt
import io
import zipfile
from dataclasses import dataclass

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from openpyxl.utils.exceptions import InvalidFileException
from openpyxl.worksheet.datavalidation import DataValidation

from .agent.intent import (
    MAX_SEQUENCE_NT,
    MIN_SEQUENCE_NT,
    clean_pasted_sequence,
    unreadable_sequence_chars,
)
from .loci import (
    LociNotConfigured,
    _fusion_from_dict,
    _locus_from_dict,
    _rna_from_dict,
)
from .workbook import B, H, HDR_FILL, TITLE

TEMPLATE_FILENAME = "mypibypy_batch_template.xlsx"
#: Targets one upload may hold. A tiled hg38 design takes a few seconds, so this
#: bounds a batch at a few minutes; split a larger order across uploads.
MAX_TARGETS = 50
#: Rows formatted as Text in the template. More than `MAX_TARGETS`, so a sheet
#: with a few blank or deleted rows still has formatted rows to type into.
TEMPLATE_ROWS = 200
#: Suggested in the build column's dropdown. Any UCSC build name is accepted.
BUILDS = ("hg38", "hg19", "mm10", "mm39")
#: The most characters an Excel cell holds. Excel cuts a longer paste to this
#: without a word, so a sequence exactly this long has probably lost its end.
EXCEL_CELL_MAX = 32_767
WRAP = Alignment(wrap_text=True, vertical="top")


@dataclass(frozen=True)
class Column:
    key: str                # the loci file's field name, used as the header
    help: str
    required: bool = True
    number: bool = False    # a whole number; otherwise text
    width: int = 14


@dataclass(frozen=True)
class SheetSpec:
    title: str
    kind: str               # "genomic" | "rna" | "fusion" | "sequence"
    columns: tuple[Column, ...]
    example: tuple          # one row, shown on the Instructions sheet only


_NAME = Column("name", "What this target is, e.g. promoter. With the gene it "
               "names the oligos (P1L_<gene>_<name>), so each gene and name "
               "pair must be unique in the upload.")
_SYSTEM = Column("system", "Amplifier system key from the lab's systems file "
                 "(the dropdown lists them).", width=12)
_DESCRIPTION = Column("description", "Optional note, copied into the order "
                      "workbook's design notes.", required=False, width=40)

SHEETS = (
    SheetSpec("Genomic regions", "genomic", (
        _NAME,
        Column("gene", "Gene symbol, e.g. Actb."),
        Column("build", "Genome build as UCSC names it, e.g. hg38 or mm10.",
               width=10),
        Column("chrom", "Chromosome as UCSC names it, e.g. chr5.", width=10),
        Column("start", "First base, 1-based inclusive (as UCSC shows it).",
               number=True),
        Column("end", "Last base, 1-based inclusive.", number=True),
        _SYSTEM, _DESCRIPTION,
    ), ("promoter", "Actb", "mm10", "chr5", 142906255, 142906754, "cy5",
        "positive control")),
    SheetSpec("RNA transcripts", "rna", (
        _NAME,
        Column("gene", "Gene symbol. The screen uses it to tell on-target hits "
               "from off-target ones."),
        Column("accession", "RefSeq (NM_...) or Ensembl (ENST...) transcript, "
               "with its version.", width=18),
        _SYSTEM, _DESCRIPTION,
    ), ("cds", "TP53", "NM_000546.6", "cy5", "")),
    SheetSpec("Fusion junctions", "fusion", (
        _NAME,
        Column("five_prime_gene", "The 5' partner gene.", width=16),
        Column("three_prime_gene", "The 3' partner gene.", width=16),
        Column("sequence", "The fused sequence, 5'->3', with enough of each "
               "partner either side of the breakpoint (A, C, G, T, N).",
               width=40),
        Column("breakpoint", "0-based index of the first base from the 3' "
               "partner, so the first <breakpoint> bases are the 5' partner.",
               number=True, width=12),
        Column("system", "Amplifier system for the sense pair.", width=12),
        Column("system_antisense", "Amplifier system for the antisense pair; "
               "blank uses the sense system.", required=False, width=16),
        _DESCRIPTION,
    ), ("demo", "GENEA", "GENEB", "ACGT... (about 120 nt either side)", 120,
        "cy5", "cy3", "")),
    SheetSpec("Sequences", "sequence", (
        _NAME,
        Column("gene", "Gene symbol, or a label if there is none. With a real "
               "symbol, a transcriptome screen ignores hits to the target's "
               "own gene; a genome screen may still report the target's own "
               "site, since a pasted sequence has no position."),
        Column("sequence", f"The bases, DNA or RNA, {MIN_SEQUENCE_NT} nt or "
               f"more. A FASTA header line and line breaks are fine. Anything "
               "other than A, C, G, T or U is refused rather than dropped, since "
               "dropping a base would shift every probe after it. A cell holds "
               f"at most {EXCEL_CELL_MAX:,} characters; use the Form for a "
               "longer sequence.", width=40),
        _SYSTEM, _DESCRIPTION,
    ), ("enhancer", "MYOD1", ">optional header, then the bases", "cy5", "")),
)


class BatchSheetError(ValueError):
    """The upload as a whole cannot be read. One bad row is not this."""


@dataclass(frozen=True)
class BatchRow:
    """One row of an uploaded template: a target ready to design, or why not."""

    sheet: str
    row: int                # the row number Excel shows
    kind: str
    label: str              # gene and name, as much as could be read
    target: object | None   # Locus | RnaTarget | FusionJunction
    system: str = ""
    problem: str = ""
    sequence: str = ""      # a Sequences row's bases; other kinds fetch theirs

    @property
    def ok(self) -> bool:
        return not self.problem

    @property
    def where(self) -> str:
        return f"{self.sheet} row {self.row}"


@dataclass(frozen=True)
class BatchLine:
    """What became of one uploaded row in a batch run. Every row gets one, the
    unreadable ones too, so nothing in the upload drops out of sight."""

    where: str
    label: str
    system: str
    results: tuple[int, ...]    # indices into the run's results; () if none
    requested: int = 0          # pairs asked for: two for a fusion, one a strand
    screen: str = ""            # what it was screened against, or why not
    problem: str = ""


# --- the template ------------------------------------------------------------

def template_bytes(system_keys: list[str]) -> bytes:
    """The empty template, with this lab's amplifier systems in its dropdowns."""
    wb = Workbook()
    _instructions_sheet(wb.active, system_keys)
    for spec in SHEETS:
        _data_sheet(wb.create_sheet(spec.title), spec, system_keys)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _instructions_sheet(ws, system_keys: list[str]) -> None:
    ws.title = "Instructions"
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 100
    ws["A1"] = "mypibypy batch template"
    ws["A1"].font = TITLE
    lines = [
        "Put one target per row on the sheet for its type, and leave the "
        "sheets you do not need empty. Then upload this file under Workspace "
        "> Batch.",
        "The pair count, the spacing and the specificity screen are chosen in "
        "the app for the whole upload.",
        f"Up to {MAX_TARGETS} targets per upload. Blank rows are skipped; any "
        "other sheet, this one included, is ignored.",
        "Text columns are formatted as Text, so Excel keeps a gene like SEPT7 "
        "as typed instead of turning it into a date. When pasting from "
        "elsewhere, paste values only (Paste Special > Values) to keep that "
        "formatting.",
        "Sequences rows are designed as the Form designs a pasted sequence: "
        "as a transcript, with one screen chosen in the app for all of them.",
        "Amplifier systems in this lab's systems file: "
        + (", ".join(system_keys) or "none") + ".",
    ]
    r = 3
    for line in lines:
        ws.cell(row=r, column=1, value="•").font = B
        cell = ws.cell(row=r, column=2, value=line)
        cell.font, cell.alignment = B, WRAP
        r += 1
    for spec in SHEETS:
        r += 1
        ws.cell(row=r, column=1, value=spec.title).font = H
        r += 1
        for col, example in zip(spec.columns, spec.example):
            ws.cell(row=r, column=1, value=col.key).font = H
            note = ("" if col.required else "Optional. ") + col.help
            if example not in ("", None):
                note += f" Example: {example}"
            cell = ws.cell(row=r, column=2, value=note)
            cell.font, cell.alignment = B, WRAP
            r += 1


def _data_sheet(ws, spec: SheetSpec, system_keys: list[str]) -> None:
    for j, col in enumerate(spec.columns, 1):
        cell = ws.cell(row=1, column=j, value=col.key)
        cell.font, cell.fill = H, HDR_FILL
        cell.comment = Comment(("" if col.required else "Optional. ") + col.help,
                               "mypibypy", width=300, height=110)
        letter = get_column_letter(j)
        ws.column_dimensions[letter].width = col.width
        if not col.number:
            for r in range(2, TEMPLATE_ROWS + 2):
                ws.cell(row=r, column=j).number_format = "@"
        span = f"{letter}2:{letter}{TEMPLATE_ROWS + 1}"
        if col.key in ("system", "system_antisense") and system_keys:
            choices = ",".join(system_keys)
            if len(choices) <= 255 and "," not in "".join(system_keys):
                dv = DataValidation(type="list", formula1=f'"{choices}"',
                                    allow_blank=True, showErrorMessage=True,
                                    errorTitle="Unknown amplifier system",
                                    error=f"Use one of: {choices}.")
                dv.add(span)
                ws.add_data_validation(dv)
        elif col.key == "build":
            # A suggestion list, not a rule: any UCSC build is accepted.
            dv = DataValidation(type="list", formula1=f'"{",".join(BUILDS)}"',
                                allow_blank=True, showErrorMessage=False)
            dv.add(span)
            ws.add_data_validation(dv)
        elif col.number:
            dv = DataValidation(type="whole", operator="greaterThanOrEqual",
                                formula1="0", allow_blank=True,
                                showErrorMessage=True, errorTitle="Whole number",
                                error="Enter a whole number, without commas.")
            dv.add(span)
            ws.add_data_validation(dv)
    ws.freeze_panes = "A2"


# --- reading an upload -------------------------------------------------------

def read_batch(data: bytes, system_keys: list[str]) -> list[BatchRow]:
    """Every non-blank row of the template's sheets, in sheet then row order.

    Raises `BatchSheetError` for a file that is not a workbook, has none of the
    template's sheets, is missing a column, or holds more than `MAX_TARGETS`
    targets. A row that cannot be designed is returned with its `problem`, so
    the rest of the upload can still go ahead.
    """
    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except (InvalidFileException, zipfile.BadZipFile, KeyError, OSError) as exc:
        raise BatchSheetError(
            "This is not an Excel workbook (.xlsx). Download the template, fill "
            "it in, and save it as .xlsx.") from exc
    try:
        titles = {name.strip().casefold(): name for name in wb.sheetnames}
        found = [(spec, wb[titles[spec.title.casefold()]]) for spec in SHEETS
                 if spec.title.casefold() in titles]
        if not found:
            raise BatchSheetError(
                "None of the template's sheets are here ("
                + ", ".join(spec.title for spec in SHEETS)
                + "). Download the template and fill that in.")
        rows: list[BatchRow] = []
        for spec, ws in found:
            rows.extend(_read_sheet(spec, ws, system_keys))
            if len(rows) > MAX_TARGETS:
                raise BatchSheetError(
                    f"More than {MAX_TARGETS} targets in one upload. Split them "
                    "across two files.")
    finally:
        wb.close()
    if not rows:
        raise BatchSheetError("The template has no targets in it yet: every "
                              "sheet is empty below its header row.")
    return _mark_duplicates(rows)


def _read_sheet(spec: SheetSpec, ws, system_keys: list[str]) -> list[BatchRow]:
    it = ws.iter_rows(values_only=True)
    header = next(it, None) or ()
    index = {str(h).strip().casefold(): j for j, h in enumerate(header)
             if h is not None and str(h).strip()}
    missing = [c.key for c in spec.columns if c.required and c.key not in index]
    if missing:
        raise BatchSheetError(
            f"Sheet '{spec.title}' has no {', '.join(missing)} column. Keep the "
            "template's header row as it is.")
    out = []
    for r, values in enumerate(it, start=2):
        cells = {c.key: values[index[c.key]] if c.key in index
                 and index[c.key] < len(values) else None for c in spec.columns}
        if all(_blank(v) for v in cells.values()):
            continue
        out.append(_read_row(spec, r, cells, system_keys))
    return out


def _blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _read_row(spec: SheetSpec, r: int, cells: dict, system_keys: list[str]) -> BatchRow:
    where = f"{spec.title} row {r}"
    values, problems = {}, []
    for col in spec.columns:
        value, problem = _cell(col, cells[col.key])
        if problem:
            problems.append(problem)
        elif value is not None:
            values[col.key] = value
    if spec.kind == "fusion":
        label = (f"{values.get('five_prime_gene', '?')}::"
                 f"{values.get('three_prime_gene', '?')} {values.get('name', '?')}")
    else:
        label = f"{values.get('gene', '?')} {values.get('name', '?')}"
    for key in ("system", "system_antisense"):
        if key in values and values[key] not in system_keys:
            problems.append(f"{key} '{values[key]}' is not one of the lab's "
                            f"systems ({', '.join(system_keys) or 'none'})")
    system = values.get("system", "")
    bases = ""
    if spec.kind == "fusion" and "sequence" in values:
        values["sequence"] = "".join(values["sequence"].split())
    elif spec.kind == "sequence" and "sequence" in values:
        bases, problem = _pasted_bases(values.pop("sequence"))
        if problem:
            problems.append(problem)
        values["accession"] = "pasted-sequence"     # as the Form names it
    if problems:
        return BatchRow(spec.title, r, spec.kind, label, None, system,
                        "; ".join(problems) + ".")
    build = {"genomic": _locus_from_dict, "rna": _rna_from_dict,
             "sequence": _rna_from_dict, "fusion": _fusion_from_dict}[spec.kind]
    try:
        target = build(values, where)
    except LociNotConfigured as exc:
        return BatchRow(spec.title, r, spec.kind, label, None, system,
                        str(exc).removeprefix(f"{where}: ") + ".")
    if spec.kind == "fusion":
        system = (system if not target.system_antisense
                  or target.system_antisense == system
                  else f"{system} + {target.system_antisense}")
    return BatchRow(spec.title, r, spec.kind, label, target, system,
                    sequence=bases)


def _pasted_bases(text: str) -> tuple[str, str]:
    """(bases, why not) for a Sequences cell, with the Form's own checks."""
    if len(text) == EXCEL_CELL_MAX:
        return "", (f"the sequence is exactly {EXCEL_CELL_MAX:,} characters, "
                    "the most an Excel cell holds, so Excel has probably cut "
                    "off its end; use the Form for a sequence this long")
    bad = unreadable_sequence_chars(text)
    if bad:
        return "", (f"the sequence has characters other than A, C, G, T and U "
                    f"({bad}); fix or remove them, since dropping them would "
                    "shift the sequence")
    bases = clean_pasted_sequence(text)
    if len(bases) < MIN_SEQUENCE_NT:
        return "", (f"the sequence is {len(bases)} nt; a pair needs at least "
                    f"{MIN_SEQUENCE_NT}")
    if len(bases) > MAX_SEQUENCE_NT:
        return "", (f"the sequence is {len(bases):,} nt, more than the "
                    f"{MAX_SEQUENCE_NT:,} one design takes")
    return bases, ""


def _cell(col: Column, value) -> tuple[object | None, str]:
    """(the value as the loci loader wants it, or None if blank; why not)."""
    if _blank(value):
        return None, ("" if not col.required else f"{col.key} is empty")
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        shown = (value.strftime("%d-%b") if isinstance(value, dt.date)
                 else str(value))
        return None, (f"Excel turned {col.key} into a date ({shown}); format the "
                      "column as Text and type it again")
    if col.number:
        if isinstance(value, bool):
            return None, f"{col.key} must be a whole number"
        if isinstance(value, float) and value.is_integer():
            return int(value), ""
        if isinstance(value, int):
            return value, ""
        text = str(value).replace(",", "").replace(" ", "").strip()
        if text.isdigit():
            return int(text), ""
        return None, f"{col.key} must be a whole number, got '{value}'"
    if isinstance(value, float) and value.is_integer():
        value = int(value)          # chrom typed as 5, read back as 5.0
    if col.key == "sequence":       # kept whole: a FASTA header is a line
        return str(value), ""
    return str(value).strip(), ""


def _mark_duplicates(rows: list[BatchRow]) -> list[BatchRow]:
    """Refuse a second row that would give its oligos the same names as an
    earlier one: two P1L_<gene>_<name> on one order sheet cannot be told apart."""
    first: dict[str, BatchRow] = {}
    out = []
    for row in rows:
        stem = _oligo_stem(row)
        if row.ok and stem in first:
            row = BatchRow(row.sheet, row.row, row.kind, row.label, None, row.system,
                           f"Same oligo names as {first[stem].where} "
                           f"({stem}); give one a different name.")
        elif row.ok:
            first[stem] = row
        out.append(row)
    return out


def _oligo_stem(row: BatchRow) -> str:
    t = row.target
    if row.kind == "fusion":
        return getattr(t, "name", "")
    return f"{getattr(t, 'gene', '')}_{getattr(t, 'name', '')}"
