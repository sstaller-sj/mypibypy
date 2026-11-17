"""The batch template and its reader: no Streamlit, no network."""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from mypibypy.batch_sheet import (
    EXCEL_CELL_MAX,
    MAX_TARGETS,
    SHEETS,
    BatchSheetError,
    read_batch,
    template_bytes,
)
from mypibypy.loci import FusionJunction, Locus, RnaTarget, load_fusions

SYSTEMS = ["cy3", "cy5"]
EXAMPLE_LOCI = Path(__file__).resolve().parents[1] / "src/mypibypy/loci.example.json"


def filled(rows: dict[str, list[tuple]], blank_rows: int = 0) -> bytes:
    """The template with `rows` typed into its sheets, after `blank_rows`."""
    wb = load_workbook(io.BytesIO(template_bytes(SYSTEMS)))
    for title, values in rows.items():
        ws = wb[title]
        for r, row in enumerate(values, start=2 + blank_rows):
            for c, value in enumerate(row, start=1):
                ws.cell(row=r, column=c, value=value)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


GENOMIC = ("promoter", "Actb", "mm10", "chr5", 142906255, 142906754, "cy5", "control")
RNA = ("cds", "TP53", "NM_000546.6", "cy3")


class TestTemplate:
    def test_it_has_instructions_then_one_sheet_per_target_kind(self):
        wb = load_workbook(io.BytesIO(template_bytes(SYSTEMS)))
        assert wb.sheetnames == ["Instructions", "Genomic regions",
                                 "RNA transcripts", "Fusion junctions",
                                 "Sequences"]

    def test_headers_are_the_loci_files_field_names(self):
        wb = load_workbook(io.BytesIO(template_bytes(SYSTEMS)))
        header = [c.value for c in wb["Genomic regions"][1]]
        assert header == ["name", "gene", "build", "chrom", "start", "end",
                          "system", "description"]
        assert all(c.comment for c in wb["Genomic regions"][1])

    def test_text_columns_are_formatted_as_text_so_genes_stay_genes(self):
        ws = load_workbook(io.BytesIO(template_bytes(SYSTEMS)))["Genomic regions"]
        assert ws["B2"].number_format == "@"            # gene
        assert ws["B200"].number_format == "@"
        assert ws["E2"].number_format != "@"            # start is a number

    def test_the_system_dropdown_lists_this_labs_systems(self):
        ws = load_workbook(io.BytesIO(template_bytes(SYSTEMS)))["Genomic regions"]
        lists = [dv.formula1 for dv in ws.data_validations.dataValidation
                 if dv.type == "list"]
        assert '"cy3,cy5"' in lists

    def test_the_example_rows_are_on_the_instructions_sheet_only(self):
        """An example left in a data sheet would be designed and ordered."""
        wb = load_workbook(io.BytesIO(template_bytes(SYSTEMS)))
        for spec in SHEETS:
            assert all(c.value is None for c in wb[spec.title][2])
        with pytest.raises(BatchSheetError, match="no targets"):
            read_batch(template_bytes(SYSTEMS), SYSTEMS)


class TestReading:
    def test_rows_become_the_targets_the_loci_file_would_give(self):
        fusion = load_fusions(EXAMPLE_LOCI)[0]
        rows = read_batch(filled({
            "Genomic regions": [GENOMIC],
            "RNA transcripts": [RNA],
            "Fusion junctions": [(fusion.name, fusion.five_prime_gene,
                                  fusion.three_prime_gene, fusion.sequence,
                                  fusion.breakpoint, "cy5", "cy3")],
        }), SYSTEMS)
        assert [r.problem for r in rows] == ["", "", ""]
        locus, rna, junction = (r.target for r in rows)
        assert isinstance(locus, Locus) and locus.interval.start == 142906255
        assert locus.description == "control"
        assert isinstance(rna, RnaTarget) and rna.accession == "NM_000546.6"
        assert isinstance(junction, FusionJunction)
        assert junction.sequence == fusion.sequence
        assert rows[2].system == "cy5 + cy3"
        assert [r.where for r in rows] == ["Genomic regions row 2",
                                           "RNA transcripts row 2",
                                           "Fusion junctions row 2"]

    def test_a_date_in_a_text_column_is_refused_not_guessed(self):
        """What Excel does to SEPT7 in a General-formatted cell."""
        row = ("promoter", dt.datetime(2026, 9, 7), "hg38", "chr1", 100, 200, "cy5")
        (got,) = read_batch(filled({"Genomic regions": [row]}), SYSTEMS)
        assert not got.ok
        assert "Excel turned gene into a date (07-Sep)" in got.problem

    def test_coordinates_typed_with_commas_are_read(self):
        row = ("promoter", "Actb", "mm10", "chr5", "142,906,255", "142,906,754", "cy5")
        (got,) = read_batch(filled({"Genomic regions": [row]}), SYSTEMS)
        assert got.ok and got.target.interval.end == 142906754

    def test_a_coordinate_that_is_not_a_number_is_refused(self):
        row = ("promoter", "Actb", "mm10", "chr5", "about 142.9M", 142906754, "cy5")
        (got,) = read_batch(filled({"Genomic regions": [row]}), SYSTEMS)
        assert "start must be a whole number" in got.problem

    def test_an_unknown_system_names_the_ones_there_are(self):
        (got,) = read_batch(filled({"RNA transcripts": [
            ("cds", "TP53", "NM_000546.6", "alexa647")]}), SYSTEMS)
        assert "system 'alexa647' is not one of the lab's systems (cy3, cy5)" \
            in got.problem

    def test_a_missing_required_cell_is_named(self):
        (got,) = read_batch(filled({"RNA transcripts": [
            ("cds", "TP53", None, "cy5")]}), SYSTEMS)
        assert got.problem == "accession is empty."

    def test_the_loci_loaders_own_checks_apply(self):
        """start > end is refused in the loader's words, once, without its
        location prefix -- the table already says which row."""
        row = ("promoter", "Actb", "mm10", "chr5", 200, 100, "cy5")
        (got,) = read_batch(filled({"Genomic regions": [row]}), SYSTEMS)
        assert got.problem == "locus 'promoter' has start > end."

    def test_a_breakpoint_outside_the_sequence_is_refused(self):
        (got,) = read_batch(filled({"Fusion junctions": [
            ("j", "A", "B", "ACGT" * 30, 500, "cy5")]}), SYSTEMS)
        assert "breakpoint 500 is outside its 120 nt sequence" in got.problem

    def test_line_breaks_in_a_pasted_sequence_are_dropped(self):
        (got,) = read_batch(filled({"Fusion junctions": [
            ("j", "A", "B", "ACGT" * 15 + "\n" + "ACGT" * 15, 60, "cy5")]}), SYSTEMS)
        assert got.ok and len(got.target.sequence) == 120

    def test_two_rows_that_would_share_oligo_names_are_refused(self):
        rows = read_batch(filled({"Genomic regions": [
            GENOMIC, ("promoter", "Actb", "mm10", "chr5", 1000, 2000, "cy3")]}),
            SYSTEMS)
        assert rows[0].ok
        assert "Same oligo names as Genomic regions row 2 (Actb_promoter)" \
            in rows[1].problem

    def test_blank_rows_are_skipped_and_row_numbers_are_excels(self):
        (got,) = read_batch(filled({"Genomic regions": [GENOMIC]}, blank_rows=3),
                            SYSTEMS)
        assert got.row == 5

    def test_one_bad_row_does_not_stop_the_rest(self):
        rows = read_batch(filled({"Genomic regions": [
            GENOMIC, ("bad", "Actb", "mm10", "chr5", "x", 10, "cy5")],
            "RNA transcripts": [RNA]}), SYSTEMS)
        assert [r.ok for r in rows] == [True, False, True]


class TestSequences:
    """The Form's pasted sequence, one row each, with the Form's checks."""

    BASES = "ACGTTGCA" * 20                                # 160 nt

    def read(self, sequence, gene="MYOD1"):
        (got,) = read_batch(filled({"Sequences": [
            ("enhancer", gene, sequence, "cy5")]}), SYSTEMS)
        return got

    def test_a_sequence_row_is_designed_as_the_form_designs_a_paste(self):
        got = self.read(self.BASES)
        assert got.ok and got.sequence == self.BASES
        assert isinstance(got.target, RnaTarget)
        assert got.target.accession == "pasted-sequence"
        assert got.target.gene == "MYOD1"            # the screen's on-target gene

    def test_a_fasta_header_line_breaks_and_rna_are_read(self):
        wrapped = "\n".join(self.BASES[i:i + 60] for i in range(0, 160, 60))
        got = self.read(">MYOD1 core enhancer, hg38\n" + wrapped.replace("T", "U"))
        assert got.ok and got.sequence == self.BASES

    def test_a_character_that_is_not_a_base_is_refused_not_dropped(self):
        got = self.read(self.BASES[:80] + "N" + self.BASES[80:])
        assert "characters other than A, C, G, T and U (N)" in got.problem

    def test_too_short_for_one_pair(self):
        assert "is 40 nt; a pair needs at least 51" in self.read("ACGT" * 10).problem

    def test_a_cell_exactly_at_excels_limit_was_probably_cut_off(self):
        got = self.read(("ACGT" * EXCEL_CELL_MAX)[:EXCEL_CELL_MAX])
        assert "Excel has probably cut off its end" in got.problem
        assert self.read(("ACGT" * EXCEL_CELL_MAX)[:EXCEL_CELL_MAX - 1]).ok

    def test_its_oligo_names_must_not_collide_with_another_sheets(self):
        rows = read_batch(filled({"Genomic regions": [
            ("enhancer", "MYOD1", "hg38", "chr11", 100, 600, "cy5")],
            "Sequences": [("enhancer", "MYOD1", self.BASES, "cy3")]}), SYSTEMS)
        assert rows[0].ok
        assert "Same oligo names as Genomic regions row 2" in rows[1].problem


class TestRefusingTheFile:
    def test_a_file_that_is_not_a_workbook(self):
        with pytest.raises(BatchSheetError, match="not an Excel workbook"):
            read_batch(b"name,gene\npromoter,Actb\n", SYSTEMS)

    def test_a_workbook_without_the_templates_sheets(self):
        wb, buf = Workbook(), io.BytesIO()
        wb.save(buf)
        with pytest.raises(BatchSheetError, match="None of the template's sheets"):
            read_batch(buf.getvalue(), SYSTEMS)

    def test_a_sheet_missing_a_column(self):
        wb = load_workbook(io.BytesIO(filled({"Genomic regions": [GENOMIC]})))
        wb["Genomic regions"]["D1"] = None              # chrom
        buf = io.BytesIO()
        wb.save(buf)
        with pytest.raises(BatchSheetError, match="has no chrom column"):
            read_batch(buf.getvalue(), SYSTEMS)

    def test_more_targets_than_one_upload_takes(self):
        rows = [(f"r{i}", "Actb", "mm10", "chr5", 100 * i + 1, 100 * i + 60, "cy5")
                for i in range(MAX_TARGETS + 1)]
        with pytest.raises(BatchSheetError, match=f"More than {MAX_TARGETS}"):
            read_batch(filled({"Genomic regions": rows}), SYSTEMS)
