#!/usr/bin/env python3
"""Where do the lab's ordered designs rank under each scoring?

Reads order workbooks, finds every design sheet with a footprint to rank (fusion
junctions are fixed and skipped), and reports for each ordered pair its rank
among the region's windows, its cost against the best, and whether the tiler
would pick it. The point is to see what a scoring change does to designs that
were ordered -- and, with --outcomes, to ones that imaged -- before it becomes
the default. See `mypibypy.benchmark`.

NOTHING identifying is printed by default: no sequences, no coordinates, no gene
names -- designs are shown by index. --names shows sheet names, for reading
locally. Run:

    export SSL_CERT_FILE=/path/to/roots.pem          # if behind a TLS proxy
    python scripts/benchmark_ranking.py ../2*.xlsx
    python scripts/benchmark_ranking.py ../2*.xlsx --outcomes outcomes.json --names

`--outcomes` is a JSON object of pair label -> "worked" / "failed" (a label is
the sheet name, or "sheet #n" for a tiled pair). Genomic regions are masked with
RepeatMasker from UCSC, as the pipeline masks them; --no-rmsk ranks unmasked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mypibypy.benchmark import RANKERS, markdown, rank_design, read_workbook  # noqa: E402
from mypibypy.sequence import rmsk_hits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workbooks", nargs="+", type=Path)
    ap.add_argument("--scoring", action="append", choices=sorted(RANKERS),
                    help="scoring(s) to compare; default: all")
    ap.add_argument("--outcomes", type=Path, help="JSON of pair label -> imaging result")
    ap.add_argument("--names", action="store_true", help="show sheet names")
    ap.add_argument("--no-rmsk", action="store_true",
                    help="do not fetch RepeatMasker; rank genomic regions unmasked")
    args = ap.parse_args()

    outcomes = json.loads(args.outcomes.read_text()) if args.outcomes else {}
    rows, problems, skipped = [], [], []
    for path in args.workbooks:
        designs = read_workbook(path)
        if not designs:
            skipped.append(path.name if args.names else "a workbook")
        for design in designs:
            problems += [p if args.names else "a pad is not at its stated footprint"
                         for p in design.problems]
            rmsk = None
            if design.kind == "genomic" and not args.no_rmsk:
                rmsk = rmsk_hits(design.interval)
            for scoring in args.scoring or sorted(RANKERS):
                rows += rank_design(design, scoring, rmsk)

    print(markdown(rows, outcomes, names=args.names))
    print(f"\n{len(rows)} ordered pair(s) ranked; footprints are 1-based within "
          f"each region. Rank 1 is the window the tiler takes first.")
    if skipped:
        print(f"No rankable design in: {', '.join(skipped)} (fusion junctions are fixed).")
    for p in problems:
        print(f"PROBLEM: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
