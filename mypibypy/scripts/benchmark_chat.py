#!/usr/bin/env python3
"""Is the chat harness working on this machine, and can this model do the job?

Runs ten scripted chat turns -- a design request, a follow-up, questions about
the result, a screen request, a request the chat cannot act on -- through the
same path the Streamlit chat runs, against a synthetic sequence, and scores the
intent fields the model returned. Each turn is timed, each explain reply goes
through `reply_check`, and a change is designed again as the app would. See
`mypibypy.agent.battery`; `scripts/validate_agent.py` is the same idea for the
decision path.

NOTHING identifying is printed: the target is a seeded random sequence, the
model sees only a placeholder for it, and only names and numbers are shown. Run:

    python scripts/benchmark_chat.py                       # local model, pinned sampler
    python scripts/benchmark_chat.py --model qwen3.5:4b    # another pulled model
    python scripts/benchmark_chat.py --sampler modelfile   # the Modelfile's sampler
    python scripts/benchmark_chat.py --loop --json out.json

Exit status 0 when every turn came back as expected, 1 when one did not, 2 when
the model could not be reached at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mypibypy.agent.battery import (  # noqa: E402
    ModelfileSamplerClient,
    run_battery,
    run_loop,
)
from mypibypy.agent.client import OllamaDecisionClient  # noqa: E402
from mypibypy.systems import systems  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="an Ollama model tag; default: the client's")
    ap.add_argument("--host", help="Ollama host; default: the client's")
    ap.add_argument("--sampler", choices=["client", "modelfile"], default="client",
                    help="client: the pinned sampler in agent.client.SAMPLING; "
                         "modelfile: leave it to the model's Modelfile")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="seconds per model call (default 120)")
    ap.add_argument("--loop", action="store_true",
                    help="also run the Form's agent loop over the synthetic design")
    ap.add_argument("--json", type=Path, help="write the report here as JSON")
    args = ap.parse_args()

    cls = ModelfileSamplerClient if args.sampler == "modelfile" else OllamaDecisionClient
    client = cls(model=args.model, host=args.host, timeout=args.timeout)
    keys = tuple(sorted(systems()))
    if not keys:
        print("no amplifier systems configured; see the README", file=sys.stderr)
        return 2

    report = run_battery(client, system_keys=keys,
                         label=f"{client.name}, {args.sampler} sampler")
    print("\n".join(report.lines()))
    out = report.to_dict()
    if args.loop:
        demo = run_loop(client, system_keys=keys)
        print()
        print("\n".join(demo.lines()))
        out["loop"] = demo.to_dict()
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwritten: {args.json}")
    if report.unreachable:
        return 2
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    sys.exit(main())
