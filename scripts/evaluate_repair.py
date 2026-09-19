#!/usr/bin/env python3
"""Evaluate frozen retrieval traces against separate gold evidence, entirely offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "SRgraphrag"))
from evaluation.repair_eval import evaluate_repair, load_gold, load_trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, help="Frozen retrieval_trace.jsonl")
    parser.add_argument("--gold", required=True, type=Path, help="Separate ordered gold JSON")
    parser.add_argument("--baseline-trace", type=Path, help="Optional trace with exactly the same query order")
    parser.add_argument("--baseline-mode", choices=("final", "round1"), default="final", help="Use round1 for a PPR single-round baseline")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Paired bootstrap resamples; zero disables CI")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, help="Write JSON report here; default prints to stdout")
    parser.add_argument("--force", action="store_true", help="Replace an existing report")
    args = parser.parse_args(argv)
    if args.output is not None:
        sources = [path.resolve() for path in (args.trace, args.gold, args.baseline_trace) if path is not None]
        if args.output.resolve() in sources:
            parser.error("Output must not overwrite trace or gold inputs")
        if args.output.exists() and not args.force:
            parser.error("Output exists; pass --force or choose another path")
    try:
        report = evaluate_repair(
            load_trace(args.trace), load_gold(args.gold),
            load_trace(args.baseline_trace) if args.baseline_trace else None,
            baseline_mode=args.baseline_mode, bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        )
        content = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output is None:
            print(content, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
            print(json.dumps({"output": str(args.output.resolve()), "total_queries": report["total_queries"]}))
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(1, f"Unable to evaluate repair trace: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
