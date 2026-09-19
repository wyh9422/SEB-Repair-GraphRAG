#!/usr/bin/env python3
"""Build a relation sidecar from an existing OpenIE cache, without model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# The leaf package is dependency-free even when the legacy top-level package is
# imported eagerly by older installations of this repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "SRgraphrag"))
from graph.relation_index import RelationIndex


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openie", required=True, type=Path, help="Existing OpenIE JSON (docs array or wrapper)")
    parser.add_argument("--output", type=Path, help="Destination sidecar JSON")
    parser.add_argument("--passage-ids", type=Path, help="Optional JSON array of current-corpus chunk IDs")
    parser.add_argument("--check-only", action="store_true", help="Build and report counts without writing a file")
    parser.add_argument("--force", action="store_true", help="Replace an existing output sidecar")
    args = parser.parse_args(argv)
    if not args.check_only and args.output is None:
        parser.error("--output is required unless --check-only is specified")
    if args.output is not None:
        if args.output.resolve() == args.openie.resolve() or args.output.suffix != ".json":
            parser.error("--output must be a separate .json sidecar, not the source cache or graph.pickle")
        if not args.check_only and args.output.exists() and not args.force:
            parser.error("Output exists; use a different path or explicitly pass --force")
    try:
        with args.openie.open(encoding="utf-8") as handle:
            records = json.load(handle)
        passage_ids = None
        if args.passage_ids is not None:
            with args.passage_ids.open(encoding="utf-8") as handle:
                passage_ids = json.load(handle)
            if not isinstance(passage_ids, list):
                raise ValueError("--passage-ids must contain a JSON array")
        index = RelationIndex.from_openie(records, passage_ids=passage_ids)
        if not args.check_only:
            index.save(args.output)
        print(json.dumps({
            "manifest": index.manifest.to_dict(),
            "output": None if args.check_only else str(args.output.resolve()),
            "check_only": args.check_only,
        }, ensure_ascii=False, sort_keys=True, indent=2))
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Unable to build relation index: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
