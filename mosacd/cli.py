from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

from .api import discover
from .skeleton import SUPPORTED_TESTS


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MosaCD: discover a causal graph from a CSV and variable descriptions."
    )
    parser.add_argument("--data", type=Path, required=True, help="CSV: rows are samples, columns are variables")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON with description and variables")
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--ci-test", choices=sorted(SUPPORTED_TESTS), default="chi_square")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max-cond-vars", type=int, default=3)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n-votes", type=int, default=5, help="Votes per edge per prompt order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-completion", action="store_true", help="Keep unresolved edges undirected")
    parser.add_argument("--dry-run", action="store_true", help="Compute skeleton and request count without API calls")
    parser.add_argument("--base-url", help="Optional OpenAI-compatible API endpoint")
    parser.add_argument("--api-key-env", help="Environment variable holding the API key")
    args = parser.parse_args(argv)
    try:
        with args.data.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle))
        if len(set(header)) != len(header):
            raise ValueError("Duplicate CSV column names")
        data = pd.read_csv(args.data)
        metadata = json.loads(args.metadata.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        result = discover(
            data, description=metadata["description"], variables=metadata["variables"],
            ci_test=args.ci_test, alpha=args.alpha, max_cond_vars=args.max_cond_vars,
            model=args.model, temperature=args.temperature, n_votes=args.n_votes,
            seed=args.seed, complete=not args.no_completion, dry_run=args.dry_run,
            base_url=args.base_url, api_key_env=args.api_key_env,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as exc:
        parser.exit(1, f"mosacd: {exc}\n")
    if args.dry_run:
        print(
            f"Skeleton: {len(result['skeleton_edges'])} edges; full run: "
            f"{result['expected_api_requests']} API requests. No API calls made."
        )
    print(f"Saved {args.output}", file=sys.stdout)
