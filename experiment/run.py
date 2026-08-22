#!/usr/bin/env python3
"""CLI for controlled POC experiment runs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.manifest import load_manifest
from experiment.report import write_experiment_outputs
from experiment.runner import run_experiment
from src.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled MyUni POC experiment from a manifest. "
            "Exports JSON, CSV, and Markdown report."
        ),
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to experiment manifest JSON",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for results.json, results.csv, report.md",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    result = run_experiment(manifest, manifest_path=str(manifest_path.resolve()))
    paths = write_experiment_outputs(result, args.out)

    summary = {
        "experiment_id": result.experiment_id,
        "n_samples": len(result.samples),
        "n_ok": result.aggregates.get("n_ok"),
        "n_errors": result.aggregates.get("n_errors"),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result.aggregates.get("n_errors", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
