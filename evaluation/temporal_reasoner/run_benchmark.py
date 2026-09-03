"""Thin CLI for temporal reasoner A/B benchmark (Phase 3B-A).

Does NOT change the production MyUni Space.
Does NOT download Qwen3-4B unless --allow-4b is passed (GPU sessions only).

Example (mocked / unit): use pytest instead.

Example (future ZeroGPU):
  python -m evaluation.temporal_reasoner.run_benchmark \\
    --models Qwen/Qwen3-1.7B Qwen/Qwen3-4B-Instruct-2507 \\
    --allow-4b --device cuda --output-dir outputs/reasoner_benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    TEMPORAL_REASONER_CANDIDATE_1_7B,
    TEMPORAL_REASONER_CANDIDATE_4B,
    TEMPORAL_REASONER_EVAL_SEED,
)
from src.temporal.benchmark.report import (  # noqa: E402
    write_human_review_markdown,
    write_results_csv,
    write_results_json,
)
from src.temporal.benchmark.runner import ReasonerBenchmarkRunner  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Temporal reasoner benchmark harness")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[TEMPORAL_REASONER_CANDIDATE_1_7B],
        help="Candidate model IDs (same frozen payloads for each)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=TEMPORAL_REASONER_EVAL_SEED)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "reasoner_benchmark"),
    )
    parser.add_argument(
        "--allow-4b",
        action="store_true",
        help="Permit loading Qwen3-4B (GPU sessions only; forbidden by default)",
    )
    parser.add_argument(
        "--fixtures",
        nargs="*",
        default=None,
        help="Optional subset of fixture IDs",
    )
    args = parser.parse_args(argv)

    forbid = set()
    if not args.allow_4b:
        forbid.add(TEMPORAL_REASONER_CANDIDATE_4B)
        for mid in args.models:
            if mid == TEMPORAL_REASONER_CANDIDATE_4B:
                print(
                    "Refusing to load Qwen3-4B without --allow-4b "
                    "(Phase 3B-A local safety).",
                    file=sys.stderr,
                )
                return 2

    runner = ReasonerBenchmarkRunner(
        model_ids=args.models,
        device=args.device,
        seed=args.seed,
        skip_missing_real=False,
        forbid_model_ids=forbid,
    )
    results, payloads = runner.run_all(fixture_ids=args.fixtures)
    out = Path(args.output_dir)
    write_results_json(results, out / "results.json")
    write_results_csv(results, out / "results.csv")
    write_human_review_markdown(
        results,
        payloads,
        out / "human_review.md",
        model_order=args.models,
    )
    print(f"Wrote benchmark artifacts under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
