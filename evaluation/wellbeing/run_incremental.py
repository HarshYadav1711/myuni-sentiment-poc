"""Incremental / checkpointed dual-head eval (fallback if a long run dies).

Does NOT replace the in-flight calibrate_margins process.
Use only to resume AFTER a failure, or for faster staged runs:

  python -m evaluation.wellbeing.run_incremental --phase collect
  python -m evaluation.wellbeing.run_incremental --phase search --freeze-config
  python -m evaluation.wellbeing.run_incremental --phase ak
  python -m evaluation.wellbeing.run_incremental --phase fh40
  python -m evaluation.wellbeing.run_incremental --phase v2

Checkpoints under evaluation/wellbeing/_checkpoints/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from evaluation.wellbeing.calibrate_margins import (  # noqa: E402
    collect_predictions,
    development_cases,
    evaluate_cases,
    freeze_config,
    search_thresholds,
)
from evaluation.wellbeing.final_holdout_cases import (  # noqa: E402
    WELLBEING_FINAL_HOLDOUT_CASES,
)
from evaluation.wellbeing.final_holdout_v2 import (  # noqa: E402
    WELLBEING_FINAL_HOLDOUT_V2_CASES,
)
from evaluation.wellbeing.run_eval import build_eval_cases  # noqa: E402
from src.wellbeing import WellbeingClassifier  # noqa: E402
from src.wellbeing.classifier import reset_wellbeing_classifier_for_tests  # noqa: E402

CKPT = ROOT / "evaluation" / "wellbeing" / "_checkpoints"


def _log(msg: str) -> None:
    print(f"[incremental] {msg}", flush=True)


def _save(name: str, payload: dict[str, Any]) -> Path:
    CKPT.mkdir(parents=True, exist_ok=True)
    path = CKPT / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log(f"wrote {path}")
    return path


def _load(name: str) -> dict[str, Any]:
    path = CKPT / name
    return json.loads(path.read_text(encoding="utf-8"))


def _clf() -> WellbeingClassifier:
    reset_wellbeing_classifier_for_tests()
    clf = WellbeingClassifier(enabled=True, include_legacy_binary_attribution=False)
    _log("loading model (CPU)…")
    t0 = time.perf_counter()
    clf.load()
    _log(f"loaded in {time.perf_counter() - t0:.1f}s is_loaded={clf.is_loaded}")
    return clf


def phase_collect() -> None:
    clf = _clf()
    clf.relevance_min_margin = 0.0
    clf.target_min_margin = 0.0
    clf.direct_self_min_score = 0.0
    clf.reported_other_block_score = 1.0
    cases = development_cases()
    _log(f"collecting dual-head raw scores on {len(cases)} development cases…")
    rows, elapsed, attr_calls = collect_predictions(clf, cases)
    _save(
        "01_collect.json",
        {
            "runtime_s": round(elapsed, 4),
            "attribution_calls": attr_calls,
            "case_count": len(cases),
            "rows": rows,
        },
    )


def phase_search(*, do_freeze: bool) -> None:
    data = _load("01_collect.json")
    search = search_thresholds(data["rows"])
    rec = search["recommended"]
    payload = {"search": search, "frozen_policy": {
        "WELLBEING_RELEVANCE_MIN_MARGIN": rec["relevance_min_margin"],
        "WELLBEING_TARGET_MIN_MARGIN": rec["target_min_margin"],
        "WELLBEING_DIRECT_SELF_MIN_SCORE": rec["direct_self_min_score"],
        "WELLBEING_REPORTED_OTHER_BLOCK_SCORE": rec[
            "reported_other_block_score"
        ],
    }}
    if do_freeze:
        path = freeze_config(
            relevance_min_margin=rec["relevance_min_margin"],
            target_min_margin=rec["target_min_margin"],
            direct_self_min_score=rec["direct_self_min_score"],
            reported_other_block_score=rec["reported_other_block_score"],
        )
        payload["frozen_into"] = str(path)
        _log(f"froze thresholds into {path}")
    _save("02_search.json", payload)
    _log(f"recommended={payload['frozen_policy']}")


def _policy_from_ckpt() -> dict[str, float]:
    return _load("02_search.json")["frozen_policy"]


def _eval_with_policy(
    clf: WellbeingClassifier,
    cases: list[dict[str, Any]],
    policy: dict[str, float],
    *,
    letter_key: bool = False,
) -> dict[str, Any]:
    return evaluate_cases(
        clf,
        cases,
        relevance_min_margin=policy["WELLBEING_RELEVANCE_MIN_MARGIN"],
        target_min_margin=policy["WELLBEING_TARGET_MIN_MARGIN"],
        direct_self_min_score=policy["WELLBEING_DIRECT_SELF_MIN_SCORE"],
        reported_other_block_score=policy[
            "WELLBEING_REPORTED_OTHER_BLOCK_SCORE"
        ],
        letter_key=letter_key,
    )


def phase_ak() -> None:
    policy = _policy_from_ckpt()
    clf = _clf()
    ak_cases = build_eval_cases()
    for c in ak_cases:
        exp = c.get("expected_human") or {}
        if "personal_wellbeing_eligible" in exp:
            c["expected_personal_eligibility"] = (
                "eligible" if exp["personal_wellbeing_eligible"] else "not_eligible"
            )
        elif exp.get("relevance") == "personal_wellbeing" and exp.get("target") == "self":
            c["expected_personal_eligibility"] = "eligible"
        elif "relevance_candidates" in exp or exp.get("relevance") in {
            "wellbeing_topic_only",
            "not_wellbeing_related",
            "ambiguous",
        }:
            c["expected_personal_eligibility"] = "not_eligible"
        else:
            c["expected_personal_eligibility"] = "uncertain"
    _log(f"A–K regression ({len(ak_cases)} cases)…")
    out = _eval_with_policy(clf, ak_cases, policy, letter_key=True)
    _save("03_ak.json", out)


def phase_fh40() -> None:
    policy = _policy_from_ckpt()
    clf = _clf()
    _log(f"FH40 regression ({len(WELLBEING_FINAL_HOLDOUT_CASES)} cases)…")
    out = _eval_with_policy(clf, list(WELLBEING_FINAL_HOLDOUT_CASES), policy)
    _save("04_fh40.json", out)


def phase_v2() -> None:
    policy = _policy_from_ckpt()
    clf = _clf()
    _log(f"final_holdout_v2 ONCE ({len(WELLBEING_FINAL_HOLDOUT_V2_CASES)} cases)…")
    out = _eval_with_policy(clf, list(WELLBEING_FINAL_HOLDOUT_V2_CASES), policy)
    _save("05_v2.json", out)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("collect", "search", "ak", "fh40", "v2", "all"),
    )
    parser.add_argument("--freeze-config", action="store_true")
    args = parser.parse_args(argv)
    phases = (
        ["collect", "search", "ak", "fh40", "v2"]
        if args.phase == "all"
        else [args.phase]
    )
    for phase in phases:
        _log(f"=== phase {phase} ===")
        if phase == "collect":
            phase_collect()
        elif phase == "search":
            phase_search(do_freeze=args.freeze_config or args.phase == "all")
        elif phase == "ak":
            phase_ak()
        elif phase == "fh40":
            phase_fh40()
        elif phase == "v2":
            phase_v2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
