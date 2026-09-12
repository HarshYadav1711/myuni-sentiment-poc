"""Calibrate dual-head attribution policy on development sets (Phase 4A.5).

Order:
1) collect dual-head predictions on development cases
2) search DIRECT_SELF_MIN_SCORE + REPORTED_OTHER_BLOCK_SCORE
3) optionally freeze into src/config.py
4) run A–K + previous FH40 as REGRESSION diagnostics (not unbiased holdout)
5) run final_holdout_v2.py EXACTLY ONCE after freeze

Development data (contaminated / used for fitting):
- calibration_cases.py
- attribution_dev_cases.py
- A–K fixtures (regression)
- previous final_holdout_cases.py (FH40) — failures influenced this design

Fresh holdout (untouched until after freeze):
- final_holdout_v2.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from evaluation.wellbeing.attribution_dev_cases import (  # noqa: E402
    WELLBEING_ATTRIBUTION_DEV_CASES,
    attribution_dev_eligibility_distribution,
)
from evaluation.wellbeing.calibration_cases import (  # noqa: E402
    WELLBEING_CALIBRATION_CASES,
    calibration_eligibility_distribution,
)
from evaluation.wellbeing.final_holdout_cases import (  # noqa: E402
    WELLBEING_FINAL_HOLDOUT_CASES,
    final_holdout_distribution,
)
from evaluation.wellbeing.final_holdout_v2 import (  # noqa: E402
    WELLBEING_FINAL_HOLDOUT_V2_CASES,
    final_holdout_v2_distribution,
)
from evaluation.wellbeing.run_eval import build_eval_cases  # noqa: E402
from src.wellbeing import WellbeingClassifier  # noqa: E402
from src.wellbeing.classifier import reset_wellbeing_classifier_for_tests  # noqa: E402
from src.wellbeing.policy import decide_eligibility  # noqa: E402
from src.wellbeing.schemas import (  # noqa: E402
    AttributionEvidence,
    SelfAttributionResult,
)

DIRECT_SELF_CANDIDATES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
REPORTED_OTHER_CANDIDATES = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def development_cases() -> list[dict[str, Any]]:
    """Development cases used for threshold search (no V2, no FH40 fitting)."""
    cases: list[dict[str, Any]] = []
    cases.extend(WELLBEING_CALIBRATION_CASES)
    cases.extend(WELLBEING_ATTRIBUTION_DEV_CASES)
    return cases


def _snapshot_prediction(result: Any) -> dict[str, Any]:
    ev = result.self_attribution.attribution_evidence
    legacy = result.self_attribution.legacy_binary
    return {
        "status": result.status,
        "relevance_label": result.relevance.label,
        "target_label": result.target.label,
        "relevance_margin": result.uncertainty.relevance_top1_top2_margin,
        "target_margin": result.uncertainty.target_top1_top2_margin,
        "attribution": {
            "status": result.self_attribution.status,
            "raw_label": result.self_attribution.raw_label,
            "final_label": result.self_attribution.final_label,
            "label": result.self_attribution.label,
            "scores": dict(result.self_attribution.scores),
            "top_score": result.self_attribution.top_score,
            "top1_top2_margin": result.self_attribution.top1_top2_margin,
            "error_code": result.self_attribution.error_code,
            "attribution_evidence": {
                "direct_self_score": ev.direct_self_score,
                "reported_other_score": ev.reported_other_score,
                "evidence_status": ev.evidence_status,
                "error_code": ev.error_code,
            },
            "legacy_binary": (
                None
                if legacy is None
                else {
                    "status": legacy.status,
                    "raw_label": legacy.raw_label,
                    "scores": dict(legacy.scores),
                    "top_score": legacy.top_score,
                    "top1_top2_margin": legacy.top1_top2_margin,
                    "error_code": legacy.error_code,
                }
            ),
        },
    }


def collect_predictions(
    clf: WellbeingClassifier,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, int]:
    texts = [c["text"] for c in cases]
    t0 = time.perf_counter()
    results = clf.classify_many(texts)
    elapsed = time.perf_counter() - t0
    rows = []
    for case, result in zip(cases, results):
        rows.append(
            {
                "id": case["id"],
                "bucket": case.get("bucket") or case.get("category"),
                "expected_personal_eligibility": case[
                    "expected_personal_eligibility"
                ],
                "prediction": _snapshot_prediction(result),
            },
        )
    return rows, elapsed, clf.last_attribution_call_count


def _raw_attribution_for_policy(prediction: dict[str, Any]) -> SelfAttributionResult:
    """Rebuild RAW dual-head attribution (pre-policy) for offline search."""
    attr = prediction["attribution"]
    ev_raw = attr.get("attribution_evidence") or {}
    scores = dict(attr.get("scores") or {})
    direct = ev_raw.get("direct_self_score")
    other = ev_raw.get("reported_other_score")
    if direct is None:
        direct = scores.get("direct_self_experience")
    if other is None:
        other = scores.get("reported_other_experience")
    evidence = AttributionEvidence(
        direct_self_score=None if direct is None else float(direct),
        reported_other_score=None if other is None else float(other),
        evidence_status=ev_raw.get("evidence_status") or (
            "ok" if attr["status"] == "ok" else "not_evaluated"
        ),
        error_code=ev_raw.get("error_code") or attr.get("error_code"),
    )
    return SelfAttributionResult(
        status=attr["status"],
        raw_label=None,
        final_label=None,
        label=None,
        scores={
            "direct_self_experience": float(direct or 0.0),
            "reported_other_experience": float(other or 0.0),
        },
        top_score=attr.get("top_score"),
        top1_top2_margin=attr.get("top1_top2_margin"),
        attribution_evidence=evidence,
        error_code=attr.get("error_code"),
    )


def apply_thresholds(
    prediction: dict[str, Any],
    *,
    relevance_min_margin: float,
    target_min_margin: float,
    direct_self_min_score: float,
    reported_other_block_score: float,
) -> dict[str, Any]:
    decision = decide_eligibility(
        classifier_status=prediction["status"],
        relevance_label=prediction["relevance_label"],
        target_label=prediction["target_label"],
        relevance_margin=prediction["relevance_margin"],
        target_margin=prediction["target_margin"],
        attribution=_raw_attribution_for_policy(prediction),
        relevance_min_margin=relevance_min_margin,
        target_min_margin=target_min_margin,
        direct_self_min_score=direct_self_min_score,
        reported_other_block_score=reported_other_block_score,
    )
    return {
        "eligibility_status": decision.status,
        "personal_wellbeing_eligible": decision.personal_wellbeing_eligible,
        "eligibility_reasons": decision.reasons,
        "final_attribution": decision.finalized_attribution.final_label,
        "evidence_status": (
            decision.finalized_attribution.attribution_evidence.evidence_status
        ),
    }


def score_policy(
    rows: list[dict[str, Any]],
    *,
    relevance_min_margin: float,
    target_min_margin: float,
    direct_self_min_score: float,
    reported_other_block_score: float,
) -> dict[str, Any]:
    confusion = {
        "eligible": {"eligible": 0, "not_eligible": 0, "uncertain": 0},
        "not_eligible": {"eligible": 0, "not_eligible": 0, "uncertain": 0},
        "uncertain": {"eligible": 0, "not_eligible": 0, "uncertain": 0},
    }
    true_eligible = 0
    pred_eligible = 0
    true_positive = 0
    false_eligible = 0
    false_not_eligible = 0
    abstention = 0

    for row in rows:
        expected = row["expected_personal_eligibility"]
        applied = apply_thresholds(
            row["prediction"],
            relevance_min_margin=relevance_min_margin,
            target_min_margin=target_min_margin,
            direct_self_min_score=direct_self_min_score,
            reported_other_block_score=reported_other_block_score,
        )
        predicted = applied["eligibility_status"]
        confusion[expected][predicted] += 1
        if expected == "eligible":
            true_eligible += 1
            if predicted == "not_eligible":
                false_not_eligible += 1
        if predicted == "eligible":
            pred_eligible += 1
            if expected == "eligible":
                true_positive += 1
            else:
                false_eligible += 1
        if predicted == "uncertain":
            abstention += 1

    precision = true_positive / pred_eligible if pred_eligible else 1.0
    recall = true_positive / true_eligible if true_eligible else 0.0
    return {
        "relevance_min_margin": relevance_min_margin,
        "target_min_margin": target_min_margin,
        "direct_self_min_score": direct_self_min_score,
        "reported_other_block_score": reported_other_block_score,
        "eligible_precision": round(precision, 4),
        "eligible_recall": round(recall, 4),
        "false_eligible_count": false_eligible,
        "true_eligible_count": true_positive,
        "false_not_eligible_count": false_not_eligible,
        "abstention_count": abstention,
        "pred_eligible_count": pred_eligible,
        "confusion": confusion,
        # Prefer zero false eligible, then higher recall, then less restrictive.
        "sort_key": (
            false_eligible,
            -recall,
            direct_self_min_score + reported_other_block_score,
            abstention,
        ),
    }


def search_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for self_min, other_block in product(
        DIRECT_SELF_CANDIDATES,
        REPORTED_OTHER_CANDIDATES,
    ):
        scored.append(
            score_policy(
                rows,
                relevance_min_margin=0.0,
                target_min_margin=0.0,
                direct_self_min_score=self_min,
                reported_other_block_score=other_block,
            ),
        )
    scored.sort(key=lambda item: item["sort_key"])
    zero_false = [s for s in scored if s["false_eligible_count"] == 0]
    pareto: list[dict[str, Any]] = []
    best_recall_by_fe: dict[int, float] = {}
    for s in scored:
        fe = int(s["false_eligible_count"])
        rec = float(s["eligible_recall"])
        if fe not in best_recall_by_fe or rec > best_recall_by_fe[fe]:
            best_recall_by_fe[fe] = rec
            pareto.append(
                {
                    "false_eligible_count": fe,
                    "eligible_recall": s["eligible_recall"],
                    "eligible_precision": s["eligible_precision"],
                    "direct_self_min_score": s["direct_self_min_score"],
                    "reported_other_block_score": s["reported_other_block_score"],
                    "abstention_count": s["abstention_count"],
                },
            )
    if zero_false:
        zero_false.sort(
            key=lambda s: (
                -s["eligible_recall"],
                s["direct_self_min_score"] + s["reported_other_block_score"],
                s["abstention_count"],
            ),
        )
        recommended = zero_false[0]
        note = (
            "Selected zero-false-eligible dual-head policy with best eligible "
            "recall, then least restrictive thresholds."
        )
    else:
        recommended = scored[0]
        note = (
            "WARNING: no policy achieved zero false eligibility; "
            "reporting best available compromise."
        )
    return {
        "candidate_count": len(scored),
        "recommended": recommended,
        "top5": scored[:5],
        "pareto_like": pareto[:12],
        "zero_false_eligible_count": len(zero_false),
        "selection_note": note,
        "multi_label_mode": True,
        "multi_label_rationale": (
            "dual independent multi_label=True heads: direct_self_experience "
            "and reported_other_experience; final unclear is policy-derived."
        ),
        "conflict_rule": (
            "if direct_self >= self_min AND reported_other >= other_block → "
            "unclear (prefer abstention on mixed evidence)"
        ),
    }


# Back-compat alias for tests that still import search_margins.
def search_margins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return search_thresholds(rows)


def freeze_config(
    *,
    relevance_min_margin: float,
    target_min_margin: float,
    direct_self_min_score: float,
    reported_other_block_score: float,
) -> Path:
    config_path = ROOT / "src" / "config.py"
    text = config_path.read_text(encoding="utf-8")

    def _replace(name: str, value: float, body: str) -> str:
        pattern = rf"^{name} = .*$"
        repl = f"{name} = {value}"
        new_body, n = re.subn(pattern, repl, body, count=1, flags=re.M)
        if n != 1:
            raise RuntimeError(f"failed to freeze {name}")
        return new_body

    text = _replace("WELLBEING_RELEVANCE_MIN_MARGIN", relevance_min_margin, text)
    text = _replace("WELLBEING_TARGET_MIN_MARGIN", target_min_margin, text)
    text = _replace(
        "WELLBEING_DIRECT_SELF_MIN_SCORE",
        direct_self_min_score,
        text,
    )
    text = _replace(
        "WELLBEING_REPORTED_OTHER_BLOCK_SCORE",
        reported_other_block_score,
        text,
    )
    config_path.write_text(text, encoding="utf-8")
    return config_path


def evaluate_cases(
    clf: WellbeingClassifier,
    cases: list[dict[str, Any]],
    *,
    relevance_min_margin: float,
    target_min_margin: float,
    direct_self_min_score: float,
    reported_other_block_score: float,
    letter_key: bool = False,
) -> dict[str, Any]:
    clf.relevance_min_margin = relevance_min_margin
    clf.target_min_margin = target_min_margin
    clf.direct_self_min_score = direct_self_min_score
    clf.reported_other_block_score = reported_other_block_score

    t0 = time.perf_counter()
    results = clf.classify_many([c["text"] for c in cases])
    elapsed = time.perf_counter() - t0

    rows = []
    for case, result in zip(cases, results):
        ev = result.self_attribution.attribution_evidence
        row = {
            "id": case["id"],
            "bucket": case.get("bucket") or case.get("category"),
            "text": case["text"],
            "expected": case.get("expected_human")
            or {
                "expected_personal_eligibility": case[
                    "expected_personal_eligibility"
                ],
            },
            "expected_personal_eligibility": case.get(
                "expected_personal_eligibility",
            ),
            "status": result.status,
            "relevance": result.relevance.label,
            "target": result.target.label,
            "relevance_margin": result.uncertainty.relevance_top1_top2_margin,
            "target_margin": result.uncertainty.target_top1_top2_margin,
            "attribution_status": result.self_attribution.status,
            "attribution_final": result.self_attribution.final_label,
            "direct_self_score": ev.direct_self_score,
            "reported_other_score": ev.reported_other_score,
            "evidence_status": ev.evidence_status,
            "attribution_scores": dict(result.self_attribution.scores),
            "eligibility_status": result.eligibility_status,
            "personal_wellbeing_eligible": result.personal_wellbeing_eligible,
            "eligibility_reasons": list(result.eligibility_reasons),
            "selected_signals": [
                s.signal for s in result.signals if s.selected
            ],
        }
        if letter_key and "letter" in case:
            row["letter"] = case["letter"]
        rows.append(row)

    metric_rows = [
        {
            "expected_personal_eligibility": r["expected_personal_eligibility"],
            "prediction": {
                "status": r["status"],
                "relevance_label": r["relevance"],
                "target_label": r["target"],
                "relevance_margin": r["relevance_margin"],
                "target_margin": r["target_margin"],
                "attribution": {
                    "status": r["attribution_status"],
                    "raw_label": None,
                    "scores": r["attribution_scores"],
                    "top_score": None,
                    "top1_top2_margin": None,
                    "error_code": None,
                    "attribution_evidence": {
                        "direct_self_score": r["direct_self_score"],
                        "reported_other_score": r["reported_other_score"],
                        "evidence_status": r["evidence_status"],
                        "error_code": None,
                    },
                },
            },
        }
        for r in rows
        if r.get("expected_personal_eligibility")
    ]
    metrics = None
    bucket_metrics: dict[str, Any] = {}
    if metric_rows:
        metrics = score_policy(
            metric_rows,
            relevance_min_margin=relevance_min_margin,
            target_min_margin=target_min_margin,
            direct_self_min_score=direct_self_min_score,
            reported_other_block_score=reported_other_block_score,
        )
        by_bucket: dict[str, list[dict[str, Any]]] = {}
        for case, mrow in zip(cases, metric_rows):
            b = str(case.get("bucket") or case.get("category") or "other")
            by_bucket.setdefault(b, []).append(mrow)
        for b, brows in by_bucket.items():
            bucket_metrics[b] = score_policy(
                brows,
                relevance_min_margin=relevance_min_margin,
                target_min_margin=target_min_margin,
                direct_self_min_score=direct_self_min_score,
                reported_other_block_score=reported_other_block_score,
            )

    return {
        "total_s": round(elapsed, 4),
        "attribution_calls": clf.last_attribution_call_count,
        "metrics": metrics,
        "bucket_metrics": bucket_metrics,
        "rows": rows,
    }


def measure_attribution_latency(
    clf: WellbeingClassifier,
    texts: list[str],
) -> dict[str, Any]:
    """Compare dual-head vs legacy binary attribution latency on same texts."""
    # Force personal+self path by classifying full pipeline with dual only.
    clf.include_legacy_binary_attribution = False
    t0 = time.perf_counter()
    dual_results = clf.classify_many(texts)
    dual_s = time.perf_counter() - t0
    dual_calls = clf.last_attribution_call_count

    clf.include_legacy_binary_attribution = True
    t1 = time.perf_counter()
    _ = clf.classify_many(texts)
    both_s = time.perf_counter() - t1
    legacy_calls = clf.last_legacy_binary_call_count
    clf.include_legacy_binary_attribution = False

    return {
        "n_texts": len(texts),
        "dual_only_total_s": round(dual_s, 4),
        "dual_plus_legacy_total_s": round(both_s, 4),
        "dual_attribution_calls": dual_calls,
        "legacy_binary_calls": legacy_calls,
        "eligible_count_dual": sum(
            1 for r in dual_results if r.personal_wellbeing_eligible
        ),
        "note": (
            "Legacy binary is comparison-only; dual-head drives eligibility."
        ),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-config", action="store_true")
    parser.add_argument(
        "--skip-final-holdout",
        action="store_true",
        help="Dev only: skip V2 holdout (default runs it once after freeze)",
    )
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args(argv)

    reset_wellbeing_classifier_for_tests()
    clf = WellbeingClassifier(enabled=True, include_legacy_binary_attribution=False)
    # Collect with permissive gates so offline search sees raw dual evidence.
    clf.relevance_min_margin = 0.0
    clf.target_min_margin = 0.0
    clf.direct_self_min_score = 0.0
    clf.reported_other_block_score = 1.0  # never block during raw collect

    t_load = time.perf_counter()
    clf.load()
    cold_load_s = time.perf_counter() - t_load

    report: dict[str, Any] = {
        "model_id": clf.model_id,
        "cold_load_s": round(cold_load_s, 4),
        "dataset_roles": {
            "calibration_cases": "development/calibration (used to fit policy)",
            "attribution_dev_cases": "development attribution coverage",
            "A_K": "regression/development (contaminated; not unbiased holdout)",
            "final_holdout_cases_fh40": (
                "NOW development/regression — failures influenced dual-head "
                "design; not unbiased"
            ),
            "final_holdout_v2": "fresh holdout (evaluate once after freeze)",
        },
        "calibration_size": len(WELLBEING_CALIBRATION_CASES),
        "calibration_eligibility_distribution": (
            calibration_eligibility_distribution()
        ),
        "attribution_dev_size": len(WELLBEING_ATTRIBUTION_DEV_CASES),
        "attribution_dev_eligibility_distribution": (
            attribution_dev_eligibility_distribution()
        ),
        "fh40_regression_size": len(WELLBEING_FINAL_HOLDOUT_CASES),
        "fh40_distribution": final_holdout_distribution(),
        "final_holdout_v2_size": len(WELLBEING_FINAL_HOLDOUT_V2_CASES),
        "final_holdout_v2_distribution": final_holdout_v2_distribution(),
    }

    dev_cases = development_cases()
    rows, calib_s, attr_calls = collect_predictions(clf, dev_cases)
    search = search_thresholds(rows)
    report["development_runtime_s"] = round(calib_s, 4)
    report["development_attribution_calls"] = attr_calls
    report["development_case_count"] = len(dev_cases)
    report["search"] = search
    recommended = search["recommended"]

    if args.freeze_config:
        path = freeze_config(
            relevance_min_margin=recommended["relevance_min_margin"],
            target_min_margin=recommended["target_min_margin"],
            direct_self_min_score=recommended["direct_self_min_score"],
            reported_other_block_score=recommended[
                "reported_other_block_score"
            ],
        )
        report["frozen_into"] = str(path)

    report["frozen_policy"] = {
        "WELLBEING_RELEVANCE_MIN_MARGIN": recommended["relevance_min_margin"],
        "WELLBEING_TARGET_MIN_MARGIN": recommended["target_min_margin"],
        "WELLBEING_DIRECT_SELF_MIN_SCORE": recommended["direct_self_min_score"],
        "WELLBEING_REPORTED_OTHER_BLOCK_SCORE": recommended[
            "reported_other_block_score"
        ],
        "conflict_rule": search["conflict_rule"],
    }

    # Latency comparison on a small personal+self-leaning sample.
    latency_texts = [
        c["text"]
        for c in WELLBEING_ATTRIBUTION_DEV_CASES
        if c["expected_personal_eligibility"] == "eligible"
    ][:8]
    report["attribution_latency_comparison"] = measure_attribution_latency(
        clf,
        latency_texts,
    )

    # A–K regression diagnostics (not holdout).
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

    report["A_K_regression"] = evaluate_cases(
        clf,
        ak_cases,
        relevance_min_margin=recommended["relevance_min_margin"],
        target_min_margin=recommended["target_min_margin"],
        direct_self_min_score=recommended["direct_self_min_score"],
        reported_other_block_score=recommended["reported_other_block_score"],
        letter_key=True,
    )

    # Previous FH40 as regression only (not for fitting after this pass).
    fh40_cases = [
        {
            **c,
            "expected_personal_eligibility": c["expected_personal_eligibility"],
        }
        for c in WELLBEING_FINAL_HOLDOUT_CASES
    ]
    report["fh40_regression"] = evaluate_cases(
        clf,
        fh40_cases,
        relevance_min_margin=recommended["relevance_min_margin"],
        target_min_margin=recommended["target_min_margin"],
        direct_self_min_score=recommended["direct_self_min_score"],
        reported_other_block_score=recommended["reported_other_block_score"],
    )

    if not args.skip_final_holdout:
        report["final_holdout_v2"] = evaluate_cases(
            clf,
            WELLBEING_FINAL_HOLDOUT_V2_CASES,
            relevance_min_margin=recommended["relevance_min_margin"],
            target_min_margin=recommended["target_min_margin"],
            direct_self_min_score=recommended["direct_self_min_score"],
            reported_other_block_score=recommended[
                "reported_other_block_score"
            ],
        )

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
