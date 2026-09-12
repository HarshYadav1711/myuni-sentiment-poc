"""Run A–K wellbeing classifier evaluation (content-level, not clinical).

Usage (from repo root, after model is cached):

    python -m evaluation.wellbeing.run_eval
    python -m evaluation.wellbeing.run_eval --batch
    python -m evaluation.wellbeing.run_eval --one-by-one
    python -m evaluation.wellbeing.run_eval --compare-latency

Reuses ``tests/wellbeing_fixtures.py`` for A–F human expectations.
G–K are diagnostic edge cases only.
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

from wellbeing_fixtures import WELLBEING_SEMANTIC_FIXTURES  # noqa: E402

from src.wellbeing import WellbeingClassifier  # noqa: E402
from src.wellbeing.classifier import reset_wellbeing_classifier_for_tests  # noqa: E402
from src.wellbeing.schemas import WellbeingClassificationResult  # noqa: E402

# Diagnostic edge cases (not committed as production fixtures).
EDGE_CASES: list[dict[str, Any]] = [
    {
        "id": "G_hate_movie",
        "letter": "G",
        "text": "I hate this movie.",
        "expected_human": {
            "relevance": "not_wellbeing_related",
            "personal_wellbeing_eligible": False,
            "selected_must_not_include": ["self_directed_negativity"],
        },
    },
    {
        "id": "H_campus_topics",
        "letter": "H",
        "text": (
            "Depression and anxiety are common mental health topics "
            "discussed on campus."
        ),
        "expected_human": {
            "relevance": "wellbeing_topic_only",
            "personal_wellbeing_eligible": False,
            "target_not": ["self"],
        },
    },
    {
        "id": "I_roommate_overwhelmed",
        "letter": "I",
        "text": (
            "My roommate told me she has been completely overwhelmed by exams."
        ),
        "expected_human": {
            "relevance_candidates": [
                "wellbeing_topic_only",
                "ambiguous",
            ],
            "target": "other_person",
            "personal_wellbeing_eligible": False,
            "selected_must_be_empty": True,
        },
    },
    {
        "id": "J_slept_better",
        "letter": "J",
        "text": (
            "I finally slept properly and I'm feeling much better this week."
        ),
        "expected_human": {
            "relevance": "personal_wellbeing",
            "target": "self",
            "signals_include": ["positive_wellbeing_or_recovery"],
            "personal_wellbeing_eligible": True,
        },
    },
    {
        "id": "K_assignment_killing",
        "letter": "K",
        "text": "This assignment is killing me lol.",
        "expected_human": {
            "relevance_candidates": [
                "ambiguous",
                "not_wellbeing_related",
            ],
            "personal_wellbeing_eligible": False,
            "notes": "figurative; do not require serious distress",
        },
    },
]


def _letter_for_fixture(fixture_id: str, index: int) -> str:
    return "ABCDEF"[index]


def build_eval_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for i, fix in enumerate(WELLBEING_SEMANTIC_FIXTURES):
        cases.append(
            {
                "id": fix["id"],
                "letter": _letter_for_fixture(fix["id"], i),
                "text": fix["text"],
                "expected_human": fix["expected_human"],
            },
        )
    cases.extend(EDGE_CASES)
    return cases


def _result_row(
    case: dict[str, Any],
    result: WellbeingClassificationResult,
    runtime_s: float,
) -> dict[str, Any]:
    return {
        "letter": case["letter"],
        "id": case["id"],
        "text": case["text"],
        "expected": case["expected_human"],
        "status": result.status,
        "predicted_relevance": result.relevance.label,
        "relevance_scores": result.relevance.scores,
        "relevance_margin": result.uncertainty.relevance_top1_top2_margin,
        "predicted_target": result.target.label,
        "target_scores": result.target.scores,
        "target_margin": result.uncertainty.target_top1_top2_margin,
        "raw_signal_scores": {
            item.signal: item.score for item in result.signals
        },
        "threshold_passed_signals": [
            item.signal for item in result.signals if item.threshold_passed
        ],
        "selected_personal_signals": [
            item.signal for item in result.signals if item.selected
        ],
        "personal_wellbeing_eligible": result.personal_wellbeing_eligible,
        "eligibility_reasons": list(result.eligibility_reasons),
        "runtime_s": round(runtime_s, 4),
    }


def _relevance_match(expected: dict[str, Any], predicted: Optional[str]) -> bool:
    if "relevance" in expected:
        return predicted == expected["relevance"]
    if "relevance_candidates" in expected:
        return predicted in set(expected["relevance_candidates"])
    return True


def _target_applicable(expected: dict[str, Any]) -> bool:
    return "target" in expected or "target_candidates" in expected


def _target_match(expected: dict[str, Any], predicted: Optional[str]) -> bool:
    if "target" in expected:
        return predicted == expected["target"]
    if "target_candidates" in expected:
        return predicted in set(expected["target_candidates"])
    return True


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevance_exact = 0
    relevance_n = 0
    target_exact = 0
    target_n = 0
    false_personal_self = 0
    signal_selection_fps = 0

    for row in rows:
        expected = row["expected"]
        if "relevance" in expected or "relevance_candidates" in expected:
            relevance_n += 1
            if _relevance_match(expected, row["predicted_relevance"]):
                relevance_exact += 1
        if _target_applicable(expected):
            target_n += 1
            if _target_match(expected, row["predicted_target"]):
                target_exact += 1

        want_eligible = expected.get("personal_wellbeing_eligible")
        if want_eligible is False and row["personal_wellbeing_eligible"]:
            false_personal_self += 1
        if want_eligible is None:
            # Infer: eligible should be false unless expected personal+self.
            expect_personal = expected.get("relevance") == "personal_wellbeing"
            expect_self = expected.get("target") == "self"
            if not (expect_personal and expect_self) and row[
                "personal_wellbeing_eligible"
            ]:
                false_personal_self += 1

        if expected.get("selected_must_be_empty") and row[
            "selected_personal_signals"
        ]:
            signal_selection_fps += 1
        forbidden = set(expected.get("selected_must_not_include") or [])
        if forbidden.intersection(row["selected_personal_signals"]):
            signal_selection_fps += 1
        # Non-eligible cases should have empty selected personal signals.
        if (
            expected.get("personal_wellbeing_eligible") is False
            or (
                expected.get("relevance")
                not in (None, "personal_wellbeing")
                and expected.get("target") != "self"
            )
        ) and row["selected_personal_signals"]:
            # Count once if selected while clearly non-personal expectation.
            if expected.get("personal_wellbeing_eligible") is False:
                signal_selection_fps += 1

    runtimes = [float(r["runtime_s"]) for r in rows]
    return {
        "n_cases": len(rows),
        "relevance_exact_matches": relevance_exact,
        "relevance_evaluated": relevance_n,
        "target_exact_matches_when_applicable": target_exact,
        "target_evaluated": target_n,
        "false_personal_self_eligibility_count": false_personal_self,
        "signal_selection_false_positives": signal_selection_fps,
        "average_latency_s": (
            round(sum(runtimes) / len(runtimes), 4) if runtimes else None
        ),
        "total_latency_s": round(sum(runtimes), 4) if runtimes else None,
        "note": (
            "Counts are content-level evaluation helpers, not clinical metrics."
        ),
    }


def run_one_by_one(
    clf: WellbeingClassifier,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        t0 = time.perf_counter()
        result = clf.classify(case["text"])
        rows.append(_result_row(case, result, time.perf_counter() - t0))
    return rows


def run_batched(
    clf: WellbeingClassifier,
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    texts = [case["text"] for case in cases]
    t0 = time.perf_counter()
    results = clf.classify_many(texts)
    total = time.perf_counter() - t0
    per = total / max(len(cases), 1)
    rows = [
        _result_row(case, result, per)
        for case, result in zip(cases, results)
    ]
    return rows, total


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4 wellbeing classifier A–K evaluation",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Evaluate with classify_many (default)",
    )
    parser.add_argument(
        "--one-by-one",
        action="store_true",
        help="Evaluate with classify() per item",
    )
    parser.add_argument(
        "--compare-latency",
        action="store_true",
        help="Run one-by-one then batched and report speedup",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write full JSON report",
    )
    args = parser.parse_args(argv)

    cases = build_eval_cases()
    reset_wellbeing_classifier_for_tests()
    clf = WellbeingClassifier(enabled=True)

    t_load = time.perf_counter()
    clf.load()
    cold_load_s = time.perf_counter() - t_load

    report: dict[str, Any] = {
        "model_id": clf.model_id,
        "cold_load_s": round(cold_load_s, 4),
        "is_loaded": clf.is_loaded,
    }

    if args.compare_latency:
        t0 = time.perf_counter()
        one_rows = run_one_by_one(clf, cases)
        one_total = time.perf_counter() - t0
        batch_rows, batch_total = run_batched(clf, cases)
        report["mode"] = "compare_latency"
        report["one_by_one"] = {
            "total_s": round(one_total, 4),
            "avg_s": round(one_total / len(cases), 4),
            "summary": summarize(one_rows),
            "rows": one_rows,
        }
        report["batched"] = {
            "total_s": round(batch_total, 4),
            "avg_s": round(batch_total / len(cases), 4),
            "summary": summarize(batch_rows),
            "rows": batch_rows,
        }
        report["speedup_ratio"] = (
            round(one_total / batch_total, 3) if batch_total > 0 else None
        )
    elif args.one_by_one:
        t0 = time.perf_counter()
        rows = run_one_by_one(clf, cases)
        total = time.perf_counter() - t0
        report["mode"] = "one_by_one"
        report["total_s"] = round(total, 4)
        report["summary"] = summarize(rows)
        report["rows"] = rows
    else:
        rows, total = run_batched(clf, cases)
        report["mode"] = "batch"
        report["total_s"] = round(total, 4)
        report["summary"] = summarize(rows)
        report["rows"] = rows

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    return 0 if clf.is_loaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
