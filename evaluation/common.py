"""Shared evaluation helpers (I/O, sampling, reporting)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from evaluation.metrics import classification_metrics, continuous_metrics

PathLike = Union[str, Path]
PredictFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class EvalExample:
    """Normalized evaluation example used by all modality runners."""

    example_id: str
    gold_label: Optional[str] = None
    gold_score: Optional[float] = None
    payload: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    dataset: str
    split: str
    n_requested: int
    n_evaluated: int
    n_errors: int
    classification: Optional[dict[str, Any]] = None
    continuous: Optional[dict[str, Any]] = None
    label_mapping: Optional[dict[str, Any]] = None
    predictions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_limit(examples: Sequence[EvalExample], limit: Optional[int]) -> list[EvalExample]:
    if limit is None:
        return list(examples)
    if limit < 0:
        raise ValueError("limit must be >= 0")
    return list(examples[:limit])


def write_results(result: EvalResult, output_dir: PathLike) -> dict[str, Path]:
    """Write JSON summary + CSV predictions under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "metrics.json"
    csv_path = out / "predictions.csv"

    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [
        "example_id",
        "gold_label",
        "pred_label",
        "gold_score",
        "pred_score",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.predictions:
            writer.writerow({k: row.get(k) for k in fieldnames})
        for err in result.errors:
            writer.writerow(
                {
                    "example_id": err.get("example_id"),
                    "gold_label": err.get("gold_label"),
                    "pred_label": None,
                    "gold_score": err.get("gold_score"),
                    "pred_score": None,
                    "error": err.get("error"),
                },
            )

    return {"metrics_json": json_path, "predictions_csv": csv_path}


def format_console_summary(result: EvalResult) -> str:
    lines = [
        f"Dataset: {result.dataset} ({result.split})",
        f"Evaluated: {result.n_evaluated}/{result.n_requested} (errors={result.n_errors})",
    ]
    if result.label_mapping:
        lines.append(f"Label mapping: {json.dumps(result.label_mapping, ensure_ascii=False)}")
    if result.classification:
        c = result.classification
        lines.append(
            "3-way classification: "
            f"acc={c['accuracy']:.4f} "
            f"macroF1={c['f1_macro']:.4f} "
            f"weightedF1={c['f1_weighted']:.4f} "
            f"P_macro={c['precision_macro']:.4f} "
            f"R_macro={c['recall_macro']:.4f}",
        )
    if result.continuous:
        cont = result.continuous
        corr = cont["pearson_correlation"]
        corr_s = "nan" if corr != corr else f"{corr:.4f}"
        lines.append(
            f"Continuous: MAE={cont['mae']:.4f} Pearson={corr_s}",
        )
    for note in result.notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def run_prediction_loop(
    examples: Sequence[EvalExample],
    predict_fn: PredictFn,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Call ``predict_fn(payload)`` expecting keys pred_label and/or pred_score."""
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for ex in examples:
        try:
            out = predict_fn(ex.payload)
            predictions.append(
                {
                    "example_id": ex.example_id,
                    "gold_label": ex.gold_label,
                    "gold_score": ex.gold_score,
                    "pred_label": out.get("pred_label"),
                    "pred_score": out.get("pred_score"),
                    "raw": out.get("raw"),
                    "error": None,
                },
            )
        except Exception as exc:  # noqa: BLE001 — isolate examples
            errors.append(
                {
                    "example_id": ex.example_id,
                    "gold_label": ex.gold_label,
                    "gold_score": ex.gold_score,
                    "error": str(exc),
                },
            )
    return predictions, errors


def build_eval_result(
    *,
    dataset: str,
    split: str,
    examples: Sequence[EvalExample],
    predictions: Sequence[dict[str, Any]],
    errors: Sequence[dict[str, Any]],
    label_mapping: Optional[dict[str, Any]] = None,
    notes: Optional[Iterable[str]] = None,
    compute_continuous: bool = False,
) -> EvalResult:
    y_true = [p["gold_label"] for p in predictions if p.get("gold_label") is not None]
    y_pred = [p["pred_label"] for p in predictions if p.get("gold_label") is not None]
    # Keep pairs aligned when pred_label missing.
    paired_true: list[str] = []
    paired_pred: list[str] = []
    for p in predictions:
        if p.get("gold_label") is None or p.get("pred_label") is None:
            continue
        paired_true.append(str(p["gold_label"]))
        paired_pred.append(str(p["pred_label"]))

    classification = None
    if paired_true:
        classification = classification_metrics(paired_true, paired_pred)

    continuous = None
    if compute_continuous:
        ct: list[float] = []
        cp: list[float] = []
        for p in predictions:
            if p.get("gold_score") is None or p.get("pred_score") is None:
                continue
            ct.append(float(p["gold_score"]))
            cp.append(float(p["pred_score"]))
        if ct:
            continuous = continuous_metrics(ct, cp)

    return EvalResult(
        dataset=dataset,
        split=split,
        n_requested=len(examples),
        n_evaluated=len(predictions),
        n_errors=len(errors),
        classification=classification,
        continuous=continuous,
        label_mapping=label_mapping,
        predictions=list(predictions),
        errors=list(errors),
        notes=list(notes or []),
    )
