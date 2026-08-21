"""CMU-MOSI local adapter for video multimodal evaluation.

Continuous sentiment scores are preserved. An **explicit, documented** mapping
to 3-way labels is available and never applied silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from evaluation.common import EvalExample, apply_limit, build_eval_result, run_prediction_loop
from evaluation.metrics import continuous_metrics

PathLike = Union[str, Path]

# Documented POC 3-way conversion for MOSI continuous scores in [-3, +3].
# CMU MultimodalSDK MOSI README discusses binary (<0 vs >=0/>0) and 7-class (round).
# This POC mapping is explicit and configurable — not a silent transform.
DEFAULT_MOSI_3WAY = {
    "negative_below": 0.0,
    "positive_above": 0.0,
    "description": (
        "POC 3-way mapping for MOSI continuous labels in approx. [-3, +3]: "
        "score < 0 => negative; score > 0 => positive; score == 0 => neutral. "
        "See CMU-MultimodalSDK MOSI README for binary/7-class conventions used in papers."
    ),
}


def mosi_score_to_3way(
    score: float,
    *,
    negative_below: float = 0.0,
    positive_above: float = 0.0,
) -> str:
    """Explicit continuous→3-way conversion (must be documented at call sites)."""
    if score < negative_below:
        return "negative"
    if score > positive_above:
        return "positive"
    return "neutral"


def scale_mosi_to_unit(score: float) -> float:
    """Map MOSI [-3, 3] intensity to approximately [-1, 1] for continuous comparison."""
    return max(-1.0, min(1.0, float(score) / 3.0))


def load_mosi_jsonl(
    path: PathLike,
    *,
    apply_3way: bool = True,
    mapping: Optional[dict[str, Any]] = None,
) -> list[EvalExample]:
    """Load prepared MOSI-style JSONL.

    Expected fields: id, video_path|media_path, score (continuous), optional text.
    """
    cfg = {**DEFAULT_MOSI_3WAY, **(mapping or {})}
    file_path = Path(path)
    base_dir = file_path.parent
    examples: list[EvalExample] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example_id = str(row.get("id") or f"mosi-{i+1}")
            media = row.get("video_path") or row.get("media_path")
            if media is None:
                raise ValueError(f"MOSI JSONL line {i+1}: video_path/media_path required")
            if "score" not in row:
                raise ValueError(f"MOSI JSONL line {i+1}: continuous score required")
            score = float(row["score"])
            resolved = Path(media)
            if not resolved.is_file():
                resolved = (base_dir / media).resolve()
            gold_label = None
            if apply_3way:
                gold_label = mosi_score_to_3way(
                    score,
                    negative_below=float(cfg["negative_below"]),
                    positive_above=float(cfg["positive_above"]),
                )
            examples.append(
                EvalExample(
                    example_id=example_id,
                    gold_label=gold_label,
                    gold_score=score,
                    payload={
                        "media_path": str(resolved),
                        "text": row.get("text") or row.get("transcript"),
                    },
                    meta={
                        "source": "local_mosi_jsonl",
                        "continuous_range": "[-3, +3] typical for MOSI",
                        "unit_scaled_gold": scale_mosi_to_unit(score),
                        "three_way_applied": apply_3way,
                    },
                ),
            )
    return examples


def evaluate_mosi_examples(
    examples: Sequence[EvalExample],
    predict_fn,
    *,
    limit: Optional[int] = None,
    split: str = "custom",
    mapping: Optional[dict[str, Any]] = None,
):
    cfg = {**DEFAULT_MOSI_3WAY, **(mapping or {})}
    selected = apply_limit(examples, limit)
    predictions, errors = run_prediction_loop(selected, predict_fn)

    result = build_eval_result(
        dataset="CMU-MOSI",
        split=split,
        examples=selected,
        predictions=predictions,
        errors=errors,
        label_mapping=cfg,
        notes=[
            "Official SDK/docs: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK "
            "(see dataset standard_datasets/CMU_MOSI README).",
            "Paper reference commonly cited: Zadeh et al., MOSI (arXiv:1606.06259).",
            "Continuous metrics compare (gold_score/3) to model pred_score in [-1, +1].",
            "3-way labels use the explicit mapping in label_mapping (never silent).",
        ],
        compute_continuous=False,
    )

    unit_true: list[float] = []
    unit_pred: list[float] = []
    by_id = {ex.example_id: ex for ex in selected}
    for pred in predictions:
        ex = by_id.get(pred["example_id"])
        if ex is None or ex.gold_score is None or pred.get("pred_score") is None:
            continue
        unit_true.append(scale_mosi_to_unit(float(ex.gold_score)))
        unit_pred.append(float(pred["pred_score"]))
    if unit_true:
        result.continuous = continuous_metrics(unit_true, unit_pred)
    return result
