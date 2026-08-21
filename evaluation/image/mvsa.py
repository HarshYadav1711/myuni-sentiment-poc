"""MVSA local-file adapter for image+text sentiment evaluation.

MVSA is not bundled. Acquire data manually from the official project page, then
point this adapter at a prepared local JSONL index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from evaluation.common import EvalExample

PathLike = Union[str, Path]

MVSA_LABELS = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "pos": "positive",
    "neu": "neutral",
    "neg": "negative",
}


def map_mvsa_label(raw: Any) -> str:
    key = str(raw).strip().lower()
    if key not in MVSA_LABELS:
        raise ValueError(f"Unknown MVSA label: {raw!r}")
    return MVSA_LABELS[key]


def load_mvsa_jsonl(path: PathLike, *, require_images: bool = False) -> list[EvalExample]:
    """Load prepared MVSA-style JSONL.

    Expected fields per line:
    - id
    - text (caption/tweet text)
    - image_path (relative or absolute)
    - label (positive|neutral|negative) — gold for the pair / agreed label
    - optional text_label / image_label when present in source annotations

    This adapter does **not** invent splits; use whatever split file you prepared.
    """
    file_path = Path(path)
    base_dir = file_path.parent
    examples: list[EvalExample] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example_id = str(row.get("id") or f"mvsa-{i+1}")
            text = row.get("text")
            image_path = row.get("image_path")
            if not text or not image_path:
                raise ValueError(f"MVSA JSONL line {i+1}: text and image_path are required")
            label = map_mvsa_label(row["label"])
            resolved = Path(image_path)
            if not resolved.is_file():
                resolved = (base_dir / image_path).resolve()
            if require_images and not resolved.is_file():
                raise FileNotFoundError(f"MVSA image missing for {example_id}: {resolved}")
            examples.append(
                EvalExample(
                    example_id=example_id,
                    gold_label=label,
                    payload={
                        "text": str(text).strip(),
                        "image_path": str(resolved),
                    },
                    meta={
                        "source": "local_mvsa_jsonl",
                        "text_label": row.get("text_label"),
                        "image_label": row.get("image_label"),
                    },
                ),
            )
    return examples


def evaluate_mvsa_examples(
    examples: Sequence[EvalExample],
    predict_fn,
    *,
    limit: Optional[int] = None,
    split: str = "custom",
):
    from evaluation.common import apply_limit, build_eval_result, run_prediction_loop

    selected = apply_limit(examples, limit)
    predictions, errors = run_prediction_loop(selected, predict_fn)
    return build_eval_result(
        dataset="MVSA",
        split=split,
        examples=selected,
        predictions=predictions,
        errors=errors,
        notes=[
            "Official project page: "
            "https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/",
            "Dataset files are not redistributed by this repository; prepare a local JSONL index.",
        ],
    )
