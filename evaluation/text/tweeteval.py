"""TweetEval sentiment adapter.

Uses local JSONL by default. Optional Hugging Face ``datasets`` loading is
available when the user explicitly requests it (network + license acceptance
are the caller's responsibility).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from evaluation.common import EvalExample

PathLike = Union[str, Path]

# Hugging Face card: https://huggingface.co/datasets/cardiffnlp/tweet_eval
# Labels for config "sentiment": 0=negative, 1=neutral, 2=positive
TWEETEVAL_LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
    "0": "negative",
    "1": "neutral",
    "2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


def map_tweeteval_label(raw: Any) -> str:
    if raw not in TWEETEVAL_LABEL_MAP:
        raise ValueError(f"Unknown TweetEval sentiment label: {raw!r}")
    return TWEETEVAL_LABEL_MAP[raw]


def load_tweeteval_jsonl(path: PathLike) -> list[EvalExample]:
    """Load local JSONL records: {id?, text, label}."""
    file_path = Path(path)
    examples: list[EvalExample] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"TweetEval JSONL line {i+1}: missing text")
            label = map_tweeteval_label(row["label"])
            example_id = str(row.get("id") or f"tweeteval-{i+1}")
            examples.append(
                EvalExample(
                    example_id=example_id,
                    gold_label=label,
                    payload={"text": text.strip()},
                    meta={"source": "local_jsonl", "path": str(file_path)},
                ),
            )
    return examples


def load_tweeteval_hf(split: str = "test") -> list[EvalExample]:
    """Optional loader via Hugging Face datasets (not used by unit tests).

    Requires: ``pip install datasets`` and network access.
    Dataset: ``cardiffnlp/tweet_eval`` config ``sentiment``.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face datasets is required for --tweeteval-hf. "
            "Install with: pip install datasets",
        ) from exc

    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    examples: list[EvalExample] = []
    for i, row in enumerate(ds):
        label = map_tweeteval_label(row["label"])
        examples.append(
            EvalExample(
                example_id=f"tweeteval-{split}-{i}",
                gold_label=label,
                payload={"text": row["text"]},
                meta={"source": "huggingface", "dataset": "cardiffnlp/tweet_eval", "split": split},
            ),
        )
    return examples


def evaluate_text_examples(
    examples: Sequence[EvalExample],
    predict_fn,
    *,
    limit: Optional[int] = None,
    split: str = "custom",
):
    from evaluation.common import (
        apply_limit,
        build_eval_result,
        run_prediction_loop,
    )

    selected = apply_limit(examples, limit)
    predictions, errors = run_prediction_loop(selected, predict_fn)
    return build_eval_result(
        dataset="TweetEval-sentiment",
        split=split,
        examples=selected,
        predictions=predictions,
        errors=errors,
        label_mapping={"source": "TweetEval sentiment 0/1/2 -> negative/neutral/positive"},
        notes=[
            "Official sources: https://github.com/cardiffnlp/tweeteval ; "
            "https://huggingface.co/datasets/cardiffnlp/tweet_eval",
        ],
    )
