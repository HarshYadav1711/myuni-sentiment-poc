"""Optional live-model smoke for the Phase 4 wellbeing classifier.

Skipped by ordinary ``pytest -m "not integration"`` runs.
Does download DeBERTa when explicitly selected.
"""

from __future__ import annotations

import os

import pytest

from src.wellbeing import WellbeingClassifier
from wellbeing_fixtures import WELLBEING_SEMANTIC_FIXTURES


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("MYUNI_RUN_WELLBEING_INTEGRATION") != "1",
    reason=(
        "optional wellbeing classifier integration "
        "(run: MYUNI_RUN_WELLBEING_INTEGRATION=1 pytest -m integration)"
    ),
)
def test_live_deberta_classify_smoke() -> None:
    clf = WellbeingClassifier(enabled=True)
    sample = WELLBEING_SEMANTIC_FIXTURES[0]["text"]
    result = clf.classify(sample)
    assert result.model_id.endswith("deberta-v3-base-zeroshot-v2.0-c")
    assert result.status in {"ok", "error", "classifier_unavailable"}
    if result.status == "ok":
        assert result.relevance.label in {
            "personal_wellbeing",
            "wellbeing_topic_only",
            "not_wellbeing_related",
            "ambiguous",
        }
        assert result.target.label is not None
        assert len(result.signals) == 9
