"""Export frozen Phase 3A controlled-video reasoner payload (no media bytes).

Usage:
  python scripts/export_reasoner_benchmark_payload.py

Prefers existing outputs/controlled_validation_phase3a.json when present so
analyzers are not re-run. Falls back to MyUniSentimentPipeline.analyze only
if the JSON dump is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.temporal.benchmark.export import (  # noqa: E402
    export_from_activity_analysis_dict,
    save_frozen_payload,
)
from src.temporal.benchmark.fixtures import REAL_CONTROLLED_PAYLOAD_PATH  # noqa: E402

PHASE3A_JSON = ROOT / "outputs" / "controlled_validation_phase3a.json"
VIDEO = ROOT / "demo_assets" / "temporal_progression_demo.mp4"

KNOWN_LIMITATIONS = [
    "Speech-window boundary artifact: window 3 local text is "
    "'it's honestly, it's honestly becoming pretty' while 'overwhelming.' "
    "is assigned to window 4 by word-midpoint rule. Do not change word "
    "assignment during model comparison.",
]

NOTES = [
    "Frozen deterministic TemporalContext + baseline fusion from Phase 3A "
    "controlled real-video validation. Reasoners must consume this payload "
    "without re-running video/SigLIP/Whisper/RoBERTa.",
]


def _load_analysis_from_phase3a_json() -> dict:
    data = json.loads(PHASE3A_JSON.read_text(encoding="utf-8"))
    return data["activity"]["analysis"]


def _run_pipeline_and_dump() -> dict:
    from src.pipeline import MyUniSentimentPipeline

    if not VIDEO.is_file():
        raise FileNotFoundError(f"Missing controlled video: {VIDEO}")
    routed = MyUniSentimentPipeline().analyze(media_path=str(VIDEO))
    if routed.analysis is None:
        raise RuntimeError(f"pipeline failed: {routed.message}")
    analysis = routed.analysis.analysis.model_dump(mode="json")
    PHASE3A_JSON.parent.mkdir(parents=True, exist_ok=True)
    PHASE3A_JSON.write_text(
        json.dumps(
            {
                "total_pipeline_seconds": None,
                "activity": routed.analysis.model_dump(mode="json"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return analysis


def main() -> int:
    if PHASE3A_JSON.is_file():
        print(f"Using existing Phase 3A dump: {PHASE3A_JSON}")
        analysis = _load_analysis_from_phase3a_json()
    else:
        print("Phase 3A dump missing — running pipeline once to export fixture...")
        analysis = _run_pipeline_and_dump()

    payload = export_from_activity_analysis_dict(
        analysis,
        fixture_id="real_phase3a_controlled_video",
        source="demo_assets/temporal_progression_demo.mp4",
        known_limitations=KNOWN_LIMITATIONS,
        notes=NOTES,
    )
    path = save_frozen_payload(payload, REAL_CONTROLLED_PAYLOAD_PATH)
    print(f"Wrote {path}")
    print(f"trajectory={payload.temporal_context.features.trajectory}")
    print(f"windows={len(payload.temporal_context.windows)}")
    print(f"valid_evidence_ids={len(payload.valid_evidence_ids)}")
    print(f"alignment={payload.temporal_context.speech_alignment_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
