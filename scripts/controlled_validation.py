"""Phase 3A controlled revalidation — word-timestamp temporal speech."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO_PATH = str(ROOT / "demo_assets" / "temporal_progression_demo.mp4")
OUT_PATH = ROOT / "outputs" / "controlled_validation_phase3a.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def sep(title: str) -> None:
    print(f"\n\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


sep("IMPORT / BUILD")
t0 = time.time()
from src.pipeline import MyUniSentimentPipeline  # noqa: E402
from src.temporal.features import _linear_slope, _normalized_temporal_positions  # noqa: E402

pipeline = MyUniSentimentPipeline()
print(f"setup_seconds={time.time() - t0:.2f}", flush=True)

sep("RUN ANALYSIS")
t0 = time.time()
routed = pipeline.analyze(media_path=VIDEO_PATH)
total = time.time() - t0
print(f"total_pipeline_seconds={total:.2f}", flush=True)
print(f"status={routed.status} input={routed.detected_input}", flush=True)
if routed.analysis is None:
    print("ERROR:", routed.message)
    sys.exit(1)

activity = routed.analysis
ab = activity.analysis
payload = {
    "total_pipeline_seconds": total,
    "activity": activity.model_dump(mode="json"),
}
OUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
print(f"Wrote {OUT_PATH}", flush=True)

d = payload["activity"]["analysis"]
vid = d.get("video") or {}
tc = d.get("temporal_context") or {}
tr = d.get("temporal_reasoning") or {}
diag = d.get("temporal_reasoner_diagnostics") or {}
feats = tc.get("features") or {}
windows = tc.get("windows") or []
overall = d.get("overall") or {}
modalities = d.get("modalities") or {}

sep("1. WHISPER TOP-LEVEL / WORDS")
print(f"transcript={repr(d.get('transcript'))}")
print(f"speech_alignment_source={tc.get('speech_alignment_source')}")
print(f"speech_word_count={tc.get('speech_word_count')}")
print(f"global speech modality={json.dumps(modalities.get('speech'), indent=2)[:900]}")

print("\nWhisper top-level segments (source_speech_segments):")
for seg in tc.get("source_speech_segments") or []:
    print(f"  SEGMENT [{seg.get('start'):.2f}-{seg.get('end'):.2f}] {repr(seg.get('text'))}")

words = tc.get("speech_words") or []
print(f"\nword_count={len(words)}")
print("first_10_words:")
for w in words[:10]:
    print(f"  [{w.get('start'):.3f}-{w.get('end'):.3f}] {repr(w.get('text'))} id={w.get('word_id')}")
print("last_10_words:")
for w in words[-10:]:
    print(f"  [{w.get('start'):.3f}-{w.get('end'):.3f}] {repr(w.get('text'))} id={w.get('word_id')}")

sep("2-6. WINDOWS / SPEECH / VISUAL")
print(f"alignment_mode={tc.get('speech_alignment_source')}")
for w in windows:
    print(f"\nwindow {w.get('index')} [{w.get('start')}-{w.get('end')}] usable={w.get('usable')}")
    print(f"  modalities={w.get('available_modalities')}")
    print(f"  dominant={w.get('dominant_label')} P(neg)={w.get('negative_probability')}")
    print(f"  visual_probs={w.get('visual_probabilities')}")
    print(f"  speech_probs={w.get('speech_probabilities')}")
    texts = [s.get("text") for s in (w.get("speech_segments") or [])]
    print(f"  speech_texts={texts}")

sep("7. VISUAL UNCHANGED CHECK")
vis = modalities.get("visual") or {}
print(f"aggregate visual label={vis.get('label')} score={vis.get('score')} conf={vis.get('confidence')}")
print(f"visual probs={vis.get('probabilities')}")
print("Previous controlled run aggregate visual: negative score≈-0.743 conf≈0.831")

sep("8-14. DETERMINISTIC FEATURES")
print(json.dumps(feats, indent=2))

usable = [w for w in windows if w.get("usable")]
centers = [(float(w["start"]) + float(w["end"])) / 2.0 for w in usable]
ys = [float(w.get("negative_probability") or 0.0) for w in usable]
xs = _normalized_temporal_positions(centers)
print(f"\n{'idx':>4} {'center':>8} {'norm_x':>8} {'P(neg)':>8}")
for w, c, x, y in zip(usable, centers, xs, ys):
    print(f"{w.get('index'):>4} {c:8.3f} {x:8.4f} {y:8.4f}")
slope = _linear_slope(xs, ys)
print(f"manual_OLS_slope={slope:.6f}")
print(f"trajectory={feats.get('trajectory')}")

sep("15-16. QWEN")
print(json.dumps(tr, indent=2)[:4000])
print("\ndiagnostics:")
print(json.dumps(diag, indent=2)[:2500])
supplied = set(diag.get("evidence_ids_supplied") or [])
cited = {e.get("evidence_id") for e in (tr.get("evidence") or [])}
for t in tr.get("important_transitions") or []:
    cited.update(t.get("evidence_ids") or [])
print(f"unknown_ids={sorted(cited - supplied)}")
print(f"repair_attempted={diag.get('repair_attempted')}")

sep("17. PERFORMANCE")
print(f"complete_request={total:.2f}s")
print(f"video.extraction={vid.get('extraction_seconds')}")
print(f"video.processing={vid.get('processing_seconds')}")
print(f"qwen.load={diag.get('model_load_seconds')}")
print(f"qwen.generation={diag.get('generation_seconds')}")
print(f"qwen.total={diag.get('total_reasoner_seconds')}")
print("previous_controlled_run_total≈773s video.processing≈37.5s")

sep("FUSION")
print(json.dumps(overall, indent=2)[:1200])

print("\n=== PHASE3A DUMP COMPLETE ===", flush=True)
