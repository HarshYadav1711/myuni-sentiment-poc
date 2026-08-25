"""Tests for client-demo asset discovery helpers (no model load)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_demo_assets_layout_documented() -> None:
    demo = ROOT / "demo_assets"
    assert demo.is_dir()
    readme = demo / "README.txt"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8").lower()
    assert "positive_image" in text
    assert "negative_image" in text
    assert "demo_video" in text


def test_demo_sample_files_when_present_are_valid_media() -> None:
    demo = ROOT / "demo_assets"
    pos = demo / "positive_image.jpg"
    neg = demo / "negative_image.jpg"
    vid = demo / "demo_video.mp4"
    if pos.is_file():
        assert pos.stat().st_size > 100
    if neg.is_file():
        assert neg.stat().st_size > 100
    if vid.is_file():
        assert vid.stat().st_size > 100


def test_compare_app_is_unified_no_mode_selector() -> None:
    source = (ROOT / "app_compare.py").read_text(encoding="utf-8")
    assert "MyUni Sentiment Intelligence" in source
    assert "Analyze Content" in source
    assert "Content type is detected automatically" in source
    assert "segmented_control" not in source
    assert 'st.radio(' not in source
    assert "get_pipeline" in source
    assert "demo_assets" in source
    assert "Please analyze one content item at a time" in source
