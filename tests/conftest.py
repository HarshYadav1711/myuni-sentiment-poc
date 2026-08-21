"""
pytest configuration for MyUni sentiment POC.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that download/run ML models (text/visual/etc.)",
    )
    config.addinivalue_line(
        "markers",
        "asr_integration: optional faster-whisper + FFmpeg tests (opt-in via MYUNI_RUN_ASR_INTEGRATION=1)",
    )
    config.addinivalue_line(
        "markers",
        "video_integration: optional end-to-end video smoke (opt-in via MYUNI_RUN_VIDEO_INTEGRATION=1)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep ordinary `pytest` runs fast: skip heavy optional integration markers."""
    markexpr = (config.option.markexpr or "").strip()
    skip_rules = [
        (
            "asr_integration",
            "optional ASR integration (run: pytest -m asr_integration with MYUNI_RUN_ASR_INTEGRATION=1)",
        ),
        (
            "video_integration",
            "optional video integration (run: pytest -m video_integration with MYUNI_RUN_VIDEO_INTEGRATION=1)",
        ),
    ]
    for marker_name, reason in skip_rules:
        if marker_name in markexpr:
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker_name in item.keywords:
                item.add_marker(skip)
