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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep ordinary `pytest` runs fast: skip asr_integration unless explicitly selected."""
    markexpr = (config.option.markexpr or "").strip()
    if "asr_integration" in markexpr:
        return
    skip = pytest.mark.skip(
        reason="optional ASR integration (run: pytest -m asr_integration with MYUNI_RUN_ASR_INTEGRATION=1)",
    )
    for item in items:
        if "asr_integration" in item.keywords:
            item.add_marker(skip)
