"""
pytest configuration for MyUni sentiment POC.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that download/run the Hugging Face text model",
    )
