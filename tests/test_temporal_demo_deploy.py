"""Hardening checks for the MyUni temporal demo HF Space bundle.

Does not push. Does not call OpenRouter. Does not modify protected Spaces.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = Path(r"D:\Work\hf-deploy\myuni-temporal-demo")
PROD = Path(r"D:\Work\hf-deploy\My-Space")
BENCH = Path(r"D:\Work\hf-deploy\myuni-temporal-reasoner-benchmark")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _list_files(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if ".git" in p.parts:
            continue
        out.append(p)
    return sorted(out)


@pytest.fixture(scope="module")
def deploy_ready() -> Path:
    if not DEPLOY.exists():
        pytest.skip(f"Deployment bundle missing: {DEPLOY}")
    return DEPLOY


def test_deployment_bundle_exists_and_has_entrypoint(deploy_ready: Path) -> None:
    assert (deploy_ready / "app_gradio.py").is_file()
    assert (deploy_ready / "README.md").is_file()
    assert (deploy_ready / "requirements.txt").is_file()
    assert (deploy_ready / "packages.txt").is_file()
    assert (deploy_ready / ".gitignore").is_file()
    assert (deploy_ready / "config" / "fusion.yaml").is_file()
    assert (deploy_ready / "src" / "pipeline.py").is_file()
    assert (deploy_ready / "src" / "temporal" / "providers" / "openrouter.py").is_file()
    assert (deploy_ready / "src" / "temporal" / "wellbeing.py").is_file()
    assert (deploy_ready / "src" / "temporal" / "final_assessment.py").is_file()


def test_readme_metadata(deploy_ready: Path) -> None:
    text = (deploy_ready / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "title: MyUni Temporal Intelligence" in text
    assert "sdk: gradio" in text
    assert "sdk_version: 5.50.0" in text
    assert "app_file: app_gradio.py" in text
    assert "python_version: 3.12.12" in text
    assert "OPENROUTER_API_KEY" in text
    # Secret value must never appear.
    assert re.search(r"sk-[A-Za-z0-9]{10,}", text) is None
    assert "Bearer " not in text


def test_requirements_match_hf_pin(deploy_ready: Path) -> None:
    req = (deploy_ready / "requirements.txt").read_text(encoding="utf-8")
    hf = (ROOT / "requirements-hf.txt").read_text(encoding="utf-8")
    assert req == hf
    assert "torch==2.11.0" in req
    assert "openai" not in req.lower() or "openai/gpt" in req.lower()
    assert "langchain" not in req.lower()
    assert "openai\n" not in req
    assert "openai==" not in req


def test_packages_has_tesseract(deploy_ready: Path) -> None:
    assert "tesseract-ocr" in (deploy_ready / "packages.txt").read_text(encoding="utf-8")


def test_spaces_imported_before_pipeline(deploy_ready: Path) -> None:
    source = (deploy_ready / "app_gradio.py").read_text(encoding="utf-8")
    spaces_idx = source.find("import spaces")
    pipeline_idx = source.find("from src.pipeline")
    assert spaces_idx != -1 and pipeline_idx != -1
    assert spaces_idx < pipeline_idx
    # Static AST: spaces before any src / torch import.
    tree = ast.parse(source)
    order: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                order.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            order.append(mod)
    relevant = [n for n in order if n in {"spaces", "torch", "src", "gradio"}]
    assert relevant.index("spaces") < relevant.index("gradio")
    assert relevant.index("spaces") < relevant.index("src")
    assert "torch" not in relevant  # torch must not be imported at app top-level


def test_siglip_zerogpu_boundary_preserved(deploy_ready: Path) -> None:
    visual = (deploy_ready / "src" / "analyzers" / "visual.py").read_text(encoding="utf-8")
    assert "import spaces" in visual
    assert "@spaces.GPU" in visual
    assert "def _siglip_gpu_forward" in visual
    app = (deploy_ready / "app_gradio.py").read_text(encoding="utf-8")
    assert 'os.environ.get("SPACE_ID")' in app
    assert "@spaces.GPU" not in app  # reasoner must not be ZeroGPU-wrapped in app


def test_openrouter_defaults_in_bundle(deploy_ready: Path) -> None:
    cfg = (deploy_ready / "src" / "config.py").read_text(encoding="utf-8")
    assert 'TEMPORAL_REASONER_PROVIDER = "openrouter"' in cfg
    assert 'TEMPORAL_REASONER_FALLBACK = "none"' in cfg
    assert 'OPENROUTER_REASONER_MODEL = "openai/gpt-oss-20b:free"' in cfg
    assert "OPENROUTER_API_KEY" in cfg or "OPENROUTER_API_KEY" in (
        deploy_ready / "src" / "temporal" / "providers" / "openrouter.py"
    ).read_text(encoding="utf-8")


def test_no_secret_in_bundle_files(deploy_ready: Path) -> None:
    secret_re = re.compile(
        r"(sk-[A-Za-z0-9_-]{16,})|(OPENROUTER_API_KEY\s*=\s*['\"][^'\"]+['\"])",
        re.IGNORECASE,
    )
    for path in _list_files(deploy_ready):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".wav"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        assert secret_re.search(text) is None, f"possible secret in {path}"
        assert path.name != ".env"
        assert not path.name.startswith(".env.")


def test_no_model_caches_in_bundle(deploy_ready: Path) -> None:
    banned_suffix = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".ckpt"}
    for path in _list_files(deploy_ready):
        if "__pycache__" in path.parts:
            continue
        assert path.suffix.lower() not in banned_suffix, path
        assert path.name != ".venv"


def test_gitignore_excludes_secrets_and_caches(deploy_ready: Path) -> None:
    gi = (deploy_ready / ".gitignore").read_text(encoding="utf-8")
    for needle in (".env", ".cache/", "*.safetensors", "*.bin", ".venv/", "outputs/"):
        assert needle in gi


def test_qwen_not_default_provider(deploy_ready: Path) -> None:
    import subprocess

    script = (
        "import sys\n"
        f"sys.path.insert(0, r'{deploy_ready}')\n"
        "from src.config import (\n"
        "    TEMPORAL_REASONER_FALLBACK,\n"
        "    TEMPORAL_REASONER_PROVIDER,\n"
        "    OPENROUTER_REASONER_MODEL,\n"
        "    resolve_temporal_reasoner_config,\n"
        ")\n"
        "assert TEMPORAL_REASONER_PROVIDER == 'openrouter'\n"
        "assert TEMPORAL_REASONER_FALLBACK == 'none'\n"
        "assert OPENROUTER_REASONER_MODEL == 'openai/gpt-oss-20b:free'\n"
        "cfg = resolve_temporal_reasoner_config()\n"
        "assert cfg.provider == 'openrouter'\n"
        "assert cfg.fallback == 'none'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(deploy_ready),
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_no_tests_or_benchmark_ui_in_bundle(deploy_ready: Path) -> None:
    assert not (deploy_ready / "tests").exists()
    assert not (deploy_ready / "evaluation").exists()
    app = (deploy_ready / "app_gradio.py").read_text(encoding="utf-8")
    assert "benchmark" not in app.lower()
    assert "Run Benchmark" not in app


def test_protected_spaces_still_exist_and_distinct(deploy_ready: Path) -> None:
    assert PROD.is_dir()
    assert BENCH.is_dir()
    assert deploy_ready.resolve() != PROD.resolve()
    assert deploy_ready.resolve() != BENCH.resolve()
    # Demo must not be nested inside protected dirs.
    with pytest.raises(ValueError):
        deploy_ready.resolve().relative_to(PROD.resolve())
    with pytest.raises(ValueError):
        deploy_ready.resolve().relative_to(BENCH.resolve())


def test_production_entrypoint_still_present() -> None:
    assert (PROD / "app_gradio.py").is_file()
    assert (PROD / "requirements.txt").is_file()


def test_benchmark_entrypoint_still_present() -> None:
    assert (BENCH / "app.py").is_file() or (BENCH / "README.md").is_file()


def test_deploy_app_import_lazy_no_openrouter(deploy_ready: Path) -> None:
    import shutil
    import subprocess

    script = (
        "import os, sys, urllib.request\n"
        "os.environ.pop('OPENROUTER_API_KEY', None)\n"
        "os.environ.pop('SPACE_ID', None)\n"
        "def boom(*a, **k):\n"
        "    raise AssertionError('OpenRouter/HTTP must not run on app import')\n"
        "urllib.request.urlopen = boom\n"
        f"sys.path.insert(0, r'{deploy_ready}')\n"
        "import app_gradio as demo_app\n"
        "assert demo_app.demo is not None\n"
        "assert demo_app._pipeline is None\n"
        "from src.temporal.reasoner import TemporalContextReasoner\n"
        "assert TemporalContextReasoner is not None\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(deploy_ready),
        env={**os.environ, "OPENROUTER_API_KEY": "", "SPACE_ID": ""},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "ok" in proc.stdout
    # Local import may create __pycache__; strip it so the bundle stays clean.
    for cache in deploy_ready.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
