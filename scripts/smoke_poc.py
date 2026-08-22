#!/usr/bin/env python3
"""POC smoke checks that avoid large model downloads by default.

Runs:
  - environment health report
  - evaluation stub path (no HF weights)
  - optional representative analysis when MYUNI_RUN_SMOKE_MODELS=1

Does not fabricate success for steps that cannot run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_check import build_health_report


def _run(cmd: list[str], *, cwd: Path = ROOT) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report: dict[str, object] = {"steps": []}
    ok = True

    health = build_health_report()
    report["health"] = health
    report["steps"].append({"name": "health", "status": "ok"})
    print("=== Health ===")
    print(json.dumps(health, indent=2, ensure_ascii=False))

    code, out = _run(
        [
            sys.executable,
            "-m",
            "evaluation.run",
            "text",
            "--data",
            "evaluation/fixtures/text_samples.jsonl",
            "--stub",
            "--limit",
            "3",
            "--out",
            "outputs/smoke_eval_stub",
        ],
    )
    step = {"name": "evaluation_stub", "exit_code": code}
    if code == 0:
        step["status"] = "ok"
        print("\n=== Evaluation stub === OK")
    else:
        step["status"] = "failed"
        step["output"] = out[-500:]
        ok = False
        print("\n=== Evaluation stub === FAILED")
        print(out[-800:])
    report["steps"].append(step)

    samples = {
        "image": ROOT / "data" / "samples" / "synthetic_sample.png",
        "video": ROOT / "data" / "samples" / "synthetic_sample.mp4",
    }
    for kind, path in samples.items():
        exists = path.is_file()
        report["steps"].append(
            {
                "name": f"sample_{kind}",
                "status": "ok" if exists else "missing",
                "path": str(path),
            },
        )
        print(f"\n=== Sample {kind} === {'found' if exists else 'MISSING'} ({path})")

    if os.environ.get("MYUNI_RUN_SMOKE_MODELS") == "1":
        print("\n=== Text analysis (downloads models) ===")
        code, out = _run(
            [sys.executable, "main.py", "Great day on campus!", "--log-level", "WARNING"],
        )
        step = {"name": "text_analysis", "exit_code": code}
        if code == 0:
            step["status"] = "ok"
            json_start = out.find("{")
            if json_start >= 0:
                try:
                    payload = json.loads(out[json_start:])
                    step["overall_label"] = payload.get("analysis", {}).get("overall", {}).get("label")
                    runtime = payload.get("analysis", {}).get("runtime")
                    if runtime:
                        step["models"] = runtime.get("models")
                except json.JSONDecodeError:
                    step["parse_error"] = "stdout was not JSON"
            else:
                step["parse_error"] = "no JSON payload in stdout"
            print("OK")
        else:
            step["status"] = "failed"
            step["output"] = out[-500:]
            ok = False
            print("FAILED")
            print(out[-800:])
        report["steps"].append(step)

        if samples["image"].is_file():
            print("\n=== Image analysis ===")
            code, out = _run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, sys; from pathlib import Path; "
                        "ROOT=Path('.').resolve(); sys.path.insert(0, str(ROOT)); "
                        "from datetime import datetime, timezone; "
                        "from src.pipeline import MyUniSentimentPipeline; "
                        "from src.schemas import ActivityInput; "
                        "p=MyUniSentimentPipeline(); "
                        "a=ActivityInput(activity_id='SMK-IMG', user_id='SMK', "
                        "activity_type='image', media_path=r'"
                        + str(samples["image"]).replace("\\", "\\\\")
                        + "', created_at=datetime.now(timezone.utc)); "
                        "r=p.analyze_activity(a); "
                        "print(json.dumps(r.model_dump_json_compatible()))"
                    ),
                ],
            )
            step = {"name": "image_analysis", "exit_code": code}
            if code == 0:
                step["status"] = "ok"
            else:
                step["status"] = "failed"
                step["output"] = out[-500:]
                ok = False
            report["steps"].append(step)
            print(step["status"].upper())
            if code != 0:
                print(out[-600:])

        if samples["video"].is_file() and health["dependencies"]["ffmpeg"]["available"]:
            print("\n=== Video analysis ===")
            code, out = _run(
                [
                    sys.executable,
                    "main.py",
                    "--video",
                    str(samples["video"]),
                    "--caption",
                    "Smoke clip",
                    "--log-level",
                    "INFO",
                ],
            )
            step = {"name": "video_analysis", "exit_code": code, "status": "ok" if code == 0 else "failed"}
            if code != 0:
                step["output"] = out[-500:]
                ok = False
            report["steps"].append(step)
            print(step["status"].upper())
        elif samples["video"].is_file():
            report["steps"].append(
                {"name": "video_analysis", "status": "skipped", "reason": "FFmpeg unavailable"},
            )
            print("\n=== Video analysis === SKIPPED (FFmpeg missing)")
    else:
        report["steps"].append(
            {
                "name": "model_smoke",
                "status": "skipped",
                "reason": "Set MYUNI_RUN_SMOKE_MODELS=1 to run text/image/video with real models",
            },
        )
        print("\n=== Model smoke === SKIPPED (set MYUNI_RUN_SMOKE_MODELS=1 to enable)")

    out_path = ROOT / "outputs" / "smoke_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
