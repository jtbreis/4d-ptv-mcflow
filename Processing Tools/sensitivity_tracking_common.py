"""Shared helpers for STM error and box_size sensitivity tracking sweeps."""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
POSTPROC_DIR = os.path.join(REPO_ROOT, "Post-Processing")
if POSTPROC_DIR not in sys.path:
    sys.path.insert(0, POSTPROC_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from pipeline_paths import ensure_output_run_dir  # noqa: E402


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = _env_str(name)
    if not raw:
        return default
    return float(raw)


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = _env_str(name)
    if not raw:
        return default
    return int(raw)


def _env_truthy(name: str) -> bool:
    return _env_str(name).lower() in ("1", "true", "yes", "on")


def parse_runs(runs: str | None) -> list[str]:
    if not runs:
        return ["Run1"]
    return [r.strip() for r in runs.split(",") if r.strip()]


def resolve_output_root(output_dir: str, output_base: str) -> str:
    output_dir = os.path.abspath(output_dir)
    output_base = output_base.strip().strip("/")
    if not output_base:
        return output_dir
    return os.path.join(output_dir, output_base)


def resolve_run_dir(output_root: str, case: str, run: str) -> str:
    return os.path.join(output_root, case, run)


def build_float_thresholds(
    *,
    start: float,
    end: float,
    step: float,
) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    if end < start:
        raise ValueError("end must be >= start")
    n = int(math.floor((end - start) / step + 1e-9)) + 1
    values = [round(start + i * step, 10) for i in range(n)]
    if values[-1] < end - 1e-12:
        values.append(end)
    # dedupe while preserving order
    out: list[float] = []
    for v in values:
        if not out or abs(v - out[-1]) > 1e-12:
            out.append(v)
    return out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_summary_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote summary: {path}")


def should_skip_analysis() -> bool:
    return _env_truthy("SKIP_SENSITIVITY_ANALYSIS")


def run_sensitivity_plots(run_dir: str, sweep: str) -> dict[str, str]:
    from plot_tracking_sensitivity import SWEEP_CONFIGS, run_analysis

    if sweep not in SWEEP_CONFIGS:
        raise ValueError(f"Unknown sweep type: {sweep}")
    print(f"Running sensitivity analysis plots ({sweep}) in {run_dir}")
    paths = run_analysis(run_dir, SWEEP_CONFIGS[sweep])
    print("Analysis PDFs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


def maybe_run_sensitivity_plots(run_dir: str, sweep: str) -> dict[str, str] | None:
    if should_skip_analysis():
        print("SKIP_SENSITIVITY_ANALYSIS=1 — skipping plot generation.")
        return None
    return run_sensitivity_plots(run_dir, sweep)
