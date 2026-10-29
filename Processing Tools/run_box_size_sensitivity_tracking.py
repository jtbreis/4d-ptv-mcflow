#!/usr/bin/env python3
"""
Sweep box_size_track and run graph tracking for each hop limit.

Writes tracks_box_size_{size:.2f}.h5part per value and optionally generates PDF
sensitivity plots when tracking completes (or run plot_box_size_sensitivity.py later).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from graph_tracking.link_block import LinkParams
from graph_tracking.load_data import load_all_errors
from graph_tracking.run_tracking import run_graph_tracking

from sensitivity_tracking_common import (  # noqa: E402
    _env_float,
    _env_int,
    _env_str,
    _env_truthy,
    build_float_thresholds,
    maybe_run_sensitivity_plots,
    parse_runs,
    resolve_output_root,
    resolve_run_dir,
    utc_now_iso,
    write_summary_json,
)


def _link_params_from_env(*, box_size_track: float) -> LinkParams:
    y_half = _env_float("BOX_SIZE_Y", 1.5) or 1.5
    z_half = _env_float("BOX_SIZE_Z", 1.5) or 1.5
    return LinkParams(
        box_size_track=float(box_size_track),
        initial_x_lo=_env_float("BOX_SIZE_INITIAL_X_LO", 0.0) or 0.0,
        initial_x_hi=_env_float("BOX_SIZE_INITIAL_X_HI", 3.5) or 3.5,
        initial_y_lo=-y_half,
        initial_y_hi=y_half,
        initial_z_lo=-z_half,
        initial_z_hi=z_half,
        max_branch=_env_int("MAX_BRANCH", 30) or 30,
        max_seeds=_env_int("MAX_SEEDS", 0) or 0,
        allow_3_frame=not _env_truthy("NO_3_FRAME"),
        use_bidirectional_graph=_env_truthy("USE_BIDIRECTIONAL_GRAPH"),
        use_numba=not _env_truthy("NO_NUMBA"),
        quiet_linking=False,
        progress_interval=_env_int("PROGRESS_INTERVAL", 5000) or 5000,
    )


def _tracks_path(run_dir: str, box_size: float) -> str:
    return os.path.join(run_dir, f"tracks_box_size_{box_size:.2f}.h5part")


def _optional_stm_masks(h5in: str, max_stereo_error: float | None) -> list[np.ndarray] | None:
    if max_stereo_error is None:
        return None
    all_errors = load_all_errors(h5in)
    return [err <= max_stereo_error for err in all_errors]


def run_box_size_sweep_for_run(
    run_dir: str,
    *,
    stm_filename: str,
    box_sizes: list[float],
    workers: int,
    max_blocks: int,
    min_track_length: int,
    dt: float,
    rep_rate: float,
    position_unit: str,
    max_stereo_error: float | None,
    skip_existing: bool,
) -> list[dict]:
    h5in = os.path.join(run_dir, stm_filename)
    if not os.path.isfile(h5in):
        raise FileNotFoundError(h5in)

    masks = _optional_stm_masks(h5in, max_stereo_error)
    if masks is not None:
        available = int(sum(int(np.sum(m)) for m in masks))
        print(
            f"STM error ceiling {max_stereo_error}: {available:,} detections in pool"
        )

    runs_out: list[dict] = []
    for box_size in box_sizes:
        out_path = _tracks_path(run_dir, box_size)
        if skip_existing and os.path.isfile(out_path):
            print(f"Skipping box_size={box_size:.2f} (exists): {out_path}")
            runs_out.append(
                {
                    "box_size_track": box_size,
                    "tracks_h5part": out_path,
                    "skipped": True,
                }
            )
            continue

        link_params = _link_params_from_env(box_size_track=box_size)
        print(f"=== box_size_track = {box_size:.2f} mm ===")

        t0 = time.time()
        tracking = run_graph_tracking(
            h5in,
            run_dir,
            output=out_path,
            max_blocks=max_blocks,
            workers=workers,
            link_params=link_params,
            min_track_length=min_track_length,
            dt=dt,
            rep_rate=rep_rate,
            position_unit=position_unit,
            show_write_progress=True,
            verbose=True,
            masks=masks,
        )
        elapsed = time.time() - t0
        tracking = dict(tracking)
        tracking["elapsed_s"] = elapsed

        runs_out.append(
            {
                "box_size_track": box_size,
                "tracks_h5part": out_path,
                "tracking": tracking,
                "skipped": False,
            }
        )

    return runs_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="box_size_track sensitivity tracking sweep.")
    p.add_argument("--output-dir", default=_env_str("OUTPUT_DIR", REPO_ROOT))
    p.add_argument(
        "--output-base",
        default=_env_str(
            "OUTPUT_BASE",
            "data/Windtunnel_PTV_TNTI/X300cm_PTV_above_TNTI",
        ),
    )
    p.add_argument("--case", default=_env_str("CASE", "TNTI_opposing_gravity"))
    p.add_argument("--runs", default=_env_str("RUNS", "Run1"))
    p.add_argument("--stm-filename", default=_env_str("STM_FILENAME", "rays_out_cpp.h5"))
    p.add_argument(
        "--box-size-min",
        type=float,
        default=_env_float("BOX_SIZE_MIN"),
    )
    p.add_argument(
        "--box-size-max",
        type=float,
        default=_env_float("BOX_SIZE_MAX"),
    )
    p.add_argument(
        "--box-size-step",
        type=float,
        default=_env_float("BOX_SIZE_STEP", 0.05),
    )
    p.add_argument(
        "--max-stereo-error",
        type=float,
        default=_env_float("MAX_STEREO_ERROR"),
        help="Optional STM error ceiling applied to all box_size runs.",
    )
    p.add_argument("--workers", type=int, default=_env_int("WORKERS", 4) or 4)
    p.add_argument("--max-blocks", type=int, default=_env_int("MAX_BLOCKS", 0) or 0)
    p.add_argument(
        "--min-track-length",
        type=int,
        default=_env_int("MIN_TRACK_LENGTH", 4) or 4,
    )
    p.add_argument("--dt", type=float, default=_env_float("DT", 1e-3) or 1e-3)
    p.add_argument("--rep-rate", type=float, default=_env_float("REP_RATE", 10.0) or 10.0)
    p.add_argument(
        "--position-unit",
        default=_env_str("POSITION_UNIT", "mm") or "mm",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=_env_truthy("SKIP_EXISTING"),
    )
    p.add_argument(
        "--analysis-only",
        action="store_true",
        default=_env_truthy("ANALYSIS_ONLY"),
        help="Only generate PDF plots from existing tracks_box_size_*.h5part files.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = resolve_output_root(args.output_dir, args.output_base)
    run_names = parse_runs(args.runs)

    if args.analysis_only:
        for run in run_names:
            run_dir = resolve_run_dir(output_root, args.case, run)
            maybe_run_sensitivity_plots(run_dir, "box_size")
        return 0

    if args.box_size_min is None or args.box_size_max is None:
        raise ValueError("--box-size-min and --box-size-max are required")

    box_sizes = build_float_thresholds(
        start=float(args.box_size_min),
        end=float(args.box_size_max),
        step=float(args.box_size_step),
    )

    for run in run_names:
        run_dir = resolve_run_dir(output_root, args.case, run)
        os.makedirs(run_dir, exist_ok=True)
        h5in = os.path.join(run_dir, args.stm_filename)
        if not os.path.isfile(h5in):
            raise FileNotFoundError(h5in)

        print(f"Run directory: {run_dir}")
        print(f"box_size_track values: {box_sizes}")

        sweep_runs = run_box_size_sweep_for_run(
            run_dir,
            stm_filename=args.stm_filename,
            box_sizes=box_sizes,
            workers=args.workers,
            max_blocks=args.max_blocks,
            min_track_length=args.min_track_length,
            dt=args.dt,
            rep_rate=args.rep_rate,
            position_unit=args.position_unit,
            max_stereo_error=args.max_stereo_error,
            skip_existing=args.skip_existing,
        )

        summary = {
            "input_stm": os.path.abspath(h5in),
            "output_dir": os.path.abspath(run_dir),
            "config": {
                "box_size_min": args.box_size_min,
                "box_size_max": args.box_size_max,
                "box_size_step": args.box_size_step,
                "max_stereo_error": args.max_stereo_error,
                "dt": args.dt,
                "workers": args.workers,
                "max_blocks": args.max_blocks,
                "min_track_length": args.min_track_length,
                "rep_rate": args.rep_rate,
                "position_unit": args.position_unit,
                "tracks_template": "tracks_box_size_{box_size:.2f}.h5part",
                "summary_name": "box_size_sensitivity_summary.json",
            },
            "box_sizes": box_sizes,
            "runs": sweep_runs,
            "timestamp_utc": utc_now_iso(),
        }
        write_summary_json(
            os.path.join(run_dir, "box_size_sensitivity_summary.json"),
            summary,
        )
        maybe_run_sensitivity_plots(run_dir, "box_size")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
