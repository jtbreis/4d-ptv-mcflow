#!/usr/bin/env python3
"""
Sweep STM error thresholds and run graph tracking for each ceiling.

For each threshold, filters particles in-memory (error <= threshold), runs graph
tracking, and writes tracks_error_{threshold:.4f}.h5part. Optionally generates
PDF sensitivity plots when tracking completes (or run plot_stm_error_sensitivity.py later).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
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
from graph_tracking.load_data import get_nframes, load_all_errors
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


def _stm_max_error(h5path: str) -> float:
    with h5py.File(h5path, "r") as f:
        if "parameters_max_distance" in f.attrs:
            return float(np.asarray(f.attrs["parameters_max_distance"]).ravel()[0])
        if "maxdistance" in f.attrs:
            return float(np.asarray(f.attrs["maxdistance"]).ravel()[0])
    raise ValueError(f"Could not read max STM error from {h5path} attrs")


def _resolve_max_blocks(h5path: str, max_blocks: int) -> int:
    if max_blocks > 0:
        return max_blocks
    return 0


def _link_params_from_env() -> LinkParams:
    y_half = _env_float("BOX_SIZE_Y", 1.5) or 1.5
    z_half = _env_float("BOX_SIZE_Z", 1.5) or 1.5
    return LinkParams(
        box_size_track=_env_float("BOX_SIZE_TRACK", 0.2) or 0.2,
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


def _tracks_path(run_dir: str, threshold: float) -> str:
    return os.path.join(run_dir, f"tracks_error_{threshold:.4f}.h5part")


def _available_detections(masks: list[np.ndarray]) -> int:
    return int(sum(int(np.sum(m)) for m in masks))


def run_error_sweep_for_run(
    run_dir: str,
    *,
    stm_filename: str,
    thresholds: list[float],
    link_params: LinkParams,
    workers: int,
    max_blocks: int,
    min_track_length: int,
    dt: float,
    rep_rate: float,
    position_unit: str,
    skip_existing: bool,
) -> list[dict]:
    h5in = os.path.join(run_dir, stm_filename)
    if not os.path.isfile(h5in):
        raise FileNotFoundError(h5in)

    all_errors = load_all_errors(h5in)
    nframes = get_nframes(h5in)
    if len(all_errors) != nframes:
        raise ValueError(
            f"Error row count ({len(all_errors)}) != frame count ({nframes}) in {h5in}"
        )

    runs_out: list[dict] = []
    for threshold in thresholds:
        out_path = _tracks_path(run_dir, threshold)
        if skip_existing and os.path.isfile(out_path):
            print(f"Skipping error={threshold:.4f} (exists): {out_path}")
            runs_out.append(
                {
                    "error_threshold": threshold,
                    "tracks_h5part": out_path,
                    "skipped": True,
                }
            )
            continue

        masks = [err <= threshold for err in all_errors]
        available = _available_detections(masks)
        print(
            f"=== error <= {threshold:.4f}: {available:,} detections across {nframes} frames ==="
        )

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
                "error_threshold": threshold,
                "available_detections": available,
                "tracks_h5part": out_path,
                "tracking": tracking,
                "skipped": False,
            }
        )

    return runs_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STM error threshold sensitivity tracking sweep.")
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
    p.add_argument("--error-min", type=float, default=_env_float("ERROR_MIN", 0.05))
    p.add_argument("--error-max", type=float, default=_env_float("ERROR_MAX"))
    p.add_argument("--error-step", type=float, default=_env_float("ERROR_STEP", 0.01))
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
        help="Only generate PDF plots from existing tracks_error_*.h5part files.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = resolve_output_root(args.output_dir, args.output_base)
    run_names = parse_runs(args.runs)
    link_params = _link_params_from_env()

    if args.analysis_only:
        for run in run_names:
            run_dir = resolve_run_dir(output_root, args.case, run)
            maybe_run_sensitivity_plots(run_dir, "stm_error")
        return 0

    for run in run_names:
        run_dir = resolve_run_dir(output_root, args.case, run)
        os.makedirs(run_dir, exist_ok=True)
        h5in = os.path.join(run_dir, args.stm_filename)
        if not os.path.isfile(h5in):
            raise FileNotFoundError(h5in)

        error_max = args.error_max
        if error_max is None:
            error_max = _stm_max_error(h5in)
        if args.error_min is None:
            raise ValueError("--error-min is required")
        thresholds = build_float_thresholds(
            start=float(args.error_min),
            end=float(error_max),
            step=float(args.error_step),
        )

        print(f"Run directory: {run_dir}")
        print(f"Thresholds: {thresholds}")

        sweep_runs = run_error_sweep_for_run(
            run_dir,
            stm_filename=args.stm_filename,
            thresholds=thresholds,
            link_params=link_params,
            workers=args.workers,
            max_blocks=_resolve_max_blocks(h5in, args.max_blocks),
            min_track_length=args.min_track_length,
            dt=args.dt,
            rep_rate=args.rep_rate,
            position_unit=args.position_unit,
            skip_existing=args.skip_existing,
        )

        summary = {
            "input_stm": os.path.abspath(h5in),
            "output_dir": os.path.abspath(run_dir),
            "config": {
                "error_min": args.error_min,
                "error_max": error_max,
                "error_step": args.error_step,
                "dt": args.dt,
                "box_size_track": link_params.box_size_track,
                "workers": args.workers,
                "max_blocks": args.max_blocks,
                "min_track_length": args.min_track_length,
                "rep_rate": args.rep_rate,
                "position_unit": args.position_unit,
                "tracks_template": "tracks_error_{error:.4f}.h5part",
                "summary_name": "error_sensitivity_summary.json",
                "link_params": {
                    "box_size_track": link_params.box_size_track,
                    "initial_x_lo": link_params.initial_x_lo,
                    "initial_x_hi": link_params.initial_x_hi,
                    "initial_y_lo": link_params.initial_y_lo,
                    "initial_y_hi": link_params.initial_y_hi,
                    "initial_z_lo": link_params.initial_z_lo,
                    "initial_z_hi": link_params.initial_z_hi,
                    "max_branch": link_params.max_branch,
                    "max_seeds": link_params.max_seeds,
                    "allow_3_frame": link_params.allow_3_frame,
                    "use_bidirectional_graph": link_params.use_bidirectional_graph,
                    "use_numba": link_params.use_numba,
                    "quiet_linking": link_params.quiet_linking,
                    "progress_interval": link_params.progress_interval,
                },
            },
            "thresholds": thresholds,
            "runs": sweep_runs,
            "timestamp_utc": utc_now_iso(),
        }
        write_summary_json(
            os.path.join(run_dir, "error_sensitivity_summary.json"),
            summary,
        )
        maybe_run_sensitivity_plots(run_dir, "stm_error")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
