"""
Run graph-based 4-frame tracking for all runs in a case.

Reads rays_out_cpp.h5 per run; writes tracks_graph_staging.h5part (+ JSON metadata).
Never overwrites tracks_graph.h5part — rename staging output manually when satisfied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pipeline_paths import default_data_base, ensure_output_run_dir, resolve_case_paths

os.chdir("/workspaces/4d-ptv-mcflow")

_POSTPROC_DIR = os.path.join("/workspaces/4d-ptv-mcflow", "Post-Processing")
if _POSTPROC_DIR not in sys.path:
    sys.path.insert(0, _POSTPROC_DIR)

from graph_tracking.link_block import LinkParams
from graph_tracking.load_data import get_nframes, iter_block_starts
from graph_tracking.run_tracking import run_graph_tracking

try:
    from python_4be_eti.utils.basic_utils import write_h5_tracking_metadata
except ImportError:
    write_h5_tracking_metadata = None

TRACKS_GRAPH_H5PART_STAGING = "tracks_graph_staging.h5part"
TRACKS_GRAPH_STAGING_JSON = "tracks_graph_staging.json"
TRACKS_GRAPH_H5PART_CANONICAL = "tracks_graph.h5part"

_DEFAULT_CASE = "TTI_opposing_gravity"


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def staging_h5part_path(folder: str) -> str:
    return os.path.join(folder, TRACKS_GRAPH_H5PART_STAGING)


def staging_json_path(folder: str) -> str:
    return os.path.join(folder, TRACKS_GRAPH_STAGING_JSON)


try:
    from check_tracking import check_tracking_file, evaluate_tracking, load_tracking_stats, print_report

    HAS_QUALITY_CHECK = True
except ImportError:
    HAS_QUALITY_CHECK = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Run graph-based 4-frame tracking for all runs in a case."
    )
    p.add_argument("--case", type=str, default=_DEFAULT_CASE)
    p.add_argument("--runs", type=str, default=None)
    p.add_argument("--output-base", type=str, default=None)
    p.add_argument("--input-base", type=str, default=None)
    p.add_argument("--rays-filename", type=str, default="rays_out_cpp.h5")
    p.add_argument("--box-size-initial-x-lo", type=float, default=None)
    p.add_argument("--box-size-initial-x-hi", type=float, default=None)
    p.add_argument("--box-size-y", type=float, default=1.5,
                   help="Half-width of initial y search (±mm from seed).")
    p.add_argument("--box-size-z", type=float, default=1.5,
                   help="Half-width of initial z search (±mm from seed).")
    p.add_argument("--box-size-track", type=float, default=0.2)
    p.add_argument("--max-branch", type=int, default=30)
    p.add_argument("--max-seeds", type=int, default=0)
    p.add_argument("--min-track-length", type=int, default=3)
    p.add_argument("--max-blocks", type=int, default=0,
                   help="Max 4-frame blocks per run (0 = all).")
    p.add_argument("--max-frame-ranges", type=int, default=None,
                   help="Alias for --max-blocks when set.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames per run; converted to block count when max-blocks unset.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel workers for load-balanced 4-frame block batches.")
    p.add_argument("--use-bidirectional-graph", action="store_true")
    p.add_argument("--no-3-frame", action="store_true")
    p.add_argument("--no-numba", action="store_true")
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--rep-rate", type=float, default=10.0)
    p.add_argument("--position-unit", type=str, default="mm")
    p.add_argument("--progress-interval", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--skip-quality-check", action="store_true")
    return p.parse_args()


def discover_runs(input_base: str, rays_filename: str) -> list[str]:
    if not os.path.isdir(input_base):
        return []
    runs = []
    for name in sorted(os.listdir(input_base)):
        path = os.path.join(input_base, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, rays_filename)):
            runs.append(name)
    return runs


def _resolve_initial_x(args) -> tuple[float, float]:
    lo = args.box_size_initial_x_lo
    hi = args.box_size_initial_x_hi
    if lo is None:
        lo = 0.0
    if hi is None:
        hi = 3.5
    return float(lo), float(hi)


def _resolve_max_blocks(h5path: str, *, max_blocks: int, max_frame_ranges, max_frames) -> int:
    if max_frame_ranges is not None and int(max_frame_ranges) > 0:
        return int(max_frame_ranges)
    if max_blocks > 0:
        return int(max_blocks)
    if max_frames is not None and int(max_frames) > 0:
        nframes = get_nframes(h5path)
        n = min(nframes, int(max_frames))
        return max(1, (n + 3) // 4)
    return 0


def _link_params_from_args(args, *, quiet: bool) -> LinkParams:
    x_lo, x_hi = _resolve_initial_x(args)
    y_half = float(args.box_size_y)
    z_half = float(args.box_size_z)
    progress = args.progress_interval
    if progress is None:
        progress = 0 if quiet else 5000
    params = LinkParams(
        box_size_track=args.box_size_track,
        initial_x_lo=x_lo,
        initial_x_hi=x_hi,
        initial_y_lo=-y_half,
        initial_y_hi=y_half,
        initial_z_lo=-z_half,
        initial_z_hi=z_half,
        max_branch=args.max_branch,
        max_seeds=args.max_seeds,
        allow_3_frame=not args.no_3_frame,
        use_bidirectional_graph=args.use_bidirectional_graph,
        use_numba=not args.no_numba,
        quiet_linking=quiet,
        progress_interval=progress,
    )
    return params


def _args_to_metadata(args, *, case, run, input_path, output_path, max_blocks_resolved) -> dict:
    x_lo, x_hi = _resolve_initial_x(args)
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job_id": "graph_tracking",
        "tracker": "graph_tracking",
        "case": case,
        "run": run,
        "input_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "input_base": args.input_base,
        "output_base": args.output_base,
        "parameters": {
            "rays_filename": args.rays_filename,
            "box_size_initial_x_lo": x_lo,
            "box_size_initial_x_hi": x_hi,
            "box_size_y": args.box_size_y,
            "box_size_z": args.box_size_z,
            "box_size_track": args.box_size_track,
            "max_branch": args.max_branch,
            "max_seeds": args.max_seeds,
            "min_track_length": args.min_track_length,
            "max_blocks": args.max_blocks,
            "max_frame_ranges": args.max_frame_ranges,
            "max_frames": args.max_frames,
            "max_blocks_resolved": max_blocks_resolved,
            "workers": args.workers,
            "use_bidirectional_graph": args.use_bidirectional_graph,
            "no_3_frame": args.no_3_frame,
            "use_numba": not args.no_numba,
            "dt": args.dt,
            "rep_rate": args.rep_rate,
            "position_unit": args.position_unit,
            "progress_interval": args.progress_interval,
        },
    }


def _write_run_json(json_path: str, record: dict) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")


def _run_quality_check(staging_path: str, *, verbose: bool) -> dict:
    report = check_tracking_file(staging_path)
    if verbose:
        stats = load_tracking_stats(staging_path)
        result = evaluate_tracking(
            stats,
            min_particles_per_frame=report["thresholds"]["min_particles_per_frame"],
            min_finite_vz=report["thresholds"]["min_finite_vz"],
            max_abs_skew=report["thresholds"]["max_abs_skew"],
            max_abs_excess_kurtosis=report["thresholds"]["max_abs_excess_kurtosis"],
            min_qq_correlation=report["thresholds"]["min_qq_correlation"],
        )
        print_report(
            stats,
            result,
            min_particles_per_frame=report["thresholds"]["min_particles_per_frame"],
        )
    else:
        ok = report["overall_ok"]
        print(
            f"Quality check: {'PASS' if ok else 'FAIL'} "
            f"(mean particles/frame={report['mean_particles']:.0f}, "
            f"v_z Q-Q r={report['vz_metrics'].get('qq_correlation', float('nan')):.4f})",
            flush=True,
        )
    return report


def run_graph_tracking_for_run(
    input_run_dir: str,
    output_run_dir: str,
    args,
    *,
    verbose: bool,
    case: str,
    run: str,
) -> str:
    ensure_output_run_dir(output_run_dir)
    h5in = os.path.join(input_run_dir, args.rays_filename)
    if not os.path.isfile(h5in):
        raise FileNotFoundError(h5in)

    staging_path = staging_h5part_path(output_run_dir)
    max_blocks = _resolve_max_blocks(
        h5in,
        max_blocks=args.max_blocks,
        max_frame_ranges=args.max_frame_ranges,
        max_frames=args.max_frames,
    )
    params = _link_params_from_args(args, quiet=not verbose)

    summary = run_graph_tracking(
        h5in,
        output_run_dir,
        output=staging_path,
        max_blocks=max_blocks,
        workers=args.workers,
        link_params=params,
        min_track_length=args.min_track_length,
        dt=args.dt,
        rep_rate=args.rep_rate,
        position_unit=args.position_unit,
        show_write_progress=verbose,
        verbose=verbose,
    )

    record = _args_to_metadata(
        args,
        case=case,
        run=run,
        input_path=h5in,
        output_path=output_run_dir,
        max_blocks_resolved=max_blocks,
    )
    record.update(summary)
    record["output_h5part"] = TRACKS_GRAPH_H5PART_STAGING
    record["output_h5part_path"] = os.path.abspath(staging_path)
    record["canonical_h5part"] = TRACKS_GRAPH_H5PART_CANONICAL

    if (
        not args.skip_quality_check
        and HAS_QUALITY_CHECK
        and os.path.isfile(staging_path)
    ):
        print("Running tracking quality check...", flush=True)
        record["quality_check"] = _run_quality_check(staging_path, verbose=verbose)
    else:
        record["quality_check"] = None

    json_path = staging_json_path(output_run_dir)
    _write_run_json(json_path, record)
    if os.path.isfile(staging_path) and write_h5_tracking_metadata is not None:
        write_h5_tracking_metadata(staging_path, record)
    print(f"Run metadata written: {json_path}", flush=True)
    return staging_path


def main():
    args = parse_args()
    if _env_truthy("USE_BIDIRECTIONAL_GRAPH"):
        args.use_bidirectional_graph = True
    quiet = (
        args.quiet
        or _env_truthy("TRACKING_QUIET")
        or _env_truthy("PTV_WEBLAUNCHER")
    )
    verbose = not quiet

    in_base, out_base = resolve_case_paths(
        args.case,
        input_base=args.input_base,
        output_base=args.output_base,
        default_output_base=default_data_base(),
    )

    if args.runs:
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    else:
        runs = discover_runs(in_base, args.rays_filename)

    if not runs:
        print(f"No runs with {args.rays_filename} found under {os.path.abspath(in_base)}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Graph tracking (all runs): case={args.case}, runs={runs}"
        + (" [quiet]" if quiet else "")
    )
    print(
        f"  Output: {TRACKS_GRAPH_H5PART_STAGING} + {TRACKS_GRAPH_STAGING_JSON} "
        f"(existing {TRACKS_GRAPH_H5PART_CANONICAL} is never overwritten)"
    )
    print(f"  input_base={in_base}  output_base={out_base}")

    for run in runs:
        in_path = os.path.join(in_base, run)
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(in_path):
            print(f"Skip {run}: missing input {in_path}")
            continue
        print(f"Processing run: {run}")
        run_graph_tracking_for_run(in_path, out_path, args, verbose=verbose, case=args.case, run=run)
        print(f"Done {run}")
    print("All runs completed.")


if __name__ == "__main__":
    main()
