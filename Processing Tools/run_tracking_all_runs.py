"""
Run 4-frame tracking only for all runs in a case.
Expects stereo matching to have been run first (rays_out_cpp.h5 in each run folder).
Requires 4BE-ETI.

Writes overwriteable tracks_staging.h5part (+ tracks_staging.json). Never overwrites
an existing tracks.h5part — rename staging output manually when satisfied.
"""
import json
import os
import sys
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pipeline_paths import default_data_base, ensure_output_run_dir, resolve_case_paths

os.chdir('/workspaces/4d-ptv-mcflow')
_POSTPROC_DIR = os.path.join('/workspaces/4d-ptv-mcflow', 'Post-Processing')
if _POSTPROC_DIR not in sys.path:
    sys.path.insert(0, _POSTPROC_DIR)

_DEFAULT_CASE = 'TTI_opposing_gravity'


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


try:
    from python_4be_eti.perform_tracking import FourFrameTracking
    from python_4be_eti.utils.basic_utils import (
        TRACKS_H5PART_CANONICAL,
        TRACKS_H5PART_STAGING,
        TRACKS_STAGING_JSON,
        staging_h5part_path,
        staging_json_path,
    )
    from check_tracking import check_tracking_file, print_report, evaluate_tracking, load_tracking_stats
    HAS_TRACKING = True
except ImportError:
    HAS_TRACKING = False


def parse_args():
    p = argparse.ArgumentParser(
        description='Run 4-frame tracking for all runs in a case (stereo matching must be done first).'
    )
    p.add_argument('--case', type=str, default=_DEFAULT_CASE,
                   help='Case name (e.g. TTI_aligned_with_gravity).')
    p.add_argument('--runs', type=str, default=None,
                   help='Comma-separated run names. If omitted, all runs with rays_out_cpp.h5 under output-base are processed.')
    p.add_argument('--output-base', type=str, default=None,
                   help='Parent of the case folder for writing tracks.')
    p.add_argument('--input-base', type=str, default=None,
                   help='Parent of the case folder for reading stereo-matching output. '
                        'Default: same as --output-base.')
    p.add_argument('--rays-filename', type=str, default='rays_out_cpp.h5',
                   help='Rays output filename (default: rays_out_cpp.h5).')
    p.add_argument('--box-size-x', type=float, default=3.5,
                   help='Box size x for track initialization (default: 3.5).')
    p.add_argument('--box-size-initial-x-lo', type=float, default=None,
                   help=('Signed x offset from seed on frame+1 (added to x0); '
                         'physical min x = x0+min(lo,hi), max x = x0+max(lo,hi). '
                         'Default if omitted: -box-size-x. Use negative lo to include points on the -x side of the seed.'))
    p.add_argument('--box-size-initial-x-hi', type=float, default=None,
                   help='Signed x offset from seed on frame+1; default if omitted: +box-size-x.')
    p.add_argument('--box-size-y', type=float, default=1.0,
                   help='Box size y for track initialization (default: 1.0).')
    p.add_argument('--box-size-z', type=float, default=1.0,
                   help='Box size z for track initialization (default: 1.0).')
    p.add_argument('--box-size-track', type=float, default=1.0,
                   help='Box size after track initialized (default: 1.0).')
    p.add_argument('--dt', type=float, default=1e-3,
                   help='Time step (default: 1e-3).')
    p.add_argument('--rep-rate', type=float, default=10.0,
                   help='Repetition rate (default: 10.0).')
    p.add_argument('--workers', type=int, default=8,
                   help='Number of workers for tracking (default: 8).')
    p.add_argument('--max-frame-ranges', type=int, default=None,
                   help='Only process the first N 4-frame batches per run (default: all). '
                        'Overridden by --max-frames when that is set.')
    p.add_argument('--max-frames', type=int, default=None, dest='max_frames',
                   help='Approximate cap on frames processed per run (uses 4-frame batches; '
                        'e.g. 8 → two batches covering frames 0–7). Ignored if --max-frame-ranges is set.')
    p.add_argument('--particle-progress-interval', type=int, default=None,
                   help='Print per-frame ``particle i/n`` every N particles; 0 = off. '
                        'Default: 1000 (verbose) or 0 when --quiet / TRACKING_QUIET / PTV_WEBLAUNCHER.')
    p.add_argument('--quiet', action='store_true',
                   help='Reduced progress output (for weblauncher / batch runs). '
                        'Also enabled by TRACKING_QUIET=1 or PTV_WEBLAUNCHER=1.')
    p.add_argument('--max-candidates-mesh', type=int, default=None,
                   help='3D tracking: max candidates per meshgrid stage (default: 1000).')
    p.add_argument('--max-targets-mesh', type=int, default=None,
                   help='3D tracking: max target particles per meshgrid stage (default: 2000).')
    p.add_argument('--debug-mesh-match-prints', type=int, default=0,
                   help='3D: print mesh-stage array sizes for the first N no_previous_tracks_3d calls with candidates (0=off). Uses NumPy for init when >0.')
    p.add_argument('--position-unit', type=str, default='mm',
                   help='Position unit for x/y/z (e.g. mm, m). Used to scale v/a to SI (default: mm).')
    p.add_argument('--write-failed-tracks', action='store_true',
                   help='Also write particles for which tracking failed (id=0, tracked=False).')
    p.add_argument('--skip-quality-check', action='store_true',
                   help='Do not run Post-Processing/check_tracking.py after each run.')
    return p.parse_args()


def discover_runs(output_base, rays_filename):
    if not os.path.isdir(output_base):
        return []
    runs = []
    for name in sorted(os.listdir(output_base)):
        path = os.path.join(output_base, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, rays_filename)):
            runs.append(name)
    return runs


def _resolve_max_frame_ranges(tracking, *, max_frame_ranges, max_frames):
    """Pick how many 4-frame batches to run (max_frame_ranges wins over max_frames)."""
    if max_frame_ranges is not None:
        return max_frame_ranges
    if max_frames is None:
        return None
    max_frames = int(max_frames)
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1 when set")
    n = min(tracking.num_frames, max_frames)
    tl = tracking.track_length
    return max(1, (n + tl - 1) // tl)


def _args_to_metadata(args, *, case, run, input_path, output_path) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job_id": "tracking",
        "tracker": "four_frame",
        "case": case,
        "run": run,
        "input_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "input_base": args.input_base,
        "output_base": args.output_base,
        "parameters": {
            "rays_filename": args.rays_filename,
            "box_size_x": args.box_size_x,
            "box_size_initial_x_lo": args.box_size_initial_x_lo,
            "box_size_initial_x_hi": args.box_size_initial_x_hi,
            "box_size_y": args.box_size_y,
            "box_size_z": args.box_size_z,
            "box_size_track": args.box_size_track,
            "dt": args.dt,
            "rep_rate": args.rep_rate,
            "workers": args.workers,
            "max_frame_ranges": args.max_frame_ranges,
            "max_frames": args.max_frames,
            "position_unit": args.position_unit,
            "write_failed_tracks": args.write_failed_tracks,
            "max_candidates_mesh": args.max_candidates_mesh,
            "max_targets_mesh": args.max_targets_mesh,
            "debug_mesh_match_prints": args.debug_mesh_match_prints,
            "particle_progress_interval": args.particle_progress_interval,
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


def run_tracking_for_run(input_path, output_path, args, *, verbose: bool, case: str, run: str):
    ensure_output_run_dir(output_path)
    tracking = FourFrameTracking(
        output_path,
        filename=args.rays_filename,
        box_size_x=args.box_size_x, box_size_y=args.box_size_y, box_size_z=args.box_size_z,
        box_size_initial_x_lo=args.box_size_initial_x_lo,
        box_size_initial_x_hi=args.box_size_initial_x_hi,
        box_size_track=args.box_size_track, dt=args.dt, rep_rate=args.rep_rate,
        use_bspline=True,
        position_unit=args.position_unit,
        write_failed_tracks=args.write_failed_tracks,
        particle_progress_interval=args.particle_progress_interval,
        max_candidates_mesh=args.max_candidates_mesh,
        max_targets_mesh=args.max_targets_mesh,
        debug_mesh_match_prints=args.debug_mesh_match_prints,
        verbose=verbose,
        rays_input_path=input_path,
    )
    max_fr = _resolve_max_frame_ranges(
        tracking,
        max_frame_ranges=args.max_frame_ranges,
        max_frames=args.max_frames,
    )
    run_metadata = _args_to_metadata(args, case=case, run=run, input_path=input_path, output_path=output_path)
    run_metadata["parameters"]["max_frame_ranges_resolved"] = max_fr

    staging_path, exec_meta = tracking.run_tracking(
        workers=args.workers,
        max_frame_ranges=max_fr,
        verbose=verbose,
        run_metadata=run_metadata,
    )

    record = dict(run_metadata)
    record.update(exec_meta)
    record["output_h5part"] = TRACKS_H5PART_STAGING
    record["output_h5part_path"] = os.path.abspath(staging_path)
    record["canonical_h5part"] = TRACKS_H5PART_CANONICAL

    if not args.skip_quality_check and os.path.isfile(staging_path):
        print("Running tracking quality check...", flush=True)
        record["quality_check"] = _run_quality_check(staging_path, verbose=verbose)
    else:
        record["quality_check"] = None

    json_path = staging_json_path(output_path)
    _write_run_json(json_path, record)
    print(f"Run metadata written: {json_path}", flush=True)
    return staging_path


def main():
    args = parse_args()
    quiet = (
        args.quiet
        or _env_truthy("TRACKING_QUIET")
        or _env_truthy("PTV_WEBLAUNCHER")
    )
    verbose = not quiet
    if args.particle_progress_interval is None:
        args.particle_progress_interval = 0 if quiet else 1000
    if not HAS_TRACKING:
        print("4BE-ETI is required for tracking. Install the 4BE-ETI package.", file=sys.stderr)
        sys.exit(1)
    if args.max_frames is not None and args.max_frames < 1:
        print('Error: --max-frames must be >= 1', file=sys.stderr)
        sys.exit(1)
    if args.max_frame_ranges is not None and args.max_frame_ranges < 1:
        print('Error: --max-frame-ranges must be >= 1', file=sys.stderr)
        sys.exit(1)

    in_base, out_base = resolve_case_paths(
        args.case,
        input_base=args.input_base,
        output_base=args.output_base,
        default_output_base=default_data_base(),
    )

    if args.runs:
        runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    else:
        runs = discover_runs(in_base, args.rays_filename)

    if not runs:
        print(f"No runs with {args.rays_filename} found under {os.path.abspath(in_base)}", file=sys.stderr)
        sys.exit(1)

    print(f"Tracking (all runs): case={args.case}, runs={runs}"
          + (" [quiet]" if quiet else ""))
    print(
        f"  Output: {TRACKS_H5PART_STAGING} + {TRACKS_STAGING_JSON} "
        f"(existing {TRACKS_H5PART_CANONICAL} is never overwritten)"
    )
    print(
        f"  box_size_x={args.box_size_x}, "
        f"box_size_initial_x_lo={args.box_size_initial_x_lo}, "
        f"box_size_initial_x_hi={args.box_size_initial_x_hi}, "
        f"box_size_y={args.box_size_y}, box_size_z={args.box_size_z}, box_size_track={args.box_size_track}"
    )
    print(
        f"  dt={args.dt}, rep_rate={args.rep_rate}, workers={args.workers}, "
        f"max_frame_ranges={args.max_frame_ranges}, max_frames={args.max_frames}"
    )
    print(f"  position_unit={args.position_unit}")
    print(f"  input_base={in_base}  output_base={out_base}")

    for run in runs:
        in_path = os.path.join(in_base, run)
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(in_path):
            print(f"Skip {run}: missing input {in_path}")
            continue
        print(f"Processing run: {run}")
        run_tracking_for_run(in_path, out_path, args, verbose=verbose, case=args.case, run=run)
        print(f"Done {run}")
    print("All runs completed.")


if __name__ == '__main__':
    main()
