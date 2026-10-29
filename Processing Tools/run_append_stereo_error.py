#!/usr/bin/env python3
"""Append STM stereomatching error onto per-run tracks.h5part files.

Looks up each tracked particle in rays_out_cpp.h5 (exact x,y,z) and writes:
  error      RMS match error from xyze row 3
  error_xy   RMS of the (x, y) residual of each contributing ray
  error_z    RMS of the depth (z) residual of each contributing ray
  error_0..  perpendicular distance to each contributing ray (needs rays.h5)

Example:
  python3 "Processing Tools/run_append_stereo_error.py" \\
    --output-dir /workspaces/4d-ptv-mcflow \\
    --output-base data/Windtunnel_PTV_TNTI/X300cm_PTV_center_TNTI \\
    --case TNTI_opposing_gravity \\
    --update-parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline_paths import default_data_base  # noqa: E402


def _load_append_module():
    path = os.path.join(
        REPO_ROOT, "Post-Processing", "4_frame_tracking", "append_stereo_error.py"
    )
    spec = importlib.util.spec_from_file_location("append_stereo_error_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_combine_module():
    path = os.path.join(
        REPO_ROOT, "Post-Processing", "4_frame_tracking", "00_combine_tracks.py"
    )
    spec = importlib.util.spec_from_file_location("combine_tracks_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_runs(runs: str | None) -> list[str] | None:
    if runs is None or not str(runs).strip():
        return None
    return [p.strip() for p in str(runs).split(",") if p.strip()]


def discover_runs(case_dir: str, tracks_name: str, stm_filename: str) -> list[str]:
    if not os.path.isdir(case_dir):
        return []
    found = []
    append_mod = None
    for name in sorted(os.listdir(case_dir)):
        if not re.match(r"^Run\d+$", name, flags=re.I):
            continue
        run_dir = os.path.join(case_dir, name)
        tracks_path = os.path.join(run_dir, tracks_name)
        stm_path = os.path.join(run_dir, stm_filename)
        if not (os.path.isfile(tracks_path) and os.path.isfile(stm_path)):
            continue
        if append_mod is None:
            append_mod = _load_append_module()
        if not append_mod.h5part_has_steps(tracks_path):
            continue
        found.append(name)
    return found


def _run_one(kwargs):
    mod = _load_append_module()
    return mod.append_stereo_error_for_run(**kwargs)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Append STM stereomatching error onto tracks.h5part files."
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", REPO_ROOT),
        help="Host/data root (default: OUTPUT_DIR or repo root).",
    )
    p.add_argument(
        "--output-base",
        default=os.environ.get("OUTPUT_BASE") or default_data_base(),
        help="Base under output-dir containing {CASE}/RunN/.",
    )
    p.add_argument(
        "--case",
        default=os.environ.get("CASE", ""),
        help="Case folder name. Empty with --all-cases under output-base.",
    )
    p.add_argument(
        "--all-cases",
        action="store_true",
        help="Process every case directory under output-base that has Run*/tracks.",
    )
    p.add_argument(
        "--runs",
        default=os.environ.get("RUNS", ""),
        help="Comma-separated Run names (empty = all Run* with tracks + STM).",
    )
    p.add_argument(
        "--tracks-name",
        default=os.environ.get("TRACKS_NAME", "tracks.h5part"),
        help="Track filename inside each run directory.",
    )
    p.add_argument(
        "--stm-filename",
        default=os.environ.get("STM_FILENAME", "rays_out_cpp.h5"),
        help="STM filename inside each run directory.",
    )
    p.add_argument(
        "--rays-filename",
        default=os.environ.get("RAYS_FILENAME", "rays.h5"),
        help="rays.h5 filename (looked up in run dir or processed_data mirror).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1") or "1"),
        help="Parallel run processes (default 1). HDF5 I/O bound.",
    )
    p.add_argument(
        "--rebuild-parquet",
        action="store_true",
        help="After updating h5part files, rebuild the combined case parquet.",
    )
    p.add_argument(
        "--update-parquet",
        action="store_true",
        help="Add error / error_xy / error_z onto the existing case parquet by "
        "recomputing residuals from STM + rays (does not require complete h5part).",
    )
    p.add_argument(
        "--parquet-only",
        action="store_true",
        help="Skip writing tracks.h5part; only update parquet files.",
    )
    p.add_argument(
        "--max-dist",
        type=float,
        default=float(os.environ.get("MAX_DIST", "0") or "0"),
        help="If >0, nearest-STM match within this distance (mm) when exact xyz fails.",
    )
    p.add_argument(
        "--output-name",
        default=os.environ.get("OUTPUT_NAME", ""),
        help="Parquet filename under the case dir (with --update-parquet or --rebuild-parquet).",
    )
    return p.parse_args(argv)


def _case_dirs(args) -> list[str]:
    base = os.path.normpath(os.path.join(args.output_dir, args.output_base))
    append_mod = _load_append_module()
    if args.all_cases:
        if not os.path.isdir(base):
            return []
        cases = []
        for name in sorted(os.listdir(base)):
            cdir = os.path.join(base, name)
            if not os.path.isdir(cdir):
                continue
            has_runs = bool(
                discover_runs(cdir, args.tracks_name, args.stm_filename)
            )
            has_parquet = append_mod.infer_tracks_parquet(
                cdir, args.tracks_name
            ) is not None
            if has_runs or (args.update_parquet and has_parquet):
                cases.append(cdir)
        return cases
    case = (args.case or "").strip()
    if not case:
        raise SystemExit("Error: --case is required unless --all-cases is set")
    return [os.path.normpath(os.path.join(base, case))]


def main(argv=None):
    args = parse_args(argv)
    case_dirs = _case_dirs(args)
    if not case_dirs:
        print("No cases found.", file=sys.stderr)
        return 1

    jobs = []
    if not args.parquet_only:
        for case_dir in case_dirs:
            runs = parse_runs(args.runs)
            if runs is None:
                runs = discover_runs(case_dir, args.tracks_name, args.stm_filename)
                print(
                    f"{os.path.basename(case_dir)}: discovered "
                    f"{', '.join(runs) or '(none)'}"
                )
            for run in runs:
                jobs.append(
                    {
                        "run_dir": os.path.join(case_dir, run),
                        "tracks_name": args.tracks_name,
                        "stm_filename": args.stm_filename,
                        "rays_filename": args.rays_filename,
                        "show_progress": args.workers <= 1,
                    }
                )

        if not jobs and not args.update_parquet:
            print("No runs to process.", file=sys.stderr)
            return 1

        if jobs:
            print(f"Appending stereo error to {len(jobs)} run(s)")
            results = []
            workers = max(1, int(args.workers))
            if workers == 1 or len(jobs) == 1:
                mod = _load_append_module()
                for job in jobs:
                    results.append(mod.append_stereo_error_for_run(**job))
            else:
                with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                    futs = {ex.submit(_run_one, job): job for job in jobs}
                    for fut in as_completed(futs):
                        job = futs[fut]
                        try:
                            results.append(fut.result())
                        except Exception as e:
                            print(f"FAILED {job['run_dir']}: {e}", file=sys.stderr)
                            raise

            for r in results:
                print(
                    f"  {r['tracks_path']}: matched {r['n_matched']:,}/{r['n_rows']:,} "
                    f"(miss {r['n_miss']:,})"
                )

    if args.update_parquet:
        mod = _load_append_module()
        for case_dir in case_dirs:
            if args.output_name.strip():
                parquet_path = os.path.join(case_dir, args.output_name.strip())
            else:
                parquet_path = mod.infer_tracks_parquet(case_dir, args.tracks_name)
            if parquet_path is None:
                print(f"No parquet found under {case_dir}, skipping")
                continue
            print(f"Updating parquet {parquet_path} …")
            info = mod.append_stereo_error_to_parquet(
                parquet_path,
                case_dir,
                stm_filename=args.stm_filename,
                rays_filename=args.rays_filename,
                max_dist=args.max_dist,
            )
            print(
                f"  {info['parquet_path']}: matched {info['n_matched']:,}/"
                f"{info['n_rows']:,} (miss {info['n_miss']:,})"
            )

    if args.rebuild_parquet:
        comb = _load_combine_module()
        append_mod = _load_append_module()
        for case_dir in case_dirs:
            runs = parse_runs(args.runs)
            if runs is None:
                runs = discover_runs(case_dir, args.tracks_name, args.stm_filename)
            if args.output_name.strip():
                parquet_path = os.path.join(case_dir, args.output_name.strip())
            else:
                parquet_path = append_mod.infer_tracks_parquet(
                    case_dir, args.tracks_name
                )
            print(f"Rebuilding parquet for {os.path.basename(case_dir)} …")
            comb.combine_to_parquet(
                case_dir=case_dir,
                runs=runs,
                tracks_name=args.tracks_name,
                parquet_path=parquet_path,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
