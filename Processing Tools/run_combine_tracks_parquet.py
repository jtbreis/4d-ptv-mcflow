#!/usr/bin/env python3
"""Combine per-run track .h5part files into one Parquet table (weblauncher / CLI).

Example:
  python3 "Processing Tools/run_combine_tracks_parquet.py" \\
    --output-dir /mnt/ssd \\
    --output-base data/Windtunnel_PTV_TNTI/X300cm_PTV_below_TNTI \\
    --case TNTI_opposing_gravity \\
    --runs Run1,Run2 \\
    --tracks-name tracks.h5part
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys


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


def discover_runs(case_dir: str, tracks_name: str) -> list[str]:
    if not os.path.isdir(case_dir):
        return []
    found = []
    for name in sorted(os.listdir(case_dir)):
        if not re.match(r"^Run\d+$", name, flags=re.I):
            continue
        if os.path.isfile(os.path.join(case_dir, name, tracks_name)):
            found.append(name)
    return found


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Combine per-run track h5part files into one Parquet file."
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
        required=not bool(os.environ.get("CASE")),
        help="Case folder name under output-base.",
    )
    p.add_argument(
        "--runs",
        default=os.environ.get("RUNS", ""),
        help="Comma-separated Run names (empty = all Run* with the tracks file).",
    )
    p.add_argument(
        "--tracks-name",
        default=os.environ.get("TRACKS_NAME", "tracks.h5part"),
        help="Track filename inside each run directory.",
    )
    p.add_argument(
        "--output-name",
        default=os.environ.get("OUTPUT_NAME", ""),
        help="Parquet filename under the case dir (default: {case}_{tracks_stem}.parquet).",
    )
    p.add_argument(
        "--output",
        default=os.environ.get("OUTPUT_PATH", ""),
        help="Full output parquet path (overrides --output-name).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "8") or "8"),
        help="Parallel run loaders.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    case = (args.case or "").strip()
    if not case:
        print("Error: --case / CASE is required", file=sys.stderr)
        return 1

    tracks_name = (args.tracks_name or "tracks.h5part").strip()
    case_dir = os.path.join(args.output_dir, args.output_base, case)
    case_dir = os.path.normpath(case_dir)

    runs = parse_runs(args.runs)
    if runs is None:
        runs = discover_runs(case_dir, tracks_name)
        print(f"Discovered runs with {tracks_name}: {', '.join(runs) or '(none)'}")
    if not runs:
        print(f"Error: no runs to combine under {case_dir}", file=sys.stderr)
        return 1

    if args.output.strip():
        parquet_path = args.output.strip()
    elif args.output_name.strip():
        parquet_path = os.path.join(case_dir, args.output_name.strip())
    else:
        stem = os.path.splitext(os.path.basename(tracks_name))[0]
        parquet_path = os.path.join(case_dir, f"{case}_{stem}.parquet")

    print(f"Case dir: {case_dir}")
    print(f"Tracks file: {tracks_name}")
    print(f"Runs: {', '.join(runs)}")
    print(f"Output: {parquet_path}")

    mod = _load_combine_module()
    out = mod.combine_to_parquet(
        case_dir=case_dir,
        runs=runs,
        tracks_name=tracks_name,
        parquet_path=parquet_path,
        max_workers=args.workers,
    )
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
