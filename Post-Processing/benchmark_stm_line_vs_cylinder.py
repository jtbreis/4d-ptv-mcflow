#!/usr/bin/env python3
"""Benchmark line STM vs cylinder STM on the same run and frame count."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from python_4dptv.matching.stereomatching import StereoMatching
from python_4dptv.utils.structure import Filenames


@dataclass
class FrameStats:
    frame_index: int
    n_matches: int
    median_matcherror: float
    p95_matcherror: float
    frac_4cam: float
    max_matches_per_ray: int
    mean_matches_per_ray: float
    frac_rays_multi_match: float


def _frame_keys(h5path: str, nframes: int) -> list[str]:
    with h5py.File(h5path, "r") as f:
        keys = sorted(
            [k for k in f.keys() if k.startswith("frame")],
            key=lambda k: int(k.replace("frame", "")),
        )
    return keys[:nframes]


def analyze_stm_output(h5path: str, nframes: int) -> tuple[list[FrameStats], dict]:
    per_frame: list[FrameStats] = []
    all_err: list[float] = []
    all_ncams: list[int] = []
    matches_per_ray_counter: Counter[int] = Counter()

    for fi, fk in enumerate(_frame_keys(h5path, nframes)):
        with h5py.File(h5path, "r") as f:
            xyze = np.asarray(f[fk]["xyze"], dtype=np.float64)
            camrayids = np.asarray(f[fk]["camrayids"], dtype=np.int64)
        if xyze.size == 0:
            per_frame.append(
                FrameStats(fi, 0, float("nan"), float("nan"), float("nan"), 0, 0.0, 0.0)
            )
            continue

        n_matches = int(xyze.shape[1])
        err = xyze[3]
        ncams_per_match = camrayids.shape[0] // 2

        ray_use: Counter[tuple[int, int]] = Counter()
        for j in range(n_matches):
            for k in range(ncams_per_match):
                cam = int(camrayids[2 * k, j])
                ray = int(camrayids[2 * k + 1, j])
                ray_use[(cam, ray)] += 1

        counts = np.fromiter(ray_use.values(), dtype=np.int64)
        max_mpr = int(counts.max()) if counts.size else 0
        mean_mpr = float(counts.mean()) if counts.size else 0.0
        frac_multi = float(np.mean(counts > 1)) if counts.size else 0.0

        for c in counts:
            matches_per_ray_counter[int(c)] += int(c)

        per_frame.append(
            FrameStats(
                frame_index=fi,
                n_matches=n_matches,
                median_matcherror=float(np.median(err)),
                p95_matcherror=float(np.percentile(err, 95)),
                frac_4cam=float(ncams_per_match >= 4),
                max_matches_per_ray=max_mpr,
                mean_matches_per_ray=mean_mpr,
                frac_rays_multi_match=frac_multi,
            )
        )
        all_err.extend(err.tolist())
        all_ncams.extend([ncams_per_match] * n_matches)

    summary = {
        "n_frames": len(per_frame),
        "total_matches": int(sum(f.n_matches for f in per_frame)),
        "mean_matches_per_frame": float(np.mean([f.n_matches for f in per_frame])),
        "median_matcherror": float(np.median(all_err)) if all_err else float("nan"),
        "p95_matcherror": float(np.percentile(all_err, 95)) if all_err else float("nan"),
        "frac_4cam": float(np.mean(np.array(all_ncams) >= 4)) if all_ncams else float("nan"),
        "mean_frac_rays_multi_match": float(np.mean([f.frac_rays_multi_match for f in per_frame])),
        "max_matches_per_ray_overall": int(max((f.max_matches_per_ray for f in per_frame), default=0)),
        "matches_per_ray_histogram": {str(k): v for k, v in sorted(matches_per_ray_counter.items())},
    }
    return per_frame, summary


def run_stm(
    *,
    input_run: str,
    output_run: str,
    nframes: int,
    cylinder: bool,
    mincameras: int,
    maxdistance: float,
    multiplematchesperraydistance: float,
    maxmatchesperray: int,
    nvoxels: list[int],
    boundingbox: list[float],
    nthreads: int,
    spatial_boxes: int,
) -> float:
    os.makedirs(output_run, exist_ok=True)
    sm = StereoMatching(
        output_run,
        mincameras,
        maxdistance,
        multiplematchesperraydistance,
        maxmatchesperray,
        nvoxels,
        boundingbox,
        rays_input_path=input_run,
    )
    t0 = time.perf_counter()
    sm.run_stereomatching(
        nthreads=nthreads,
        nframes=nframes,
        timing=True,
        spatial_boxes=spatial_boxes,
        spatial_overlap_cells=1,
        frame_parallelism=min(4, nthreads),
        cylinder=cylinder,
        cylinder_max_shells=32,
        write_particle_props=True,
    )
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-run",
        default="processed_data/Windtunnel_PTV_TNTI/X300cm_PTV_center_TNTI/TNTI_aligned_with_gravity/Run4",
    )
    p.add_argument("--output-base", default=None)
    p.add_argument("--nframes", type=int, default=48)
    p.add_argument("--min-cameras", type=int, default=3)
    p.add_argument("--max-distance", type=float, default=0.15)
    p.add_argument("--multiple-matches-per-ray-distance", type=float, default=0.5)
    p.add_argument("--max-matches-per-ray", type=int, default=4)
    p.add_argument("--nvoxels", default="1100,700,400")
    p.add_argument("--bounding-box", default="-55,55,-35,35,-20,20")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--spatial-boxes", type=int, default=16)
    p.add_argument("--skip-run", action="store_true")
    args = p.parse_args()

    input_run = os.path.abspath(args.input_run)
    rays_file = input_run + Filenames.RAYS.value
    if not os.path.isfile(rays_file):
        raise FileNotFoundError(f"Missing {rays_file}")

    nvoxels = [int(x) for x in args.nvoxels.split(",")]
    bbox = [float(x) for x in args.bounding_box.split(",")]
    out_parent = args.output_base or os.path.join(
        input_run, f"stm_benchmark_{args.nframes}f"
    )
    line_out = os.path.join(out_parent, "line")
    cyl_out = os.path.join(out_parent, "cylinder")

    results: dict = {
        "input_run": input_run,
        "nframes": args.nframes,
        "parameters": {
            "min_cameras": args.min_cameras,
            "max_distance": args.max_distance,
            "multiple_matches_per_ray_distance": args.multiple_matches_per_ray_distance,
            "max_matches_per_ray": args.max_matches_per_ray,
            "nvoxels": nvoxels,
            "bounding_box": bbox,
            "spatial_boxes": args.spatial_boxes,
        },
        "modes": {},
    }

    for label, out_dir, cylinder in (
        ("line", line_out, False),
        ("cylinder", cyl_out, True),
    ):
        stm_path = out_dir + Filenames.STM.value
        if not args.skip_run:
            print(f"\n=== Running {label} STM ({args.nframes} frames) -> {out_dir} ===")
            elapsed = run_stm(
                input_run=input_run,
                output_run=out_dir,
                nframes=args.nframes,
                cylinder=cylinder,
                mincameras=args.min_cameras,
                maxdistance=args.max_distance,
                multiplematchesperraydistance=args.multiple_matches_per_ray_distance,
                maxmatchesperray=args.max_matches_per_ray,
                nvoxels=nvoxels,
                boundingbox=bbox,
                nthreads=args.threads,
                spatial_boxes=args.spatial_boxes,
            )
            results["modes"][label] = {"output_dir": out_dir, "wall_seconds": elapsed}
        elif not os.path.isfile(stm_path):
            raise FileNotFoundError(f"Missing {stm_path}; run without --skip-run first")

        print(f"\n=== Analyzing {label} -> {stm_path} ===")
        _frames, summary = analyze_stm_output(stm_path, args.nframes)
        results["modes"].setdefault(label, {})["output_dir"] = out_dir
        results["modes"][label]["summary"] = summary

    line_s = results["modes"]["line"]["summary"]
    cyl_s = results["modes"]["cylinder"]["summary"]
    results["comparison"] = {
        "delta_total_matches": cyl_s["total_matches"] - line_s["total_matches"],
        "delta_mean_matches_per_frame": cyl_s["mean_matches_per_frame"]
        - line_s["mean_matches_per_frame"],
        "delta_median_matcherror": cyl_s["median_matcherror"] - line_s["median_matcherror"],
        "delta_frac_rays_multi_match": cyl_s["mean_frac_rays_multi_match"]
        - line_s["mean_frac_rays_multi_match"],
        "cylinder_match_reduction_pct": 100.0
        * (1.0 - cyl_s["total_matches"] / line_s["total_matches"])
        if line_s["total_matches"]
        else float("nan"),
    }

    os.makedirs(out_parent, exist_ok=True)
    summary_path = os.path.join(out_parent, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Benchmark summary ===")
    print(json.dumps(results["comparison"], indent=2))
    for mode in ("line", "cylinder"):
        print(f"\n{mode.upper()}:")
        print(json.dumps(results["modes"][mode]["summary"], indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
