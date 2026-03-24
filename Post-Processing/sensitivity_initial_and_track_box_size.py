"""
Grid sensitivity: combine initial box sizes (symmetric x/y/z) and track box_size.

For each tuple (box_size_x, box_size_y, box_size_z) from the Cartesian product
of --initial-x, --initial-y, --initial-z lists, and each --track-box-size,
runs 4-frame tracking and saves tracks to a unique .h5part filename.

Optionally set asymmetric initial bounds for x and y with --initial-x-lo,
--initial-x-hi, --initial-y-lo, --initial-y-hi (comma-separated lists; Cartesian
product with each other and with initial x/y/z and track sizes). Omit a flag to
use FourFrameTracking symmetric defaults from --initial-x/y/z for that bound.
z asymmetric bounds: --initial-z-lo / --initial-z-hi (same pattern).

Parallel runs (e.g. 24 processes or 24 Docker containers): use the same CLI on
each process with a different --shard i/N (0 <= i < N). Each shard only runs jobs
where job_index % N == i. Shards use isolated work dirs (symlink to rays) so they
do not clobber the same tracks.h5part. Merge CSVs after: --merge-manifests.

Docker: split across N containers with docker/run_sensitivity_initial_and_track_parallel.sh
(NUM_SHARDS=24). Or set SHARD=i/N per container. Single container: run_sensitivity_initial_and_track.sh

Run from project root:
  python Post-Processing/sensitivity_initial_and_track_box_size.py
  python Post-Processing/sensitivity_initial_and_track_box_size.py \\
    --initial-x 2.5,3.5 --initial-y 0.8,1.0 --track-box-sizes 0.5,1.0
  python ... --shard=3/24 --workers 1   # shard 3 of 24; workers=1 per shard
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


class Tee:
    """Write to both stdout and a log file, flushing so tail -f works."""

    def __init__(self, log_path: str):
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


DEFAULT_DATASET = os.path.join(
    PROJECT_ROOT, "data", "tracking_test_threshold1")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "sensitivity_initial_and_track", "tracking_test_threshold1"
)
DEFAULT_INITIAL_1D = [3.5, 3.0, 2.5]
DEFAULT_TRACK_BOX_SIZES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

DT = 1e-3
REP_RATE = 10
RAYS_FILENAME = "rays_out_cpp.h5"
TRACKS_FILENAME = "tracks.h5part"


def _parse_float_list(s: str | None, default: list[float]) -> list[float]:
    if not s or not s.strip():
        return list(default)
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_asym_list(s: str | None) -> list[float | None]:
    """
    Comma-separated floats for one bound (lo/hi). Empty / omitted flag => [None]
    meaning do not pass that keyword (symmetric default from initial x/y/z).
    """
    if s is None or not str(s).strip():
        return [None]
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _fmt_bound(v: float | None) -> str:
    if v is None:
        return "sym"
    return f"{v:.2f}".replace(".", "p")


def _tracks_filename(
    ix: float,
    iy: float,
    iz: float,
    ttrack: float,
    x_lo: float | None,
    x_hi: float | None,
    y_lo: float | None,
    y_hi: float | None,
    z_lo: float | None,
    z_hi: float | None,
) -> str:
    base = f"tracks_init_{ix:.2f}_{iy:.2f}_{iz:.2f}_track_{ttrack:.2f}"
    asym = (
        f"_xl{_fmt_bound(x_lo)}_xh{_fmt_bound(x_hi)}"
        f"_yl{_fmt_bound(y_lo)}_yh{_fmt_bound(y_hi)}"
        f"_zl{_fmt_bound(z_lo)}_zh{_fmt_bound(z_hi)}"
    )
    # Shorten if all symmetric
    if asym == "_xlsym_xhsym_ylsym_yhsym_zlsym_zhsym":
        return f"{base}.h5part"
    return f"{base}{asym}.h5part"


def parse_shard(shard_str: str | None) -> tuple[int | None, int | None]:
    """Parse 'i/N' -> (i, N). None if shard_str is empty."""
    if shard_str is None or not str(shard_str).strip():
        return None, None
    parts = str(shard_str).strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"--shard must be i/N, got {shard_str!r}")
    i, n = int(parts[0].strip()), int(parts[1].strip())
    if n < 1:
        raise ValueError("shard denominator N must be >= 1")
    if not (0 <= i < n):
        raise ValueError(f"shard index i must satisfy 0 <= i < N, got {i}/{n}")
    return i, n


def enumerate_jobs(
    initial_xyz: list[tuple[float, float, float]],
    asym_bounds: list[tuple[float | None, float | None, float | None, float | None, float | None, float | None]],
    track_box_sizes: list[float],
) -> list[tuple[float, float, float, tuple, float]]:
    """Flat list of (ix, iy, iz, asym_6tuple, ttrack) in deterministic order."""
    jobs = []
    for ix, iy, iz in initial_xyz:
        for ab in asym_bounds:
            for ttrack in track_box_sizes:
                jobs.append((ix, iy, iz, ab, ttrack))
    return jobs


def _symlink_rays(dataset_folder: str, work_dir: str) -> None:
    """Symlink rays file into work_dir so FourFrameTracking can run there."""
    src = os.path.abspath(os.path.join(dataset_folder, RAYS_FILENAME))
    dst = os.path.join(work_dir, RAYS_FILENAME)
    os.makedirs(work_dir, exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def merge_manifests(output_dir: str, pattern: str = "grid_manifest_shard_*.csv") -> str | None:
    """Concatenate shard manifests into grid_manifest_merged.csv (one header)."""
    paths = sorted(glob.glob(os.path.join(output_dir, pattern)))
    if not paths:
        return None
    out_path = os.path.join(output_dir, "grid_manifest_merged.csv")
    first = True
    with open(out_path, "w", newline="", encoding="utf-8") as outf:
        writer = None
        for p in paths:
            with open(p, newline="", encoding="utf-8") as inf:
                r = csv.DictReader(inf)
                if first:
                    writer = csv.DictWriter(outf, fieldnames=r.fieldnames)
                    writer.writeheader()
                    first = False
                for row in r:
                    writer.writerow(row)
    return out_path


def _asym_kwargs_for_run(
    x_lo: float | None,
    x_hi: float | None,
    y_lo: float | None,
    y_hi: float | None,
    z_lo: float | None,
    z_hi: float | None,
) -> dict:
    m = [
        (x_lo, "box_size_initial_x_lo"),
        (x_hi, "box_size_initial_x_hi"),
        (y_lo, "box_size_initial_y_lo"),
        (y_hi, "box_size_initial_y_hi"),
        (z_lo, "box_size_initial_z_lo"),
        (z_hi, "box_size_initial_z_hi"),
    ]
    return {kw: v for v, kw in m if v is not None}


def run_grid(
    dataset_folder: str,
    output_dir: str,
    jobs: list[tuple[float, float, float, tuple, float]],
    workers: int,
    write_paraview: bool,
    write_failed_tracks: bool,
    use_bspline: bool,
    export_bspline_paraview: bool,
    shard_id: int | None,
    num_shards: int | None,
    use_isolated_workdir: bool,
) -> list[dict]:
    try:
        from python_4be_eti.perform_tracking import FourFrameTracking
        from python_4be_eti.utils.basic_utils import create_h5_file
    except ImportError as e:
        raise RuntimeError(
            "4BE-ETI is required. Install with: pip install -e 4BE-ETI"
        ) from e

    rays_path = os.path.join(dataset_folder, RAYS_FILENAME)
    if not os.path.isfile(rays_path):
        raise FileNotFoundError(
            f"Rays file not found: {rays_path}. Run stereo matching first."
        )

    os.makedirs(output_dir, exist_ok=True)
    manifest_rows: list[dict] = []
    total_jobs = len(jobs)
    if shard_id is not None and num_shards is not None:
        my_jobs = [(idx, j) for idx, j in enumerate(
            jobs) if idx % num_shards == shard_id]
    else:
        my_jobs = list(enumerate(jobs))

    n_local = 0
    total_local = len(my_jobs)

    for global_idx, job in my_jobs:
        ix, iy, iz, ab, ttrack = job
        n_local += 1
        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = ab
        asym_kwargs = _asym_kwargs_for_run(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        asym_s = ""
        if asym_kwargs:
            asym_s = f" asym={asym_kwargs}"
        label = f"[{n_local}/{total_local}]"
        if shard_id is not None:
            label = f"[shard {shard_id}/{num_shards} {label}]"
        print(
            f"{label} job#{global_idx} initial=({ix}, {iy}, {iz}) "
            f"track_box={ttrack}{asym_s}",
            flush=True,
        )

        work_dir = None
        if use_isolated_workdir:
            work_dir = os.path.join(
                output_dir, "_parallel_work", f"job_{global_idx:06d}"
            )
            if os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
            _symlink_rays(dataset_folder, work_dir)
            track_folder = work_dir
        else:
            track_folder = dataset_folder

        try:
            create_h5_file(folder=track_folder)
            tracking = FourFrameTracking(
                track_folder,
                filename=RAYS_FILENAME,
                box_size_x=ix,
                box_size_y=iy,
                box_size_z=iz,
                box_size_track=ttrack,
                dt=DT,
                rep_rate=REP_RATE,
                write_paraview=write_paraview,
                write_failed_tracks=write_failed_tracks,
                use_bspline=use_bspline,
                export_bspline_paraview=export_bspline_paraview,
                **asym_kwargs,
            )
            tracking.run_tracking(workers=workers)
            tracks_src = os.path.join(track_folder, TRACKS_FILENAME)
            if not os.path.isfile(tracks_src):
                print(f"  Warning: {TRACKS_FILENAME} missing, skip.")
                continue
            dest_name = _tracks_filename(
                ix, iy, iz, ttrack, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
            )
            dest_path = os.path.join(output_dir, dest_name)
            shutil.copy2(tracks_src, dest_path)
            print(f"  Saved -> {dest_path}")
            manifest_rows.append(
                {
                    "job_index": global_idx,
                    "initial_x": ix,
                    "initial_y": iy,
                    "initial_z": iz,
                    "initial_x_lo": x_lo if x_lo is not None else "",
                    "initial_x_hi": x_hi if x_hi is not None else "",
                    "initial_y_lo": y_lo if y_lo is not None else "",
                    "initial_y_hi": y_hi if y_hi is not None else "",
                    "initial_z_lo": z_lo if z_lo is not None else "",
                    "initial_z_hi": z_hi if z_hi is not None else "",
                    "track_box": ttrack,
                    "filename": dest_name,
                }
            )
        finally:
            if use_isolated_workdir and work_dir and os.path.isdir(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)

    print(
        f"Shard summary: {len(manifest_rows)} file(s) written "
        f"(of {total_local} job(s) in this shard; total grid {total_jobs}).",
        flush=True,
    )
    return manifest_rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Grid: initial box (x,y,z) × track box_size for 4-frame tracking."
    )
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--initial-x",
        type=str,
        default=None,
        help=f"Comma-separated initial box half-widths in x (symmetric if no lo/hi). "
        f"Default: {','.join(map(str, DEFAULT_INITIAL_1D))}",
    )
    p.add_argument(
        "--initial-y",
        type=str,
        default=None,
        help="Comma-separated y half-widths (default: same as --initial-x).",
    )
    p.add_argument(
        "--initial-z",
        type=str,
        default=None,
        help="Comma-separated z half-widths (default: same as --initial-x).",
    )
    p.add_argument(
        "--track-box-sizes",
        type=str,
        default=None,
        help="Comma-separated box_size_track values.",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--write-paraview", action="store_true")
    p.add_argument("--write-failed-tracks", action="store_true", default=True)
    p.add_argument("--no-write-failed-tracks",
                   action="store_false", dest="write_failed_tracks")
    p.add_argument("--use-bspline", action="store_true")
    p.add_argument("--export-bspline-paraview", action="store_true")
    p.add_argument("--log-file", type=str, default=None)

    # Optional asymmetric initial bounds (comma-separated lists; product across lo/hi for x,y,z)
    for axis in ("x", "y", "z"):
        for tag in ("lo", "hi"):
            p.add_argument(
                f"--initial-{axis}-{tag}",
                type=str,
                default=None,
                dest=f"initial_{axis}_{tag}",
                help=f"{axis} initial bound ({tag}): comma-separated floats; "
                f"omit flag for symmetric default from --initial-{axis}. "
                f"If a value starts with '-', use --initial-{axis}-{tag}=-0.5,1 (equals form).",
            )

    p.add_argument(
        "--shard",
        type=str,
        default=None,
        help="Parallel split: i/N runs only jobs with job_index %% N == i (0-based). "
        "Uses isolated work dirs. Example: --shard=5/24",
    )
    p.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print job count and each job line, then exit (no tracking).",
    )
    p.add_argument(
        "--merge-manifests",
        action="store_true",
        help="Merge grid_manifest_shard_*.csv in --output-dir into grid_manifest_merged.csv and exit.",
    )

    return p.parse_args()


MANIFEST_FIELDS = [
    "job_index",
    "initial_x",
    "initial_y",
    "initial_z",
    "initial_x_lo",
    "initial_x_hi",
    "initial_y_lo",
    "initial_y_hi",
    "initial_z_lo",
    "initial_z_hi",
    "track_box",
    "filename",
]


def _build_job_list(args: argparse.Namespace):
    xs = _parse_float_list(args.initial_x, DEFAULT_INITIAL_1D)
    if args.initial_y:
        ys = _parse_float_list(args.initial_y, DEFAULT_INITIAL_1D)
    else:
        ys = list(xs)
    if args.initial_z:
        zs = _parse_float_list(args.initial_z, DEFAULT_INITIAL_1D)
    else:
        zs = list(xs)

    track_sizes = _parse_float_list(
        args.track_box_sizes, DEFAULT_TRACK_BOX_SIZES)

    initial_xyz = list(itertools.product(xs, ys, zs))
    x_los = _parse_asym_list(args.initial_x_lo)
    x_his = _parse_asym_list(args.initial_x_hi)
    y_los = _parse_asym_list(args.initial_y_lo)
    y_his = _parse_asym_list(args.initial_y_hi)
    z_los = _parse_asym_list(args.initial_z_lo)
    z_his = _parse_asym_list(args.initial_z_hi)
    asym_bounds = list(
        itertools.product(x_los, x_his, y_los, y_his, z_los, z_his)
    )
    return enumerate_jobs(initial_xyz, asym_bounds, track_sizes)


def main():
    args = parse_args()

    if args.merge_manifests:
        merged = merge_manifests(args.output_dir)
        if merged:
            print(f"Merged manifest: {merged}")
        else:
            print("No grid_manifest_shard_*.csv found.", file=sys.stderr)
            return 1
        return 0

    shard_id, num_shards = parse_shard(args.shard)
    use_isolated = shard_id is not None and num_shards is not None
    if use_isolated and args.workers != 1:
        print(
            "Note: with --shard, prefer --workers 1 per process to avoid oversubscribing CPUs.",
            flush=True,
        )

    jobs = _build_job_list(args)
    if args.list_jobs:
        print(f"Total jobs: {len(jobs)}")
        for idx, (ix, iy, iz, ab, ttrack) in enumerate(jobs):
            x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = ab
            fn = _tracks_filename(ix, iy, iz, ttrack, x_lo,
                                  x_hi, y_lo, y_hi, z_lo, z_hi)
            print(f"  {idx:6d}  ({ix},{iy},{iz})  track={ttrack}  -> {fn}")
        if use_isolated:
            per = (len(jobs) + num_shards - 1) // num_shards
            print(
                f"With --shard i/{num_shards}, each shard runs ~{per} job(s).")
        return 0

    tee = None
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        tee = Tee(args.log_file)
        sys.stdout = tee
        sys.stderr = tee
    try:
        jobs = _build_job_list(args)
        initial_xyz = list(
            {(j[0], j[1], j[2]) for j in jobs}
        )  # unique triples for stats only
        n_asym = len(
            {j[3] for j in jobs}
        )
        track_sizes = sorted({j[4] for j in jobs})

        print(f"Dataset: {args.dataset}")
        print(f"Output:  {args.output_dir}")
        print(f"Unique initial (x,y,z) triples: {len(initial_xyz)}")
        print(f"Unique asym bounds tuples: {n_asym}")
        print(f"Track box sizes: {track_sizes}")
        print(f"Total jobs: {len(jobs)}")
        if use_isolated:
            print(
                f"Parallel: shard {shard_id}/{num_shards} (isolated work dirs under output/_parallel_work/)")

        rows = run_grid(
            args.dataset,
            args.output_dir,
            jobs,
            workers=args.workers,
            write_paraview=args.write_paraview,
            write_failed_tracks=args.write_failed_tracks,
            use_bspline=args.use_bspline,
            export_bspline_paraview=args.export_bspline_paraview,
            shard_id=shard_id,
            num_shards=num_shards,
            use_isolated_workdir=use_isolated,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        if use_isolated and num_shards is not None:
            manifest_path = os.path.join(
                args.output_dir,
                f"grid_manifest_shard_{shard_id:02d}_of_{num_shards:02d}.csv",
            )
        else:
            manifest_path = os.path.join(args.output_dir, "grid_manifest.csv")
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"Manifest: {manifest_path}")
        print(f"Done. Saved {len(rows)} track files.")
        if use_isolated:
            print(
                "After all shards finish: "
                f"python Post-Processing/sensitivity_initial_and_track_box_size.py "
                f"--merge-manifests --output-dir {args.output_dir!r}",
            )
        return 0
    finally:
        if tee is not None:
            tee.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
