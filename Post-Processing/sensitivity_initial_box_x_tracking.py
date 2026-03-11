"""
Sensitivity analysis on initial box size in x (one-direction / asymmetric) for 4-frame tracking.

Runs tracking on the tracking_test_threshold1 dataset for three initial x box configurations:
  - (-1, 3): x in [x0-1, x0+3]  (lo=1, hi=3)
  - (0, 3):  x in [x0, x0+3]    (lo=0, hi=3) — one direction only
  - (1, 3.5): x in [x0-1, x0+3.5] (lo=1, hi=3.5)

All runs use box_size_track=0.5 and initial box y/z = 0.5 (symmetric).
Run from project root: python Post-Processing/sensitivity_initial_box_x_tracking.py
"""
import argparse
import os
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Default dataset and output paths (same as sensitivity_box_size_tracking)
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data", "tracking_test_threshold1")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "sensitivity_initial_box_x", "tracking_test_threshold1"
)

# (x_lo, x_hi) for initial search in x: [x0 - x_lo, x0 + x_hi]
# - (-1 to 3) -> lo=1, hi=3
# - (0 to 3)  -> lo=0, hi=3  (one direction only)
# - (1 to 3.5)-> lo=1, hi=3.5
DEFAULT_INITIAL_X_CONFIGS = [(1.0, 3.0), (0.0, 3.0), (1.0, 3.5)]

BOX_SIZE_TRACK = 0.5
BOX_SIZE_INITIAL_Y = 0.5
BOX_SIZE_INITIAL_Z = 0.5
DT = 1e-3
REP_RATE = 10
RAYS_FILENAME = "rays_out_cpp.h5"
TRACKS_FILENAME = "tracks.h5part"


def _tracks_basename(x_lo: float, x_hi: float) -> str:
    """Filename base for saved tracks, e.g. tracks_initial_x_1.00_3.00.h5part."""
    return f"tracks_initial_x_{x_lo:.2f}_{x_hi:.2f}.h5part"


def run_sensitivity(
    dataset_folder: str,
    output_dir: str,
    initial_x_configs: list[tuple[float, float]],
    workers: int = 1,
    write_paraview: bool = False,
    write_failed_tracks: bool = True,
) -> list[str]:
    """
    Run tracking for each (x_lo, x_hi) initial box x config, copying tracks.h5part to output_dir.

    Returns list of output .h5part paths.
    """
    try:
        from python_4be_eti.perform_tracking import FourFrameTracking
        from python_4be_eti.utils.basic_utils import create_h5_file
    except ImportError as e:
        raise RuntimeError(
            "4BE-ETI is required for tracking. Install with: pip install -e 4BE-ETI"
        ) from e

    rays_path = os.path.join(dataset_folder, RAYS_FILENAME)
    if not os.path.isfile(rays_path):
        raise FileNotFoundError(
            f"Rays file not found: {rays_path}. Run stereo matching first."
        )

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for i, (x_lo, x_hi) in enumerate(initial_x_configs):
        print(f"[{i + 1}/{len(initial_x_configs)}] initial x: [{x_lo}, {x_hi}] (x in [x0-{x_lo}, x0+{x_hi}])")
        create_h5_file(folder=dataset_folder)
        tracking = FourFrameTracking(
            dataset_folder,
            filename=RAYS_FILENAME,
            box_size_x=3.5,  # nominal; overridden by lo/hi below
            box_size_y=BOX_SIZE_INITIAL_Y,
            box_size_z=BOX_SIZE_INITIAL_Z,
            box_size_track=BOX_SIZE_TRACK,
            dt=DT,
            rep_rate=REP_RATE,
            write_paraview=write_paraview,
            write_failed_tracks=write_failed_tracks,
            box_size_initial_x_lo=x_lo,
            box_size_initial_x_hi=x_hi,
            # y, z symmetric 0.5 (no _lo/_hi => use box_size_y/z for both)
        )
        tracking.run_tracking(workers=workers)
        tracks_src = os.path.join(dataset_folder, TRACKS_FILENAME)
        if not os.path.isfile(tracks_src):
            print(f"  Warning: {TRACKS_FILENAME} not found after run, skipping.")
            continue
        dest_name = _tracks_basename(x_lo, x_hi)
        dest_path = os.path.join(output_dir, dest_name)
        shutil.copy2(tracks_src, dest_path)
        saved_paths.append(dest_path)
        print(f"  Saved -> {dest_path}")

    return saved_paths


def parse_args():
    p = argparse.ArgumentParser(
        description="Sensitivity analysis on initial box x (lo/hi) for tracking_test_threshold1."
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Folder containing {RAYS_FILENAME} (default: tracking_test_threshold1).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where tracks_initial_x_*.h5part will be written.",
    )
    p.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated 'lo,hi' pairs, e.g. '1,3 0,3 1,3.5' (default: 1,3 0,3 1,3.5).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers for tracking (default: 1).",
    )
    p.add_argument(
        "--write-paraview",
        action="store_true",
        help="Export ParaView files for each run (slower).",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write all progress to this file (use tail -f to monitor when running detached).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    tee = None
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        tee = Tee(args.log_file)
        sys.stdout = tee
        sys.stderr = tee
    try:
        configs = DEFAULT_INITIAL_X_CONFIGS
        if args.configs:
            configs = []
            for part in args.configs.split():
                lo, hi = part.strip().split(",")
                configs.append((float(lo), float(hi)))
        print(f"Dataset: {args.dataset}")
        print(f"Output:  {args.output_dir}")
        print(f"box_size_track={BOX_SIZE_TRACK}, initial y/z={BOX_SIZE_INITIAL_Y}/{BOX_SIZE_INITIAL_Z}")
        print(f"Initial x configs (lo, hi): {configs}")
        paths = run_sensitivity(
            args.dataset,
            args.output_dir,
            configs,
            workers=args.workers,
            write_paraview=args.write_paraview,
        )
        print(f"Done. Saved {len(paths)} track files.")
        return 0
    finally:
        if tee is not None:
            tee.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
