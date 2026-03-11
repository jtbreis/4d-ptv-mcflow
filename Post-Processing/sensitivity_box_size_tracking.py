"""
Sensitivity analysis on box_size (box_size_track) for 4-frame tracking.

Runs tracking on the tracking_test_threshold1 dataset for a range of box_size
values, saving each tracks.h5part to a separate file so results can be compared.
Run from project root: python Post-Processing/sensitivity_box_size_tracking.py
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

# Default dataset and output paths
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data", "tracking_test_threshold1")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "sensitivity_box_size", "tracking_test_threshold1"
)
DEFAULT_BOX_SIZES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# Tracking parameters (match 05_4frame_tracking.py)
BOX_SIZE_INITIAL_X = 3.5
BOX_SIZE_INITIAL_Y = 1.0
BOX_SIZE_INITIAL_Z = 1.0
DT = 1e-3
REP_RATE = 10
RAYS_FILENAME = "rays_out_cpp.h5"
TRACKS_FILENAME = "tracks.h5part"


def _tracks_basename(box_size: float) -> str:
    """Filename base for saved tracks, e.g. tracks_box_size_1.00.h5part."""
    return f"tracks_box_size_{box_size:.2f}.h5part"


def run_sensitivity(
    dataset_folder: str,
    output_dir: str,
    box_sizes: list[float],
    workers: int = 1,
    write_paraview: bool = False,
    write_failed_tracks: bool = True,
) -> list[str]:
    """
    Run tracking for each box_size, copying tracks.h5part to output_dir.

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

    for i, box_size in enumerate(box_sizes):
        print(f"[{i + 1}/{len(box_sizes)}] box_size_track = {box_size}")
        create_h5_file(folder=dataset_folder)
        tracking = FourFrameTracking(
            dataset_folder,
            filename=RAYS_FILENAME,
            box_size_x=BOX_SIZE_INITIAL_X,
            box_size_y=BOX_SIZE_INITIAL_Y,
            box_size_z=BOX_SIZE_INITIAL_Z,
            box_size_track=box_size,
            dt=DT,
            rep_rate=REP_RATE,
            write_paraview=write_paraview,
            write_failed_tracks=write_failed_tracks,
        )
        tracking.run_tracking(workers=workers)
        tracks_src = os.path.join(dataset_folder, TRACKS_FILENAME)
        if not os.path.isfile(tracks_src):
            print(f"  Warning: {TRACKS_FILENAME} not found after run, skipping.")
            continue
        dest_name = _tracks_basename(box_size)
        dest_path = os.path.join(output_dir, dest_name)
        shutil.copy2(tracks_src, dest_path)
        saved_paths.append(dest_path)
        print(f"  Saved -> {dest_path}")

    return saved_paths


def parse_args():
    p = argparse.ArgumentParser(
        description="Sensitivity analysis on box_size for tracking_test_threshold1."
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
        help="Directory where tracks_box_size_*.h5part will be written.",
    )
    p.add_argument(
        "--box-sizes",
        type=str,
        default=None,
        help="Comma-separated box_size values (default: 0.5,0.75,1.0,1.25,1.5,2.0).",
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
        box_sizes = DEFAULT_BOX_SIZES
        if args.box_sizes:
            box_sizes = [float(x.strip()) for x in args.box_sizes.split(",") if x.strip()]
        print(f"Dataset: {args.dataset}")
        print(f"Output:  {args.output_dir}")
        print(f"box_sizes: {box_sizes}")
        paths = run_sensitivity(
            args.dataset,
            args.output_dir,
            box_sizes,
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
