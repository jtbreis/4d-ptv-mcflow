#!/usr/bin/env python3
"""
Launcher for ``mart_test.py``: sets repo-root working directory, default paths,
and always requests ParaView output (``.vti``).

Default reconstruction is **accumulate** (zero volume + summed ray intensity);
use ``--method mart`` for the iterative MART path.

Edit ``DEFAULT_*`` below for your dataset, or pass CLI overrides.

Usage:
  python MART/launch_mart.py
  python MART/launch_mart.py --nz 128 --images /a.tif,/b.tif,/c.tif,/d.tif
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Defaults — raw_data/mart_test: Calibration/calib.h5 + images/Camera{1..4}_*.tif
# Image order must match calib.h5: Camera 0 → Camera1 file, …, Camera 3 → Camera4 file.
# ---------------------------------------------------------------------------
DEFAULT_FOLDER = _REPO / "raw_data" / "mart_test"
_DEFAULT_IMG_DIR = DEFAULT_FOLDER / "images"
DEFAULT_IMAGES = ",".join(
    str(_DEFAULT_IMG_DIR / f"Camera{i}_0014.tif") for i in (1, 2, 3, 4)
)
DEFAULT_NZ = 128
DEFAULT_OUT_DIR = DEFAULT_FOLDER / "output"
# mart_test.py writes legacy BINARY .vtk next to .npy by default; .vti is optional.
DEFAULT_NPY = DEFAULT_OUT_DIR / "mart_volume.npy"
DEFAULT_VTI = ""


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    mart = repo / "MART" / "mart_test.py"

    p = argparse.ArgumentParser(
        description="Run mart_test: .npy + robust .vtk by default; optional .vti"
    )
    p.add_argument(
        "--folder",
        type=str,
        default=str(DEFAULT_FOLDER),
        help="Experiment folder for import_calibration (calib.h5)",
    )
    p.add_argument(
        "--images",
        type=str,
        default=DEFAULT_IMAGES,
        help="Comma-separated raw image paths (required if DEFAULT_IMAGES empty)",
    )
    p.add_argument("--nz", type=int, default=DEFAULT_NZ,
                   help="Voxel count in Z")
    p.add_argument(
        "--zmin",
        type=float,
        default=None,
        help="Optional volume Z min (mm); forward with --zmax to mart_test.py",
    )
    p.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="Optional volume Z max (mm); forward with --zmin to mart_test.py",
    )
    p.add_argument(
        "--vti",
        type=str,
        default=DEFAULT_VTI,
        help="Optional path for VTK XML ImageData (.vti); empty = skip",
    )
    p.add_argument(
        "--npy",
        type=str,
        default=str(DEFAULT_NPY),
        help="Also save NumPy volume here (empty string to skip)",
    )
    p.add_argument("--pixel-stride", type=int, default=1)
    p.add_argument(
        "--black-pixel-max",
        type=float,
        default=2.0,
        metavar="V",
        help="Forwarded to mart_test.py: values <= V after bg sub are black (-1 off)",
    )
    p.add_argument(
        "--crop",
        type=str,
        default=None,
        metavar="W[,H]",
        help="Optional center crop passed to mart_test.py, e.g. 512 or 640,512",
    )
    p.add_argument(
        "--method",
        choices=("accumulate", "mart", "pinv"),
        default="accumulate",
        help="accumulate, mart, or pinv; forwarded to mart_test.py",
    )
    p.add_argument(
        "--accumulate-beam-sigma",
        type=float,
        default=None,
        metavar="MM",
        help="When set and method accumulate: forwarded to mart_test.py (Gaussian σ mm)",
    )
    p.add_argument(
        "--accumulate-beam-chunk-samples",
        type=int,
        default=0,
        metavar="N",
        help="Optional: forwarded to mart_test.py when nonzero (beam vectorization chunk)",
    )
    p.add_argument(
        "--pinv-rcond",
        type=float,
        default=None,
        help="When --method pinv: forwarded as --pinv-rcond (SVD truncation / lsmr damp)",
    )
    p.add_argument(
        "--pinv-max-elements",
        type=int,
        default=None,
        help="When --method pinv: forwarded as --pinv-max-elements",
    )
    p.add_argument(
        "--pinv-lsmr-damp",
        type=float,
        default=None,
        help="When --method pinv: forwarded as --pinv-lsmr-damp",
    )
    p.add_argument("--mart-iters", type=int, default=10)
    p.add_argument("--relax", type=float, default=0.15)
    p.add_argument("--center-rotate", type=str, default=None)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Forward to mart_test.py (disable tqdm / stderr bars)",
    )
    p.add_argument(
        "--no-vtk",
        action="store_true",
        help="Forward to mart_test.py (skip auto .vtk beside --out)",
    )
    p.add_argument(
        "--vtk",
        type=str,
        default="",
        help="Optional explicit .vtk path forwarded to mart_test.py",
    )
    args = p.parse_args()

    if not args.images.strip():
        p.error(
            "No images: set DEFAULT_IMAGES in launch_mart.py or pass --images a.tif,b.tif,..."
        )

    cmd = [
        sys.executable,
        str(mart),
        "--folder",
        args.folder,
        "--images",
        args.images,
        "--nz",
        str(args.nz),
        "--pixel-stride",
        str(args.pixel_stride),
        "--black-pixel-max",
        str(args.black_pixel_max),
        "--method",
        args.method,
        "--mart-iters",
        str(args.mart_iters),
        "--relax",
        str(args.relax),
    ]
    if args.pinv_rcond is not None:
        cmd += ["--pinv-rcond", str(args.pinv_rcond)]
    if args.pinv_max_elements is not None:
        cmd += ["--pinv-max-elements", str(args.pinv_max_elements)]
    if args.pinv_lsmr_damp is not None:
        cmd += ["--pinv-lsmr-damp", str(args.pinv_lsmr_damp)]
    if args.center_rotate:
        cmd += ["--center-rotate", args.center_rotate]
    if args.image_width is not None:
        cmd += ["--image-width", str(args.image_width)]
    if args.image_height is not None:
        cmd += ["--image-height", str(args.image_height)]
    if args.npy.strip():
        cmd += ["--out", args.npy]
    if args.vti.strip():
        cmd += ["--vti", args.vti]
    if args.no_progress:
        cmd.append("--no-progress")
    if args.no_vtk:
        cmd.append("--no-vtk")
    if args.vtk.strip():
        cmd += ["--vtk", args.vtk]
    if args.crop:
        cmd += ["--crop", args.crop]
    if args.accumulate_beam_sigma is not None:
        cmd += ["--accumulate-beam-sigma", str(args.accumulate_beam_sigma)]
    if int(args.accumulate_beam_chunk_samples) > 0:
        cmd += [
            "--accumulate-beam-chunk-samples",
            str(args.accumulate_beam_chunk_samples),
        ]
    if args.zmin is not None:
        cmd += ["--zmin", str(args.zmin)]
    if args.zmax is not None:
        cmd += ["--zmax", str(args.zmax)]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(repo))


if __name__ == "__main__":
    main()
