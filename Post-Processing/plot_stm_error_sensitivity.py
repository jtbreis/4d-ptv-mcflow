"""Plot STM error sensitivity (wrapper around plot_tracking_sensitivity)."""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from plot_tracking_sensitivity import STM_ERROR_SWEEP, run_analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot STM error sensitivity: tracks, velocity, and acceleration statistics."
    )
    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: <input-dir>/{STM_ERROR_SWEEP.output_subdir}).",
    )
    p.add_argument("--stats-max-samples", type=int, default=None)
    p.add_argument("--pdf-max-samples", type=int, default=200_000)
    p.add_argument("--bins", type=int, default=80)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = run_analysis(
        args.input_dir,
        STM_ERROR_SWEEP,
        args.output_dir,
        stats_max_samples=args.stats_max_samples,
        pdf_max_samples=args.pdf_max_samples,
        n_bins=args.bins,
    )
    print("Wrote:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
