"""
Plot PDF of settling velocity v_y for different box_size tracking results.

Reads tracks_box_size_*.h5part from the sensitivity analysis output directory
and plots the probability density of v_y (y-component of velocity) for each
box_size. Use after running sensitivity_box_size_tracking.py.

Run from project root:
  python Post-Processing/plot_settling_velocity_pdf_by_box_size.py
  python Post-Processing/plot_settling_velocity_pdf_by_box_size.py --input-dir path/to/sensitivity/output
"""
import argparse
import glob
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Default: same output dir as sensitivity_box_size_tracking.py
DEFAULT_INPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "sensitivity_box_size", "tracking_test_threshold1"
)
TRACKS_PATTERN = "tracks_box_size_*.h5part"
BOX_SIZE_RE = re.compile(r"tracks_box_size_([\d.]+)\.h5part")


def load_vy_per_file(input_dir: str):
    """
    Load all v_y values from tracks_box_size_*.h5part in input_dir.

    Yields (box_size, vy_array) where vy_array is 1D float array (finite values only).
    """
    pattern = os.path.join(input_dir, TRACKS_PATTERN)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {TRACKS_PATTERN} in {os.path.abspath(input_dir)}"
        )

    for path in files:
        basename = os.path.basename(path)
        m = BOX_SIZE_RE.match(basename)
        if not m:
            continue
        box_size = float(m.group(1))
        vy_list = []
        with h5py.File(path, "r") as f:
            step_keys = sorted(
                [k for k in f.keys() if k.startswith("Step#")],
                key=lambda k: int(k[5:]) if k.startswith("Step#") else -1,
            )
            for key in step_keys:
                if "vy" not in f[key]:
                    continue
                vy = np.asarray(f[key]["vy"]).ravel()
                vy_list.append(vy)
        if vy_list:
            all_vy = np.concatenate(vy_list)
            finite = np.isfinite(all_vy)
            yield box_size, all_vy[finite]
        else:
            yield box_size, np.array([], dtype=float)


def plot_pdf(
    input_dir: str,
    output_path: str | None = None,
    bins: int | str = "auto",
    density: bool = True,
    hist_alpha: float = 0.5,
    use_kde: bool = False,
    xlabel: str = r"Settling velocity $v_y$",
    ylabel: str = "PDF",
):
    """
    Plot PDF of v_y for each box_size (overlapping histograms or KDE).
    """
    data = list(load_vy_per_file(input_dir))
    if not data:
        raise ValueError(f"No track data found in {input_dir}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Sort by box_size for consistent colors
    data.sort(key=lambda x: x[0])

    for box_size, vy in data:
        label = f"box_size = {box_size:.2f} (n={len(vy)})"
        if use_kde and len(vy) > 10:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vy)
                x_min, x_max = vy.min(), vy.max()
                pad = (x_max - x_min) * 0.1 or 1.0
                xx = np.linspace(x_min - pad, x_max + pad, 200)
                ax.plot(xx, kde(xx), label=label, lw=2)
            except Exception:
                ax.hist(
                    vy,
                    bins=bins,
                    density=density,
                    alpha=hist_alpha,
                    label=label,
                    histtype="step",
                    linewidth=1.5,
                )
        else:
            ax.hist(
                vy,
                bins=bins,
                density=density,
                alpha=hist_alpha,
                label=label,
                histtype="step",
                linewidth=1.5,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Settling velocity $v_y$ PDF by tracking box_size")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot PDF of v_y from sensitivity_box_size tracking results."
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing tracks_box_size_*.h5part (default: {DEFAULT_INPUT_DIR}).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output figure path (default: show interactively).",
    )
    p.add_argument(
        "--bins",
        type=str,
        default="auto",
        help="Histogram bins: integer or 'auto' (default: auto).",
    )
    p.add_argument(
        "--kde",
        action="store_true",
        help="Use KDE instead of histogram (requires scipy).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.bins != "auto":
        try:
            args.bins = int(args.bins)
        except ValueError:
            pass
    output = args.output
    if not output and not os.environ.get("DISPLAY"):
        output = os.path.join(
            args.input_dir, "settling_velocity_vy_pdf_by_box_size.pdf"
        )
        print(f"No DISPLAY; saving to {output}")
    plot_pdf(
        args.input_dir,
        output_path=output,
        bins=args.bins,
        use_kde=args.kde,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
