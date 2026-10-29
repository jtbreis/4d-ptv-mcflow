"""
Quick sanity check for 4-frame tracking output (tracks.h5part).

Successful tracking should show:
  - Mean particle count per frame on the order of a couple thousand
  - Approximately normal distribution of z-velocity (vz)

Run from project root:
  python Post-Processing/check_tracking.py path/to/tracks.h5part
  python Post-Processing/check_tracking.py path/to/run_dir
  python Post-Processing/check_tracking.py --input-dir data/.../TNTI_no_gravity
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STEP_RE = re.compile(r"Step#(\d+)")

VELOCITY_FIELDS = ("vx", "vy", "vz")
ACCELERATION_FIELDS = ("ax", "ay", "az")
KINEMATIC_FIELDS = VELOCITY_FIELDS + ACCELERATION_FIELDS


def _step_frame(key: str) -> int:
    m = STEP_RE.match(key)
    return int(m.group(1)) if m else -1


def find_h5part_files(path: str) -> list[str]:
    """Resolve a file, run directory, or search directory to h5part paths."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        direct = os.path.join(path, "tracks.h5part")
        if os.path.isfile(direct):
            return [direct]
        pattern = os.path.join(path, "**", "tracks.h5part")
        found = sorted(glob.glob(pattern, recursive=True))
        if found:
            return found
        found = sorted(glob.glob(os.path.join(path, "*.h5part")))
        if found:
            return found
    raise FileNotFoundError(f"No tracks.h5part found under {path}")


def _subsample_frames(step_keys: list[str], max_frames: int | None) -> list[str]:
    if max_frames is None or len(step_keys) <= max_frames:
        return step_keys
    idx = np.linspace(0, len(step_keys) - 1, max_frames, dtype=int)
    return [step_keys[i] for i in idx]


def load_tracking_stats(
    filepath: str,
    *,
    max_frames_for_kinematics: int | None = 200,
    max_kinematic_samples: int = 200_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Read h5part and return particle-count and kinematic statistics."""
    rng = rng or np.random.default_rng(0)
    counts: list[int] = []
    kinematic_chunks: dict[str, list[np.ndarray]] = {f: [] for f in KINEMATIC_FIELDS}

    with h5py.File(filepath, "r") as f:
        attrs = {k: _attr_to_python(v) for k, v in f.attrs.items()}
        step_keys = sorted(
            [k for k in f.keys() if k.startswith("Step#")],
            key=_step_frame,
        )
        if not step_keys:
            raise ValueError(f"No Step# groups in {filepath}")

        for key in step_keys:
            grp = f[key]
            counts.append(int(len(grp["x"])))

        sample_keys = _subsample_frames(step_keys, max_frames_for_kinematics)
        for key in sample_keys:
            grp = f[key]
            for field in KINEMATIC_FIELDS:
                if field not in grp:
                    continue
                values = np.asarray(grp[field], dtype=np.float64).ravel()
                finite = values[np.isfinite(values)]
                if finite.size:
                    kinematic_chunks[field].append(finite)

    kinematics: dict[str, np.ndarray] = {}
    for field in KINEMATIC_FIELDS:
        chunks = kinematic_chunks[field]
        arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
        if arr.size > max_kinematic_samples:
            arr = rng.choice(arr, max_kinematic_samples, replace=False)
        kinematics[field] = arr

    counts_arr = np.asarray(counts, dtype=np.float64)
    return {
        "filepath": filepath,
        "attrs": attrs,
        "n_frames": len(counts),
        "particles_per_frame": counts_arr,
        "mean_particles": float(counts_arr.mean()),
        "min_particles": int(counts_arr.min()),
        "max_particles": int(counts_arr.max()),
        "median_particles": float(np.median(counts_arr)),
        "kinematics": kinematics,
        "n_finite_vz": int(kinematics["vz"].size),
        "vz": kinematics["vz"],
    }


def _attr_to_python(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def distribution_metrics(values: np.ndarray, *, qq_sample_size: int = 5000) -> dict:
    if values.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "skew": np.nan,
            "excess_kurtosis": np.nan,
            "qq_correlation": np.nan,
        }

    sample = values if values.size <= qq_sample_size else np.random.default_rng(0).choice(
        values, qq_sample_size, replace=False
    )
    sample_sorted = np.sort(sample)
    probs = np.linspace(0.001, 0.999, sample_sorted.size)
    theoretical = stats.norm.ppf(probs)
    qq_r = float(np.corrcoef(sample_sorted, theoretical)[0, 1])

    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "skew": float(stats.skew(values)),
        "excess_kurtosis": float(stats.kurtosis(values)),
        "qq_correlation": qq_r,
    }


def kinematic_metrics(kinematics: dict[str, np.ndarray]) -> dict[str, dict]:
    return {field: distribution_metrics(kinematics[field]) for field in KINEMATIC_FIELDS}


def evaluate_tracking(
    stats: dict,
    *,
    min_particles_per_frame: float,
    min_finite_vz: int,
    max_abs_skew: float,
    max_abs_excess_kurtosis: float,
    min_qq_correlation: float,
) -> dict:
    counts_ok = stats["mean_particles"] >= min_particles_per_frame
    all_metrics = kinematic_metrics(stats["kinematics"])
    vz_metrics = all_metrics["vz"]

    vz_count_ok = vz_metrics["n"] >= min_finite_vz
    skew_ok = (
        abs(vz_metrics["skew"]) <= max_abs_skew
        if np.isfinite(vz_metrics["skew"])
        else False
    )
    kurt_ok = (
        abs(vz_metrics["excess_kurtosis"]) <= max_abs_excess_kurtosis
        if np.isfinite(vz_metrics["excess_kurtosis"])
        else False
    )
    qq_ok = (
        vz_metrics["qq_correlation"] >= min_qq_correlation
        if np.isfinite(vz_metrics["qq_correlation"])
        else False
    )
    normality_ok = vz_count_ok and skew_ok and kurt_ok and qq_ok
    overall_ok = counts_ok and normality_ok

    return {
        "counts_ok": counts_ok,
        "vz_count_ok": vz_count_ok,
        "skew_ok": skew_ok,
        "kurt_ok": kurt_ok,
        "qq_ok": qq_ok,
        "normality_ok": normality_ok,
        "overall_ok": overall_ok,
        "kinematic_metrics": all_metrics,
        "vz_metrics": vz_metrics,
    }


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check_tracking_file(
    filepath: str,
    *,
    min_particles_per_frame: float = 1000.0,
    min_finite_vz: int = 10_000,
    max_abs_skew: float = 1.0,
    max_abs_excess_kurtosis: float = 3.0,
    min_qq_correlation: float = 0.97,
    max_frames_for_kinematics: int | None = 200,
) -> dict:
    """
    Run tracking quality checks and return a JSON-serializable report dict.
    """
    stats = load_tracking_stats(
        filepath, max_frames_for_kinematics=max_frames_for_kinematics
    )
    result = evaluate_tracking(
        stats,
        min_particles_per_frame=min_particles_per_frame,
        min_finite_vz=min_finite_vz,
        max_abs_skew=max_abs_skew,
        max_abs_excess_kurtosis=max_abs_excess_kurtosis,
        min_qq_correlation=min_qq_correlation,
    )
    return {
        "filepath": os.path.abspath(filepath),
        "thresholds": {
            "min_particles_per_frame": min_particles_per_frame,
            "min_finite_vz": min_finite_vz,
            "max_abs_skew": max_abs_skew,
            "max_abs_excess_kurtosis": max_abs_excess_kurtosis,
            "min_qq_correlation": min_qq_correlation,
        },
        "n_frames": stats["n_frames"],
        "mean_particles": stats["mean_particles"],
        "median_particles": stats["median_particles"],
        "min_particles": stats["min_particles"],
        "max_particles": stats["max_particles"],
        "n_finite_vz": stats["n_finite_vz"],
        "counts_ok": result["counts_ok"],
        "vz_count_ok": result["vz_count_ok"],
        "skew_ok": result["skew_ok"],
        "kurt_ok": result["kurt_ok"],
        "qq_ok": result["qq_ok"],
        "normality_ok": result["normality_ok"],
        "overall_ok": result["overall_ok"],
        "kinematic_metrics": result["kinematic_metrics"],
        "vz_metrics": result["vz_metrics"],
        "existing_h5_attrs": stats.get("attrs") or {},
    }


def _format_metric_line(field: str, metrics: dict, unit: str) -> str:
    if metrics["n"] == 0:
        return f"    {field}: no finite samples"
    return (
        f"    {field}: mean={metrics['mean']:.4f}, std={metrics['std']:.4f}, "
        f"skew={metrics['skew']:.3f}, excess kurtosis={metrics['excess_kurtosis']:.3f} "
        f"(n={metrics['n']}) [{unit}]"
    )


def print_report(stats: dict, result: dict, *, min_particles_per_frame: float) -> None:
    rel = os.path.relpath(stats["filepath"], PROJECT_ROOT)
    kinematic_metrics_map = result["kinematic_metrics"]
    vz_metrics = result["vz_metrics"]
    print(f"\n{'=' * 72}")
    print(f"File: {rel}")
    print(f"Overall: {_status(result['overall_ok'])}")
    print(f"  Frames: {stats['n_frames']}")
    print(
        f"  Particles/frame: mean={stats['mean_particles']:.0f}, "
        f"median={stats['median_particles']:.0f}, "
        f"min={stats['min_particles']}, max={stats['max_particles']} "
        f"[{_status(result['counts_ok'])}; need mean >= {min_particles_per_frame:.0f}]"
    )

    print("  Velocity:")
    for field in VELOCITY_FIELDS:
        print(_format_metric_line(field, kinematic_metrics_map[field], "m/s"))

    print("  Acceleration:")
    for field in ACCELERATION_FIELDS:
        print(_format_metric_line(field, kinematic_metrics_map[field], "m/s^2"))

    print(
        f"  v_z normality check [{_status(result['normality_ok'])}]: "
        f"n={vz_metrics['n']} [{_status(result['vz_count_ok'])}]"
    )
    if vz_metrics["n"] > 0:
        print(
            f"    skew={vz_metrics['skew']:.3f} [{_status(result['skew_ok'])}], "
            f"excess kurtosis={vz_metrics['excess_kurtosis']:.3f} [{_status(result['kurt_ok'])}], "
            f"Q-Q r={vz_metrics['qq_correlation']:.4f} [{_status(result['qq_ok'])}]"
        )
    else:
        print("    no finite v_z values")


def plot_vz_check(
    track_stats: dict,
    result: dict,
    output_path: str,
    *,
    bins: int = 80,
) -> None:
    vz = track_stats["vz"]
    if vz.size == 0:
        return

    metrics = result["vz_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(vz, bins=bins, density=True, alpha=0.65, color="steelblue", label="v_z")
    x = np.linspace(vz.min(), vz.max(), 200)
    pdf = stats.norm.pdf(x, metrics["mean"], metrics["std"])
    ax.plot(x, pdf, "r-", lw=2, label="Normal fit")
    ax.set_xlabel(r"$v_z$ [m/s]")
    ax.set_ylabel("PDF")
    ax.set_title(
        f"v_z histogram ({_status(result['normality_ok'])})\n"
        f"mean={track_stats['mean_particles']:.0f} particles/frame"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sample_size = min(5000, vz.size)
    sample = np.sort(
        vz if vz.size <= sample_size else np.random.default_rng(0).choice(vz, sample_size, replace=False)
    )
    probs = np.linspace(0.001, 0.999, sample.size)
    ax.scatter(stats.norm.ppf(probs), sample, s=4, alpha=0.5)
    lims = [min(sample.min(), stats.norm.ppf(0.001)), max(sample.max(), stats.norm.ppf(0.999))]
    ax.plot(lims, lims, "r--", lw=1.5)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title(f"Q-Q plot (r={metrics['qq_correlation']:.4f})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Check tracks.h5part for successful 4-frame tracking."
    )
    p.add_argument(
        "paths",
        nargs="*",
        help="tracks.h5part file, run directory, or tree to search.",
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory to search recursively for tracks.h5part.",
    )
    p.add_argument(
        "--min-particles",
        type=float,
        default=1000.0,
        help="Minimum mean particles per frame (default: 1000).",
    )
    p.add_argument(
        "--min-finite-vz",
        type=int,
        default=10_000,
        help="Minimum finite v_z samples for normality check (default: 10000).",
    )
    p.add_argument(
        "--max-abs-skew",
        type=float,
        default=1.0,
        help="Maximum |skewness| for v_z (default: 1.0).",
    )
    p.add_argument(
        "--max-abs-kurtosis",
        type=float,
        default=3.0,
        help="Maximum |excess kurtosis| for v_z (default: 3.0).",
    )
    p.add_argument(
        "--min-qq-r",
        type=float,
        default=0.97,
        help="Minimum Q-Q correlation for v_z vs normal (default: 0.97).",
    )
    p.add_argument(
        "--max-frames",
        "--max-frames-for-vz",
        dest="max_frames",
        type=int,
        default=200,
        help="Subsample Step# frames when loading kinematics (default: 200; 0 = all).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save v_z histogram and Q-Q plot next to each file.",
    )
    p.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory for plot output (default: same directory as h5part).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    search_paths = list(args.paths)
    if args.input_dir:
        search_paths.append(args.input_dir)
    if not search_paths:
        search_paths = [os.getcwd()]

    files: list[str] = []
    for path in search_paths:
        files.extend(find_h5part_files(path))
    files = sorted(set(files))
    if not files:
        print("No tracks.h5part files found.", file=sys.stderr)
        return 1

    max_frames = args.max_frames or None
    all_ok = True
    print(f"Checking {len(files)} file(s)...")

    for filepath in files:
        stats = load_tracking_stats(filepath, max_frames_for_kinematics=max_frames)
        result = evaluate_tracking(
            stats,
            min_particles_per_frame=args.min_particles,
            min_finite_vz=args.min_finite_vz,
            max_abs_skew=args.max_abs_skew,
            max_abs_excess_kurtosis=args.max_abs_kurtosis,
            min_qq_correlation=args.min_qq_r,
        )
        print_report(stats, result, min_particles_per_frame=args.min_particles)
        all_ok &= result["overall_ok"]

        if args.plot and stats["n_finite_vz"] > 0:
            plot_dir = args.plot_dir or os.path.dirname(filepath)
            os.makedirs(plot_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(filepath))[0]
            plot_path = os.path.join(plot_dir, f"{stem}_check_vz.pdf")
            plot_vz_check(stats, result, plot_path)

    print(f"\n{'=' * 72}")
    print(f"Summary: {sum(1 for _ in files)} file(s), overall {_status(all_ok)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
