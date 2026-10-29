"""
Plot tracking sensitivity sweeps (STM error or box_size) from tracks_*.h5part files.

Produces four PDFs in a sensitivity subfolder:
  1. Parameter vs distribution of assigned tracks per 4-frame snapshot (mean marked)
  2. Velocity PDFs and mean/std/skew/kurtosis vs parameter
  3. Acceleration PDFs and mean/std/skew/kurtosis vs parameter
  4. Per-run 2D heatmaps of velocity vs acceleration (x, y, z) with low-density scatter
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass

import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

SNAPSHOT_LENGTH = 4

VELOCITY_FIELDS = ("vx", "vy", "vz")
ACCELERATION_FIELDS = ("ax", "ay", "az")
VELOCITY_LABELS = (r"$v_x$", r"$v_y$", r"$v_z$")
ACCELERATION_LABELS = (r"$a_x$", r"$a_y$", r"$a_z$")
COMPONENT_PAIRS = (
    ("vx", "ax", r"$v_x$", r"$a_x$"),
    ("vy", "ay", r"$v_y$", r"$a_y$"),
    ("vz", "az", r"$v_z$", r"$a_z$"),
)


@dataclass(frozen=True)
class SweepConfig:
    name: str
    tracks_pattern: str
    tracks_re: re.Pattern
    param_key: str
    param_label: str
    param_short: str
    param_format: str
    output_subdir: str
    tracks_pdf_name: str
    velocity_pdf_name: str
    acceleration_pdf_name: str
    velocity_acceleration_pdf_name: str
    kinematics_vs_title: str
    log_acceleration_pdf_y: bool = True


STM_ERROR_SWEEP = SweepConfig(
    name="stm_error",
    tracks_pattern="tracks_error_*.h5part",
    tracks_re=re.compile(r"tracks_error_([\d.]+)\.h5part"),
    param_key="value",
    param_label="STM error threshold",
    param_short="error",
    param_format=".4f",
    output_subdir="STM_error_sensitivity",
    tracks_pdf_name="stm_error_vs_tracks_per_4frame_snapshot.pdf",
    velocity_pdf_name="stm_error_velocity_statistics.pdf",
    acceleration_pdf_name="stm_error_acceleration_statistics.pdf",
    velocity_acceleration_pdf_name="stm_error_velocity_acceleration_heatmaps.pdf",
    kinematics_vs_title="STM error threshold",
)

BOX_SIZE_SWEEP = SweepConfig(
    name="box_size",
    tracks_pattern="tracks_box_size_*.h5part",
    tracks_re=re.compile(r"tracks_box_size_([\d.]+)\.h5part"),
    param_key="value",
    param_label="box_size_track (mm)",
    param_short="box",
    param_format=".2f",
    output_subdir="box_size_sensitivity",
    tracks_pdf_name="box_size_vs_tracks_per_4frame_snapshot.pdf",
    velocity_pdf_name="box_size_velocity_statistics.pdf",
    acceleration_pdf_name="box_size_acceleration_statistics.pdf",
    velocity_acceleration_pdf_name="box_size_velocity_acceleration_heatmaps.pdf",
    kinematics_vs_title="box_size_track",
)

SWEEP_CONFIGS = {
    "stm_error": STM_ERROR_SWEEP,
    "box_size": BOX_SIZE_SWEEP,
}


def _step_sort_key(key: str) -> int:
    if key.startswith("Step#"):
        return int(key[5:])
    return -1


def discover_runs(input_dir: str, config: SweepConfig) -> list[dict]:
    """Discover sensitivity track files and their sweep parameter values."""
    input_dir = os.path.abspath(input_dir)
    pattern = os.path.join(input_dir, config.tracks_pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {config.tracks_pattern} in {input_dir}"
        )

    runs: list[dict] = []
    for path in files:
        basename = os.path.basename(path)
        m = config.tracks_re.match(basename)
        if not m:
            continue
        runs.append(
            {
                config.param_key: float(m.group(1)),
                "path": path,
            }
        )

    runs.sort(key=lambda r: r[config.param_key])
    return runs


def load_per_snapshot_counts(
    filepath: str, *, snapshot_length: int = SNAPSHOT_LENGTH
) -> np.ndarray:
    """Return assigned track count for each non-overlapping 4-frame snapshot."""
    with h5py.File(filepath, "r") as f:
        step_keys = sorted(
            [k for k in f.keys() if k.startswith("Step#")],
            key=_step_sort_key,
        )
        if not step_keys:
            return np.array([], dtype=np.float64)

        counts: list[int] = []
        for key in step_keys:
            frame = _step_sort_key(key)
            if frame % snapshot_length != 0:
                continue
            counts.append(len(f[key]["id"]))
        return np.asarray(counts, dtype=np.float64)


def load_run_snapshot_stats(runs: list[dict]) -> list[dict]:
    stats_out: list[dict] = []
    for run in runs:
        counts = load_per_snapshot_counts(run["path"])
        if counts.size == 0:
            stats_out.append(
                {
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "counts": counts,
                }
            )
            continue
        stats_out.append(
            {
                "mean": float(counts.mean()),
                "std": float(counts.std()),
                "median": float(np.median(counts)),
                "min": float(counts.min()),
                "max": float(counts.max()),
                "counts": counts,
            }
        )
    return stats_out


def load_kinematic_field(
    filepath: str,
    field: str,
    *,
    max_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    chunks: list[np.ndarray] = []

    with h5py.File(filepath, "r") as f:
        step_keys = sorted(
            [k for k in f.keys() if k.startswith("Step#")],
            key=_step_sort_key,
        )
        for key in step_keys:
            if field not in f[key]:
                continue
            values = np.asarray(f[key][field], dtype=np.float64).ravel()
            finite = values[np.isfinite(values)]
            if finite.size:
                chunks.append(finite)

    if not chunks:
        return np.array([], dtype=np.float64)

    arr = np.concatenate(chunks)
    if max_samples is not None and arr.size > max_samples:
        arr = rng.choice(arr, max_samples, replace=False)
    return arr


def load_paired_kinematic(
    filepath: str,
    vel_field: str,
    acc_field: str,
    *,
    max_samples: int = 200_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load paired velocity/acceleration samples from the same particles per frame."""
    rng = rng or np.random.default_rng(0)
    vel_chunks: list[np.ndarray] = []
    acc_chunks: list[np.ndarray] = []

    with h5py.File(filepath, "r") as f:
        step_keys = sorted(
            [k for k in f.keys() if k.startswith("Step#")],
            key=_step_sort_key,
        )
        for key in step_keys:
            grp = f[key]
            if vel_field not in grp or acc_field not in grp:
                continue
            vel = np.asarray(grp[vel_field], dtype=np.float64).ravel()
            acc = np.asarray(grp[acc_field], dtype=np.float64).ravel()
            if vel.size != acc.size:
                continue
            mask = np.isfinite(vel) & np.isfinite(acc)
            if not np.any(mask):
                continue
            vel_chunks.append(vel[mask])
            acc_chunks.append(acc[mask])

    if not vel_chunks:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    vel_arr = np.concatenate(vel_chunks)
    acc_arr = np.concatenate(acc_chunks)
    if vel_arr.size > max_samples:
        idx = rng.choice(vel_arr.size, max_samples, replace=False)
        vel_arr = vel_arr[idx]
        acc_arr = acc_arr[idx]
    return vel_arr, acc_arr


def load_run_paired_kinematics(
    runs: list[dict],
    config: SweepConfig,
    *,
    max_samples: int = 200_000,
) -> tuple[list[float], dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    """Load (velocity, acceleration) pairs for each component and sweep value."""
    xs: list[float] = []
    pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        vel_field: [] for vel_field, _, _, _ in COMPONENT_PAIRS
    }
    rng = np.random.default_rng(0)

    for run in runs:
        xs.append(run[config.param_key])
        for vel_field, acc_field, _, _ in COMPONENT_PAIRS:
            vel_arr, acc_arr = load_paired_kinematic(
                run["path"],
                vel_field,
                acc_field,
                max_samples=max_samples,
                rng=rng,
            )
            pairs[vel_field].append((vel_arr, acc_arr))

    return xs, pairs


def _bin_edges_1d(values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.linspace(-1.0, 1.0, n_bins + 1)
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return np.linspace(lo, hi, n_bins + 1)


def _low_density_point_mask(
    vel: np.ndarray,
    acc: np.ndarray,
    hist: np.ndarray,
    vel_edges: np.ndarray,
    acc_edges: np.ndarray,
    *,
    density_fraction: float,
) -> np.ndarray:
    """Mask True for samples in bins below density_fraction × max(bin density)."""
    if vel.size == 0 or hist.size == 0:
        return np.zeros(vel.size, dtype=bool)

    vi = np.searchsorted(vel_edges, vel, side="right") - 1
    ai = np.searchsorted(acc_edges, acc, side="right") - 1
    vi = np.clip(vi, 0, hist.shape[0] - 1)
    ai = np.clip(ai, 0, hist.shape[1] - 1)
    point_density = hist[vi, ai]
    max_density = float(hist.max())
    if max_density <= 0:
        return np.ones(vel.size, dtype=bool)
    return point_density < (max_density * density_fraction)


def _plot_single_va_heatmap(
    ax: plt.Axes,
    vel: np.ndarray,
    acc: np.ndarray,
    *,
    vel_label: str,
    acc_label: str,
    n_bins: int,
    density_fraction: float,
    max_scatter_points: int,
    rng: np.random.Generator,
    cmap: str = "magma",
) -> None:
    if vel.size == 0:
        ax.set_title(f"{vel_label} vs {acc_label} (no data)")
        ax.set_xlabel(f"Velocity {vel_label} (m/s)")
        ax.set_ylabel(f"Acceleration {acc_label} (m/s²)")
        return

    vel_edges = _bin_edges_1d(vel, n_bins)
    acc_edges = _bin_edges_1d(acc, n_bins)
    hist, _, _ = np.histogram2d(
        vel,
        acc,
        bins=[vel_edges, acc_edges],
        density=True,
    )

    mesh = ax.pcolormesh(
        vel_edges,
        acc_edges,
        hist.T,
        shading="auto",
        cmap=cmap,
        norm=plt.cm.colors.LogNorm(vmin=max(float(hist[hist > 0].min()), 1e-12), vmax=float(hist.max()))
        if np.any(hist > 0)
        else None,
    )
    if mesh.norm is None:
        mesh.set_clim(0, 1)

    low_mask = _low_density_point_mask(
        vel,
        acc,
        hist,
        vel_edges,
        acc_edges,
        density_fraction=density_fraction,
    )
    scatter_vel = vel[low_mask]
    scatter_acc = acc[low_mask]
    if scatter_vel.size > max_scatter_points:
        idx = rng.choice(scatter_vel.size, max_scatter_points, replace=False)
        scatter_vel = scatter_vel[idx]
        scatter_acc = scatter_acc[idx]
    if scatter_vel.size > 0:
        ax.scatter(
            scatter_vel,
            scatter_acc,
            s=4,
            c="white",
            alpha=0.35,
            linewidths=0,
            rasterized=True,
        )

    ax.set_xlabel(f"Velocity {vel_label} (m/s)")
    ax.set_ylabel(f"Acceleration {acc_label} (m/s²)")
    ax.set_title(f"{vel_label} vs {acc_label}")
    ax.set_aspect("auto")
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="PDF")


def plot_velocity_acceleration_heatmaps(
    *,
    xs: list[float],
    paired_data: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    config: SweepConfig,
    output_path: str,
    n_bins: int = 70,
    density_fraction: float = 0.12,
    max_scatter_points: int = 4000,
) -> None:
    """One PDF page per sweep value: 2D heatmaps + low-density scatter for x, y, z."""
    rng = np.random.default_rng(0)

    with PdfPages(output_path) as pdf:
        for run_idx, param_val in enumerate(xs):
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
            for ax, (vel_field, _, vel_label, acc_label) in zip(axes, COMPONENT_PAIRS):
                vel_arr, acc_arr = paired_data[vel_field][run_idx]
                _plot_single_va_heatmap(
                    ax,
                    vel_arr,
                    acc_arr,
                    vel_label=vel_label,
                    acc_label=acc_label,
                    n_bins=n_bins,
                    density_fraction=density_fraction,
                    max_scatter_points=max_scatter_points,
                    rng=rng,
                )

            fig.suptitle(
                f"Velocity–acceleration heatmaps ({config.param_short} = "
                f"{param_val:{config.param_format}})",
                fontsize=13,
                y=1.02,
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {output_path} ({len(xs)} page(s))")


def distribution_metrics(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "skew": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values, fisher=False)),
    }


def load_run_kinematics(
    runs: list[dict],
    fields: tuple[str, ...],
    config: SweepConfig,
    *,
    stats_max_samples: int | None = None,
    pdf_max_samples: int = 200_000,
) -> tuple[list[float], dict[str, list[np.ndarray]], dict[str, list[dict[str, float]]]]:
    values: list[float] = []
    pdf_data: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    metrics: dict[str, list[dict[str, float]]] = {f: [] for f in fields}
    rng = np.random.default_rng(0)

    for run in runs:
        values.append(run[config.param_key])
        for field in fields:
            stats_arr = load_kinematic_field(
                run["path"],
                field,
                max_samples=stats_max_samples,
                rng=rng,
            )
            pdf_arr = stats_arr
            if stats_arr.size > pdf_max_samples:
                pdf_arr = rng.choice(stats_arr, pdf_max_samples, replace=False)
            pdf_data[field].append(pdf_arr)
            metrics[field].append(distribution_metrics(stats_arr))

    return values, pdf_data, metrics


def plot_tracks_vs_parameter(
    runs: list[dict],
    config: SweepConfig,
    output_path: str,
) -> None:
    """Show the per-snapshot track-count distribution at each sweep value, with mean marked."""
    xs = [r[config.param_key] for r in runs]
    snapshot_stats = load_run_snapshot_stats(runs)

    positions: list[float] = []
    dist_data: list[np.ndarray] = []
    means: list[float] = []
    mean_xs: list[float] = []
    for x, s in zip(xs, snapshot_stats):
        counts = np.asarray(s["counts"], dtype=np.float64)
        mean = s["mean"]
        if counts.size == 0 or not np.isfinite(mean):
            continue
        positions.append(float(x))
        dist_data.append(counts)
        means.append(float(mean))
        mean_xs.append(float(x))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if not dist_data:
        ax.text(0.5, 0.5, "No snapshot track counts found", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")
        return

    # Violin width from local spacing so adjacent sweep values do not overlap too much.
    if len(positions) > 1:
        gaps = np.diff(sorted(positions))
        widths = float(np.median(gaps[gaps > 0])) * 0.55 if np.any(gaps > 0) else 0.01
    else:
        widths = abs(positions[0]) * 0.05 if positions[0] else 0.01
    widths = max(widths, 1e-4)

    parts = ax.violinplot(
        dist_data,
        positions=positions,
        widths=widths,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("#2F4B7C")
        body.set_alpha(0.35)

    rng = np.random.default_rng(0)
    max_scatter = 400
    for x, counts in zip(positions, dist_data):
        sample = counts
        if sample.size > max_scatter:
            sample = rng.choice(sample, max_scatter, replace=False)
        jitter = rng.uniform(-0.35 * widths, 0.35 * widths, size=sample.size)
        ax.scatter(
            x + jitter,
            sample,
            s=10,
            c="#4C78A8",
            alpha=0.25,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

    ax.plot(
        mean_xs,
        means,
        "o-",
        color="#E45756",
        lw=2,
        ms=7,
        zorder=4,
        label="Mean",
    )

    ax.set_xlabel(config.param_label)
    ax.set_ylabel("Assigned tracks per 4-frame snapshot")
    ax.set_title(
        f"Distribution of assigned tracks per 4-frame snapshot vs {config.param_label}"
    )
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    if len(positions) > 1:
        span = max(positions) - min(positions)
        pad = span * 0.06 if span > 0 else widths
        ax.set_xlim(min(positions) - pad, max(positions) + pad)
    else:
        ax.set_xlim(positions[0] - widths, positions[0] + widths)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _shared_bin_edges(arrays: list[np.ndarray], n_bins: int = 80) -> np.ndarray | int:
    nonempty = [a for a in arrays if a.size > 0]
    if not nonempty:
        return n_bins
    lo = min(float(a.min()) for a in nonempty)
    hi = max(float(a.max()) for a in nonempty)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return np.linspace(lo, hi, n_bins + 1)


def _plot_pdf_panels(
    axes_pdf: np.ndarray,
    xs: list[float],
    pdf_data: dict[str, list[np.ndarray]],
    fields: tuple[str, ...],
    labels: tuple[str, ...],
    config: SweepConfig,
    *,
    n_bins: int = 80,
    log_y: bool = False,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(xs)))

    for ax, field, label in zip(axes_pdf, fields, labels):
        all_arrays = pdf_data[field]
        bins = _shared_bin_edges(all_arrays, n_bins=n_bins)
        for x, arr, color in zip(xs, all_arrays, colors):
            if arr.size == 0:
                continue
            ax.hist(
                arr,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.5,
                alpha=0.9,
                color=color,
                label=(
                    f"{config.param_short} = {x:{config.param_format}} "
                    f"(n={arr.size:,})"
                ),
            )
        ax.set_xlabel(label)
        ax.set_ylabel("PDF")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")


def _plot_metric_panels(
    axes_metric: np.ndarray,
    xs: list[float],
    metrics: dict[str, list[dict[str, float]]],
    fields: tuple[str, ...],
    labels: tuple[str, ...],
    config: SweepConfig,
    metric_keys: tuple[str, ...],
    metric_titles: tuple[str, ...],
) -> None:
    line_styles = ["-", "--", ":"]
    for ax, metric_key, title in zip(axes_metric, metric_keys, metric_titles):
        for field, label, ls in zip(fields, labels, line_styles):
            ys = [m[metric_key] for m in metrics[field]]
            ax.plot(xs, ys, f"o{ls}", lw=2, ms=6, label=label)
        ax.set_xlabel(config.param_label)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")


def plot_kinematics_report(
    *,
    xs: list[float],
    pdf_data: dict[str, list[np.ndarray]],
    metrics: dict[str, list[dict[str, float]]],
    fields: tuple[str, ...],
    labels: tuple[str, ...],
    config: SweepConfig,
    title_prefix: str,
    output_path: str,
    n_bins: int = 80,
    log_pdf_y: bool = False,
) -> None:
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.35, wspace=0.25)

    axes_pdf = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    _plot_pdf_panels(
        np.asarray(axes_pdf),
        xs,
        pdf_data,
        fields,
        labels,
        config,
        n_bins=n_bins,
        log_y=log_pdf_y,
    )

    metric_keys = ("mean", "std", "skew", "kurtosis")
    metric_titles = ("Mean", "Std. dev.", "Skewness", "Kurtosis (normal = 3)")
    axes_metric = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    _plot_metric_panels(
        np.asarray(axes_metric),
        xs,
        metrics,
        fields,
        labels,
        config,
        metric_keys,
        metric_titles,
    )

    fig.suptitle(
        f"{title_prefix} vs {config.kinematics_vs_title}",
        fontsize=13,
        y=0.995,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_analysis(
    input_dir: str,
    config: SweepConfig,
    output_dir: str | None = None,
    *,
    stats_max_samples: int | None = None,
    pdf_max_samples: int = 200_000,
    n_bins: int = 80,
) -> dict[str, str]:
    input_dir = os.path.abspath(input_dir)
    out_dir = output_dir or os.path.join(input_dir, config.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    runs = discover_runs(input_dir, config)
    if not runs:
        raise ValueError(
            f"No {config.name} sensitivity runs found in {input_dir}"
        )

    paths = {
        "tracks": os.path.join(out_dir, config.tracks_pdf_name),
        "velocity": os.path.join(out_dir, config.velocity_pdf_name),
        "acceleration": os.path.join(out_dir, config.acceleration_pdf_name),
        "velocity_acceleration": os.path.join(
            out_dir, config.velocity_acceleration_pdf_name
        ),
    }

    plot_tracks_vs_parameter(runs, config, paths["tracks"])

    vel_xs, vel_pdf, vel_metrics = load_run_kinematics(
        runs,
        VELOCITY_FIELDS,
        config,
        stats_max_samples=stats_max_samples,
        pdf_max_samples=pdf_max_samples,
    )
    plot_kinematics_report(
        xs=vel_xs,
        pdf_data=vel_pdf,
        metrics=vel_metrics,
        fields=VELOCITY_FIELDS,
        labels=VELOCITY_LABELS,
        config=config,
        title_prefix="Velocity",
        output_path=paths["velocity"],
        n_bins=n_bins,
    )

    acc_xs, acc_pdf, acc_metrics = load_run_kinematics(
        runs,
        ACCELERATION_FIELDS,
        config,
        stats_max_samples=stats_max_samples,
        pdf_max_samples=pdf_max_samples,
    )
    plot_kinematics_report(
        xs=acc_xs,
        pdf_data=acc_pdf,
        metrics=acc_metrics,
        fields=ACCELERATION_FIELDS,
        labels=ACCELERATION_LABELS,
        config=config,
        title_prefix="Acceleration",
        output_path=paths["acceleration"],
        n_bins=n_bins,
        log_pdf_y=config.log_acceleration_pdf_y,
    )

    contour_xs, paired_data = load_run_paired_kinematics(
        runs,
        config,
        max_samples=pdf_max_samples,
    )
    plot_velocity_acceleration_heatmaps(
        xs=contour_xs,
        paired_data=paired_data,
        config=config,
        output_path=paths["velocity_acceleration"],
        n_bins=min(n_bins, 70),
    )

    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot tracking sensitivity sweeps (STM error or box_size)."
    )
    p.add_argument(
        "--sweep",
        choices=sorted(SWEEP_CONFIGS),
        default="stm_error",
        help="Sensitivity sweep type (default: stm_error).",
    )
    p.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing tracks_*.h5part sensitivity outputs.",
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--stats-max-samples", type=int, default=None)
    p.add_argument("--pdf-max-samples", type=int, default=200_000)
    p.add_argument("--bins", type=int, default=80)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = SWEEP_CONFIGS[args.sweep]
    paths = run_analysis(
        args.input_dir,
        config,
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
