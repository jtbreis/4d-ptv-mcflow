#!/usr/bin/env python3
"""
Visualize Stage 2 spatial indexing on the two-particle synthetic dataset.

One plot: all particle positions (all frames), matched 4-frame tracks highlighted,
plus SpatialGrid query boxes for the Stage 2 search steps along particle A.

Writes:
  graph_tracking/synthetic_data/two_particles/stage2_spatial_grid.png

Run from repo root (after ``pip install -e graph_tracking``):
  python scripts/plot_stage2_spatial_grid.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from graph_tracking.assign import greedy_assign_paths
from graph_tracking.kinematics import predict_third_frame_position_xyz
from graph_tracking.link_block import LinkParams, _initial_box_offsets, link_block
from graph_tracking.load_data import load_block
from graph_tracking.spatial_grid import SpatialGrid
from graph_tracking.synthetic.two_particles import expected_link_params

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAYS = (
    PROJECT_ROOT
    / "graph_tracking/synthetic_data/two_particles/rays_out_cpp.h5"
)
DEFAULT_OUT = DEFAULT_RAYS.parent / "stage2_spatial_grid.png"

FRAME_MARKERS = ("o", "s", "^", "D")
FRAME_COLORS = ("#bdbdbd", "#9e9e9e", "#757575", "#616161")
TRACK_COLORS = ("#1f77b4", "#ff7f0e")  # particle A, B
QUERY_COLORS = ("#d62728", "#9467bd", "#2ca02c")  # f0→f1, f1→f2, f2→f3
CELL_EDGE = "#888888"
CELL_EMPTY = "#fafafa"
CELL_OCCUPIED = "#fff8e1"
CELL_OCCUPIED_EDGE = "#5c4a00"


def _xy_cells_in_view(
    grid: SpatialGrid,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
):
    cs = grid.cell_size
    ix0 = int(np.floor((xlim[0] - grid.xmin) / cs))
    ix1 = int(np.ceil((xlim[1] - grid.xmin) / cs))
    iy0 = int(np.floor((ylim[0] - grid.ymin) / cs))
    iy1 = int(np.ceil((ylim[1] - grid.ymin) / cs))
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            yield ix, iy


def _draw_cell_mesh(
    ax: plt.Axes,
    grid: SpatialGrid,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    occupied_keys: set[tuple[int, int]] | None = None,
) -> None:
    """Draw 0.5 mm grid cells; highlight cells that contain particles."""
    occupied_keys = occupied_keys or set()
    for ix, iy in _xy_cells_in_view(grid, xlim, ylim):
        xlo, xhi, ylo, yhi, _, _ = grid.cell_bounds(ix, iy, 0)
        occupied = (ix, iy) in occupied_keys
        ax.add_patch(
            Rectangle(
                (xlo, ylo),
                xhi - xlo,
                yhi - ylo,
                facecolor=CELL_OCCUPIED if occupied else CELL_EMPTY,
                edgecolor=CELL_OCCUPIED_EDGE if occupied else CELL_EDGE,
                linewidth=1.2 if occupied else 0.7,
                zorder=0 if not occupied else 0.5,
            )
        )


def _occupied_xy_keys(grid: SpatialGrid) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for cell_key, _indices in grid._cells.items():
        keys.add((cell_key[0], cell_key[1]))
    return keys


def _particle_cell_key(grid: SpatialGrid, i: int) -> tuple[int, int, int]:
    return grid._cell_key(grid.x[i], grid.y[i], grid.z[i])


def _shade_query_cells(
    ax: plt.Axes,
    grid: SpatialGrid,
    query_bounds: tuple[float, float, float, float, float, float],
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float = 0.22,
) -> set[tuple[int, int]]:
    """Shade and outline cells overlapping a query; return visited (ix, iy) keys."""
    xlo, xhi, ylo, yhi, zlo, zhi = query_bounds
    visited: set[tuple[int, int]] = set()
    for ix, iy, _iz in grid.iter_cells_overlapping_aabb(xlo, xhi, ylo, yhi, zlo, zhi):
        key = (ix, iy)
        if key in visited:
            continue
        visited.add(key)
        cxlo, cxhi, cylo, cyhi, _, _ = grid.cell_bounds(ix, iy, 0)
        ax.add_patch(
            Rectangle(
                (cxlo, cylo),
                cxhi - cxlo,
                cyhi - cylo,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.6,
                linestyle="--",
                alpha=alpha,
                zorder=1.5,
            )
        )
    return visited


def _draw_query_box(
    ax: plt.Axes,
    query_bounds: tuple[float, float, float, float, float, float],
    *,
    edgecolor: str,
    label: str,
    linestyle: str = "-",
) -> None:
    xlo, xhi, ylo, yhi, _, _ = query_bounds
    ax.add_patch(
        Rectangle(
            (xlo, ylo),
            xhi - xlo,
            yhi - ylo,
            fill=False,
            edgecolor=edgecolor,
            linewidth=1.8,
            linestyle=linestyle,
            zorder=4,
            label=label,
        )
    )


def _track_points(block, path) -> list[tuple[int, float, float]]:
    """(frame_index, x, y) for each detection on a PathCandidate."""
    pts = []
    for j, local_idx in enumerate(path.indices):
        layer = path.start_layer + j
        fr = block.frames[layer]
        pts.append((fr.frame_index, float(fr.x[local_idx]), float(fr.y[local_idx])))
    return pts


def make_figure(rays_h5: Path, out_path: Path, *, seed_particle: int = 0) -> None:
    kw = expected_link_params()
    params = LinkParams(**kw)
    block = load_block(str(rays_h5), 0)
    result = link_block(block, params)
    assigned = greedy_assign_paths(result.candidates)
    assigned_4f = [p for p in assigned if p.length == 4 and p.start_layer == 0]
    assigned_4f.sort(key=lambda p: p.indices[0])

    cell_size = max(params.box_size_track, 0.5)
    offsets = _initial_box_offsets(params)
    b = params.box_size_track

    f0, f1, f2, f3 = block.frames
    grids = [
        SpatialGrid(f1.x, f1.y, f1.z, cell_size=cell_size),
        SpatialGrid(f2.x, f2.y, f2.z, cell_size=cell_size),
        SpatialGrid(f3.x, f3.y, f3.z, cell_size=cell_size),
    ]

    # Stage-2 example: follow seed particle A through f0→f1→f2→f3 queries
    si = seed_particle
    p0 = np.array([f0.x[si], f0.y[si], f0.z[si]])
    p1 = np.array([f1.x[si], f1.y[si], f1.z[si]])
    p2 = np.array([f2.x[si], f2.y[si], f2.z[si]])
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = offsets
    q_initial = (
        p0[0] + x_lo, p0[0] + x_hi,
        p0[1] + y_lo, p0[1] + y_hi,
        p0[2] + z_lo, p0[2] + z_hi,
    )
    pred2 = 2.0 * p1 - p0
    q_track2 = (pred2[0] - b, pred2[0] + b, pred2[1] - b, pred2[1] + b, pred2[2] - b, pred2[2] + b)
    pred3 = predict_third_frame_position_xyz(p0, p1, p2)
    q_track3 = (pred3[0] - b, pred3[0] + b, pred3[1] - b, pred3[1] + b, pred3[2] - b, pred3[2] + b)

    fig, ax = plt.subplots(figsize=(12, 7))

    xlim = (-3.2, 3.2)
    ylim = (-1.2, 1.2)
    ref_grid = grids[0]
    cs = ref_grid.cell_size

    # Base cell mesh (frame-1 grid): empty cells + cells containing f1 particles
    occupied_f1 = _occupied_xy_keys(grids[0])
    _draw_cell_mesh(ax, ref_grid, xlim, ylim, occupied_keys=occupied_f1)

    # Query-visited cells (dashed colored outlines on top of mesh)
    visited1 = _shade_query_cells(
        ax, grids[0], q_initial, facecolor=QUERY_COLORS[0], edgecolor=QUERY_COLORS[0],
    )
    visited2 = _shade_query_cells(
        ax, grids[1], q_track2, facecolor=QUERY_COLORS[1], edgecolor=QUERY_COLORS[1],
    )
    visited3 = _shade_query_cells(
        ax, grids[2], q_track3, facecolor=QUERY_COLORS[2], edgecolor=QUERY_COLORS[2],
    )
    _draw_query_box(ax, q_initial, edgecolor=QUERY_COLORS[0], label="query f0→f1 (initial box)")
    _draw_query_box(ax, q_track2, edgecolor=QUERY_COLORS[1], label="query f1→f2 (±box_track)")
    _draw_query_box(ax, q_track3, edgecolor=QUERY_COLORS[2], label="query f2→f3 (±box_track)")

    # Annotate cell concept: hash example + cell size scale
    ex_i = seed_particle
    ex_key = _particle_cell_key(grids[0], ex_i)
    ex_xlo, ex_xhi, ex_ylo, ex_yhi, _, _ = grids[0].cell_bounds(*ex_key)
    ax.add_patch(
        Rectangle(
            (ex_xlo, ex_ylo),
            ex_xhi - ex_xlo,
            ex_yhi - ex_ylo,
            fill=False,
            edgecolor="#000000",
            linewidth=2.8,
            zorder=5,
        )
    )
    ax.annotate(
        f"particle → cell ({ex_key[0]}, {ex_key[1]})\n"
        f"cell_size = {cs:.1f} mm",
        ((ex_xlo + ex_xhi) / 2, ex_yhi),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#333", alpha=0.92),
        zorder=10,
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0),
    )

    # One query-region cell that is visited but not occupied (concept: skip empty cells fast)
    for ix, iy in sorted(visited1):
        if (ix, iy) not in occupied_f1:
            vxlo, vxhi, vylo, vyhi, _, _ = grids[0].cell_bounds(ix, iy, 0)
            ax.annotate(
                "visited cell\n(no particle)",
                ((vxlo + vxhi) / 2, (vylo + vyhi) / 2),
                ha="center",
                va="center",
                fontsize=7,
                color=QUERY_COLORS[0],
                fontstyle="italic",
                zorder=10,
            )
            break

    # Cell-size ruler in corner
    ruler_ix = int(np.floor((xlim[0] - ref_grid.xmin) / cs))
    ruler_iy = int(np.floor((ylim[0] - ref_grid.ymin) / cs))
    rxlo, rxhi, rylo, ryhi, _, _ = ref_grid.cell_bounds(ruler_ix, ruler_iy, 0)
    ax.annotate(
        "",
        xy=(rxlo, rylo),
        xytext=(rxhi, rylo),
        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2),
        zorder=10,
    )
    ax.text(
        (rxlo + rxhi) / 2,
        rylo - 0.12,
        f"{cs:.1f} mm",
        ha="center",
        va="top",
        fontsize=8,
        color="#333",
        zorder=10,
    )

    # All detections (muted), one marker shape per frame
    for layer, fr in enumerate(block.frames):
        for pi in range(fr.n):
            ax.scatter(
                fr.x[pi],
                fr.y[pi],
                s=55,
                marker=FRAME_MARKERS[layer],
                c=FRAME_COLORS[layer],
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
                alpha=0.85,
            )
            ax.annotate(
                f"f{fr.frame_index}",
                (fr.x[pi], fr.y[pi]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
                color="#555555",
                zorder=2,
            )

    # Matched 4-frame tracks (bold lines + frame-shaped markers)
    for ti, path in enumerate(assigned_4f):
        pts = _track_points(block, path)
        color = TRACK_COLORS[ti % len(TRACK_COLORS)]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(xs, ys, "-", color=color, linewidth=2.5, alpha=0.9, zorder=6)
        for fi, x, y in ((p[0], p[1], p[2]) for p in pts):
            layer = block.frame_indices.index(fi)
            ax.scatter(
                [x],
                [y],
                s=200,
                marker=FRAME_MARKERS[layer],
                facecolors=color,
                edgecolors="black",
                linewidths=1.2,
                zorder=7,
            )
        pid = path.indices[0]
        ax.annotate(
            f"track {pid}",
            (xs[0], ys[0]),
            textcoords="offset points",
            xytext=(-12, -14),
            fontsize=9,
            fontweight="bold",
            color=color,
            zorder=8,
        )

    # Seed star for Stage 2 walkthrough
    ax.scatter(
        [p0[0]],
        [p0[1]],
        s=220,
        marker="*",
        c=TRACK_COLORS[seed_particle],
        edgecolors="black",
        linewidths=0.8,
        zorder=9,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        "Stage 2 — spatial grid cells (0.5 mm) + all detections + matched tracks\n"
        f"tan = occupied cell, dashed tint = query visits only those cells "
        f"(box_track = {b} mm)",
        fontsize=11,
    )

    cell_handles = [
        Rectangle((0, 0), 1, 1, facecolor=CELL_EMPTY, edgecolor=CELL_EDGE, label="empty cell"),
        Rectangle(
            (0, 0), 1, 1,
            facecolor=CELL_OCCUPIED,
            edgecolor=CELL_OCCUPIED_EDGE,
            label="cell with particle",
        ),
        Line2D(
            [0], [0],
            color=QUERY_COLORS[0],
            linestyle="--",
            linewidth=1.6,
            label="visited by query",
        ),
    ]
    frame_handles = [
        Line2D(
            [0], [0],
            marker=FRAME_MARKERS[i],
            color="w",
            markerfacecolor=FRAME_COLORS[i],
            markeredgecolor="white",
            markersize=8,
            label=f"frame {block.frame_indices[i]} (all particles)",
        )
        for i in range(4)
    ]
    track_handles = [
        Line2D([0], [0], color=TRACK_COLORS[i], linewidth=2.5, label=f"matched track {i}")
        for i in range(len(assigned_4f))
    ]
    query_handles, query_labels = ax.get_legend_handles_labels()
    seed_handle = Line2D(
        [0], [0],
        marker="*",
        color="w",
        markerfacecolor=TRACK_COLORS[seed_particle],
        markeredgecolor="black",
        markersize=12,
        label=f"seed particle {seed_particle}",
    )
    ax.legend(
        handles=cell_handles + frame_handles + track_handles + query_handles + [seed_handle],
        labels=[h.get_label() for h in cell_handles + frame_handles + track_handles]
        + query_labels
        + [seed_handle.get_label()],
        loc="upper right",
        fontsize=7,
        framealpha=0.95,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Stage 2 spatial grid on synthetic data")
    p.add_argument("--input", type=Path, default=DEFAULT_RAYS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed-particle", type=int, default=0, help="Particle index for query demo")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing synthetic rays file: {args.input}")
    make_figure(args.input, args.output, seed_particle=args.seed_particle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
