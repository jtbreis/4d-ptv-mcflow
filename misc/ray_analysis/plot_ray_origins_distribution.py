#!/usr/bin/env python3
"""
Plot the spatial distribution of ray origins (xyz0) from rays.h5.

rays.h5 layout matches python_4dptv Rays output: groups "Camera N" / "frame#####" / dataset "xyz0" (N×3).

CLI (needs h5py, numpy, matplotlib):
  Set ``RUN_OR_RAYS`` near the top of this file, then run with no arguments; or pass a path
  on the command line (overrides ``RUN_OR_RAYS``):
  python misc/ray_analysis/plot_ray_origins_distribution.py
  python misc/ray_analysis/plot_ray_origins_distribution.py /path/to/rays.h5 --frames all

Jupyter: open ``plot_ray_origins_distribution.ipynb`` in this folder, or import
``resolve_rays_path``, ``compute_frame_indices``, ``load_ray_origins``, ``make_ray_origins_figure``.

By default only ``FRAME_INDEX`` (below) is loaded; use ``--frames all`` or set ``FRAME_INDEX = None`` for every frame.
"""
# %%
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import TYPE_CHECKING

import h5py
import numpy as np

# -----------------------------------------------------------------------------
# Default input (run folder containing rays.h5, or path to rays.h5).
# Used when you run this script with no path argument. CLI path overrides this.
# Jupyter: import RUN_OR_RAYS from this module, or override in a notebook cell.
# -----------------------------------------------------------------------------
RUN_OR_RAYS: str = "/workspaces/4d-ptv-mcflow/raw_data/low_threshold/PTV_above/TTI_aligned_with_gravity/Run2/rays.h5"

# Which frame(s) when --frames is not passed (CLI) or when FRAMES_SPEC is None (notebook).
# int: single 0-based frame. None: load all frames.
FRAME_INDEX: int | None = 0

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# %% — path and frame helpers


def resolve_rays_path(path: str) -> str:
    """Run directory (containing rays.h5) or direct path to rays.h5."""
    p = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(p):
        candidate = os.path.join(p, "rays.h5")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"No rays.h5 in directory: {p}")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"Not a file or directory with rays.h5: {path}")


def _camera_sort_key(name: str) -> tuple[int, str]:
    m = re.match(r"Camera\s+(\d+)$", name.strip())
    if m:
        return (int(m.group(1)), name)
    return (10**9, name)


def _frame_sort_key(name: str) -> int:
    if name.startswith("frame") and name[5:].isdigit():
        return int(name[5:])
    return 0


def _parse_frame_spec(spec: str | None, n_frames: int) -> list[int]:
    if spec is None or spec.strip().lower() in ("all", ""):
        return list(range(n_frames))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a.strip()), int(b.strip())
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    out = sorted(set(i for i in out if 0 <= i < n_frames))
    if not out:
        raise ValueError(
            f"No valid frame indices in {spec!r} (n_frames={n_frames})"
        )
    return out


def _list_cameras(f: h5py.File) -> list[str]:
    return sorted(
        (
            k
            for k in f.keys()
            if isinstance(f[k], h5py.Group) and k.startswith("Camera")
        ),
        key=_camera_sort_key,
    )


def _n_frames_for_camera(f: h5py.File, cam: str) -> int:
    g = f[cam]
    keys = [k for k in g.keys() if k.startswith("frame")]
    if not keys:
        return 0
    return max(_frame_sort_key(k) for k in keys) + 1


def _selected_cameras(f: h5py.File, cameras: list[int] | None) -> list[str]:
    cam_names = _list_cameras(f)
    if not cam_names:
        raise ValueError("No Camera groups in file")
    if cameras is None:
        return cam_names
    want = set(cameras)
    selected = [c for c in cam_names if _camera_sort_key(c)[0] in want]
    if not selected:
        raise ValueError(f"No cameras matching {cameras}; have {cam_names}")
    return selected


def effective_frames_spec(frames_spec: str | None) -> str | None:
    """
    Resolve notebook/CLI ``frames_spec`` against ``FRAME_INDEX``.

    * ``frames_spec`` not None → use it (``\"all\"`` / ranges / lists as in CLI).
    * ``frames_spec`` is None → ``FRAME_INDEX`` is int → that frame only; ``None`` → all frames.
    """
    if frames_spec is not None and str(frames_spec).strip():
        return frames_spec
    if FRAME_INDEX is None:
        return None
    return str(int(FRAME_INDEX))


def compute_frame_indices(
    rays_h5: str,
    *,
    cameras: list[int] | None = None,
    frames_spec: str | None = None,
) -> list[int]:
    """Resolve ``frames_spec`` (same strings as CLI ``--frames``) to 0-based indices."""
    with h5py.File(rays_h5, "r") as f:
        selected = _selected_cameras(f, cameras)
        n_frames = min(_n_frames_for_camera(f, c) for c in selected)
    return _parse_frame_spec(frames_spec, n_frames)


# %% — load origins


def load_ray_origins(
    rays_h5: str,
    *,
    cameras: list[int] | None,
    frame_indices: list[int],
    frame_stride: int = 1,
    max_points: int | None = 800_000,
    seed: int = 0,
) -> np.ndarray:
    """
    Returns (M, 3) float64 array of ray origins; may subsample to max_points.
    """
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []

    with h5py.File(rays_h5, "r") as f:
        selected_cams = _selected_cameras(f, cameras)

        n_frames = min(_n_frames_for_camera(f, c) for c in selected_cams)
        if n_frames == 0:
            raise ValueError("No frame groups under camera(s)")

        use_frames = frame_indices[:: max(1, frame_stride)]
        use_frames = [i for i in use_frames if i < n_frames]
        if not use_frames:
            raise ValueError("No frames to load after stride/indices")

        frame_keys = sorted(
            (k for k in f[selected_cams[0]].keys() if k.startswith("frame")),
            key=_frame_sort_key,
        )

        for cam in selected_cams:
            cg = f[cam]
            for fi in use_frames:
                if fi >= len(frame_keys):
                    continue
                fk = frame_keys[fi]
                if fk not in cg:
                    continue
                ds = cg[fk]["xyz0"]
                xyz = np.asarray(ds[()], dtype=np.float64)
                if xyz.ndim != 2 or xyz.shape[1] != 3:
                    raise ValueError(
                        f"{cam}/{fk}/xyz0 expected (N,3), got {xyz.shape}"
                    )
                if xyz.size == 0:
                    continue
                chunks.append(xyz)

    if not chunks:
        raise ValueError("No ray origins loaded (empty xyz0 everywhere?)")

    pts = np.vstack(chunks)
    if max_points is not None and pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    return pts


def load_first_frame_ray_directions(
    rays_h5: str,
    *,
    cameras: list[int] | None,
    max_rays_per_camera: int = 5,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Returns per-camera arrays from the first frame:
      {"Camera N": (K, 6)} where columns are [x0, y0, z0, dx, dy, dz].
    """
    rng = np.random.default_rng(seed)
    per_camera: dict[str, np.ndarray] = {}

    with h5py.File(rays_h5, "r") as f:
        selected_cams = _selected_cameras(f, cameras)
        for cam in selected_cams:
            cg = f[cam]
            frame_keys = sorted(
                (k for k in cg.keys() if k.startswith("frame")),
                key=_frame_sort_key,
            )
            if not frame_keys:
                continue

            first = frame_keys[0]
            if "xyz0" not in cg[first] or "dd" not in cg[first]:
                continue

            xyz0 = np.asarray(cg[first]["xyz0"][()], dtype=np.float64)
            dd = np.asarray(cg[first]["dd"][()], dtype=np.float64)
            if xyz0.ndim != 2 or dd.ndim != 2 or xyz0.shape[1] != 3 or dd.shape[1] != 3:
                raise ValueError(
                    f"{cam}/{first} expected xyz0 and dd as (N,3), got {xyz0.shape} and {dd.shape}"
                )
            if xyz0.shape[0] != dd.shape[0]:
                raise ValueError(
                    f"{cam}/{first} xyz0 and dd row mismatch: {xyz0.shape[0]} vs {dd.shape[0]}"
                )
            if xyz0.shape[0] == 0:
                continue

            k = min(max_rays_per_camera, xyz0.shape[0])
            idx = rng.choice(xyz0.shape[0], size=k, replace=False)
            per_camera[cam] = np.hstack((xyz0[idx], dd[idx]))

    if not per_camera:
        raise ValueError(
            "No first-frame rays with xyz0+dd loaded from selected camera(s)")
    return per_camera


# %% — plot (notebook-friendly: returns Figure, does not close)


def make_ray_origins_figure(
    xyz: np.ndarray,
    *,
    bins_2d: int = 80,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 7),
    cmap: str = "viridis",
) -> Figure:
    """
    Build the ray-origin distribution figure (hexbin marginals + 1D histograms).
    Safe for Jupyter: returns the figure; caller displays with ``display(fig)`` or relies on inline backend.
    """
    import matplotlib.pyplot as plt

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    if title:
        fig.suptitle(title)

    pairs = [("X", "Y", x, y), ("X", "Z", x, z), ("Y", "Z", y, z)]
    for ax, (a_lab, b_lab, a, b) in zip(axes[0], pairs):
        ax.hexbin(a, b, gridsize=bins_2d, mincnt=1, cmap=cmap)
        ax.set_xlabel(a_lab)
        ax.set_ylabel(b_lab)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{a_lab}–{b_lab} density")

    for ax, lab, arr in zip(axes[1], ("X", "Y", "Z"), (x, y, z)):
        ax.hist(
            arr,
            bins=min(120, max(20, int(np.sqrt(len(arr))))),
            color="steelblue",
            alpha=0.85,
        )
        ax.set_xlabel(lab)
        ax.set_ylabel("count")
        ax.set_title(f"{lab} marginal")

    fig.tight_layout()
    return fig


def make_first_frame_ray_directions_figure(
    rays_by_camera: dict[str, np.ndarray],
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    ray_t_limits: tuple[float, float] = (-25.0, 25.0),
) -> Figure:
    """Build a 3D line plot of first-frame ray directions for t in [min,max]."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10", max(1, len(rays_by_camera)))
    t0, t1 = ray_t_limits

    for i, cam in enumerate(sorted(rays_by_camera, key=_camera_sort_key)):
        arr = rays_by_camera[cam]
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        u, v, w = arr[:, 3], arr[:, 4], arr[:, 5]
        d = np.column_stack((u, v, w))
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        keep = (norms[:, 0] > 0)
        if not np.any(keep):
            continue
        d_unit = d[keep] / norms[keep]
        p0 = np.column_stack((x, y, z))[keep]
        p_start = p0 + t0 * d_unit
        p_end = p0 + t1 * d_unit

        color = cmap(i)
        for j in range(p_start.shape[0]):
            label = f"{cam} ({p_start.shape[0]})" if j == 0 else None
            ax.plot(
                [p_start[j, 0], p_end[j, 0]],
                [p_start[j, 1], p_end[j, 1]],
                [p_start[j, 2], p_end[j, 2]],
                color=color,
                linewidth=0.8,
                alpha=0.8,
                label=label,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    # Top-down view onto the x-y plane.
    # Standard 3D perspective (non top-down).
    ax.view_init(elev=30, azim=-60)
    ax.set_title(
        title or f"First-frame ray directions (sampled per camera, t in [{t0}, {t1}])"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def load_first_frame_ray_matches(
    rays_h5: str,
    rays_out_cpp_h5: str,
    *,
    cameras: list[int] | None,
    max_matches: int = 150,
    seed: int = 0,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
    """
    Load sampled first-frame stereo matches and corresponding camera ray origins.

    Returns:
      - match_xyz: (K, 3) triangulated points from rays_out_cpp frame00000/xyze.
      - cam_segments: list of (camera name, origins (K,3), valid_mask (K,))
        where valid_mask indicates matches with valid ray IDs for that camera.
    """
    rng = np.random.default_rng(seed)

    with h5py.File(rays_h5, "r") as fr, h5py.File(rays_out_cpp_h5, "r") as fm:
        selected_cams = _selected_cameras(fr, cameras)
        if "frame00000" not in fm:
            raise ValueError(f"No frame00000 in {rays_out_cpp_h5}")

        gm = fm["frame00000"]
        if "xyze" not in gm or "camrayids" not in gm:
            raise ValueError(
                "Expected frame00000/xyze and frame00000/camrayids in rays_out_cpp.h5"
            )

        xyze = np.asarray(gm["xyze"][()], dtype=np.float64)
        camrayids = np.asarray(gm["camrayids"][()], dtype=np.int64)
        if xyze.ndim != 2 or xyze.shape[0] < 3:
            raise ValueError(f"xyze expected (>=3, N), got {xyze.shape}")
        if camrayids.ndim != 2 or camrayids.shape[1] != xyze.shape[1]:
            raise ValueError(
                f"camrayids expected (M, N) with same N as xyze, got {camrayids.shape} and {xyze.shape}"
            )

        n = xyze.shape[1]
        if n == 0:
            raise ValueError("No matches found in frame00000")
        k = min(max_matches, n)
        sel = rng.choice(n, size=k, replace=False)
        match_xyz = xyze[:3, sel].T
        camrayids_sel = camrayids[:, sel]
        n_pairs = camrayids_sel.shape[0] // 2

        cam_segments: list[tuple[str, np.ndarray, np.ndarray]] = []
        for cam in selected_cams:
            cam_idx = _camera_sort_key(cam)[0]
            origin = np.zeros((k, 3), dtype=np.float64)
            valid = np.zeros(k, dtype=bool)

            frame_key = "frame00000"
            if frame_key not in fr[cam]:
                cam_segments.append((cam, origin, valid))
                continue
            rays_xyz0 = np.asarray(fr[cam][frame_key]["xyz0"][()], dtype=np.float64)

            for r in range(n_pairs):
                cam_ids = camrayids_sel[2 * r]
                ray_ids = camrayids_sel[2 * r + 1]
                mask = cam_ids == cam_idx
                if not np.any(mask):
                    continue
                ray_ok = (ray_ids >= 0) & (ray_ids < rays_xyz0.shape[0])
                use = mask & ray_ok
                if not np.any(use):
                    continue
                ridx = ray_ids[use].astype(np.int64)
                origin[use] = rays_xyz0[ridx]
                valid[use] = True

            cam_segments.append((cam, origin, valid))

    return match_xyz, cam_segments


def make_first_frame_ray_matches_figure(
    match_xyz: np.ndarray,
    cam_segments: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Build a top-view x-y plot of sampled matched rays (origin -> match point)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10", max(1, len(cam_segments)))

    for i, (cam, origins, valid) in enumerate(cam_segments):
        p0 = origins[valid]
        p1 = match_xyz[valid]
        if p0.shape[0] == 0:
            continue
        color = cmap(i)
        for j in range(p0.shape[0]):
            label = f"{cam} ({p0.shape[0]} rays)" if j == 0 else None
            ax.plot(
                [p0[j, 0], p1[j, 0]],
                [p0[j, 1], p1[j, 1]],
                [p0[j, 2], p1[j, 2]],
                color=color,
                linewidth=0.6,
                alpha=0.6,
                label=label,
            )

    ax.scatter(
        match_xyz[:, 0],
        match_xyz[:, 1],
        match_xyz[:, 2],
        s=8,
        c="k",
        alpha=0.4,
        label=f"Matched points ({match_xyz.shape[0]})",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=90, azim=-90)
    ax.set_title(title or "First-frame matched rays from rays_out_cpp.h5")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _strip_jupyter_kernel_args(argv: list[str]) -> list[str]:
    """
    ipykernel passes ``-f /path/to/kernel-....json``; argparse may treat ``-f`` as an
    abbreviation of ``--frames`` / ``--frame-stride``. Drop that pair (and ``--f=...``).
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-f" and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.endswith(".json") or "/jupyter/runtime/" in nxt.replace("\\", "/"):
                i += 2
                continue
        if a.startswith("--f=") and a.endswith(".json"):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = _strip_jupyter_kernel_args(sys.argv[1:])
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Run directory containing rays.h5, or path to rays.h5. "
        "If omitted, uses RUN_OR_RAYS set near the top of this file.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="PNG output path (default: ray_origins_distribution.png next to rays.h5 unless --show)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display the figure instead of writing a PNG (non-headless)",
    )
    p.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated camera indices (e.g. 0,1,2). Default: all.",
    )
    p.add_argument(
        "--frames",
        type=str,
        default=None,
        help='Frame indices: "all", "0,1,5", or range "0-10". '
        "Default: single FRAME_INDEX from this file (see top).",
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth frame from the selected set (>=1).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=800_000,
        help="Random subsample cap for plotting (default 800000; 0 = no cap).",
    )
    p.add_argument(
        "--bins-2d",
        type=int,
        default=80,
        help="Hexbin grid resolution per axis.",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for subsampling.")
    p.add_argument("--title", type=str, default=None, help="Figure title.")
    args = p.parse_args(argv)

    chosen = (args.path or "").strip() or (RUN_OR_RAYS or "").strip()
    if not chosen:
        p.error(
            "Pass a path on the command line or set RUN_OR_RAYS near the top of this script."
        )

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("matplotlib is required.", file=sys.stderr)
        raise SystemExit(1) from e

    rays_h5 = resolve_rays_path(chosen)

    cameras = None
    if args.cameras:
        cameras = [int(x.strip())
                   for x in args.cameras.split(",") if x.strip()]

    frame_indices = compute_frame_indices(
        rays_h5,
        cameras=cameras,
        frames_spec=effective_frames_spec(args.frames),
    )

    max_pts = args.max_points if args.max_points > 0 else None

    xyz = load_ray_origins(
        rays_h5,
        cameras=cameras,
        frame_indices=frame_indices,
        frame_stride=max(1, args.frame_stride),
        max_points=max_pts,
        seed=args.seed,
    )
    n = xyz.shape[0]
    t = args.title or f"Ray origins ({n} points)\n{rays_h5}"
    fig = make_ray_origins_figure(
        xyz, bins_2d=max(10, args.bins_2d), title=t
    )
    rays_by_camera = load_first_frame_ray_directions(
        rays_h5,
        cameras=cameras,
        max_rays_per_camera=50,
        seed=args.seed,
    )
    directions_fig = make_first_frame_ray_directions_figure(
        rays_by_camera,
        title="First-frame ray directions (50 rays/camera)",
    )
    rays_out_cpp_h5 = os.path.join(os.path.dirname(rays_h5), "rays_out_cpp.h5")
    matches_fig = None
    if os.path.isfile(rays_out_cpp_h5):
        match_xyz, cam_segments = load_first_frame_ray_matches(
            rays_h5,
            rays_out_cpp_h5,
            cameras=cameras,
            max_matches=150,
            seed=args.seed,
        )
        matches_fig = make_first_frame_ray_matches_figure(
            match_xyz,
            cam_segments,
            title="First-frame matched rays from rays_out_cpp.h5",
        )
    else:
        print(f"Skipping match plot; file not found: {rays_out_cpp_h5}")

    if args.show:
        plt.show()
    else:
        out_path = args.output or os.path.join(
            os.path.dirname(rays_h5), "ray_origins_distribution.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Wrote {out_path}")
        directions_out_path = os.path.join(
            os.path.dirname(out_path), "ray_directions_first_frame.png"
        )
        directions_fig.savefig(directions_out_path,
                               dpi=150, bbox_inches="tight")
        print(f"Wrote {directions_out_path}")
        if matches_fig is not None:
            matches_out_path = os.path.join(
                os.path.dirname(out_path), "ray_matches_first_frame.png"
            )
            matches_fig.savefig(matches_out_path, dpi=150, bbox_inches="tight")
            print(f"Wrote {matches_out_path}")
        plt.show()
    plt.close(fig)
    plt.close(directions_fig)
    if matches_fig is not None:
        plt.close(matches_fig)


if __name__ == "__main__":
    main()
