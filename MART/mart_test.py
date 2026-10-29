"""
Test harness: load one raw image per camera, map pixels to 3D rays via
calibration (same path as STMCpp / stereo pipeline: PTVcalib calib.h5 +
polynomial px→world per plane + fit3dline), then fill a voxel grid.

**accumulate** (default): start from an **all-zero** volume; for each ray, **add**
(``+=``) the (per-camera normalized) pixel intensity along the ray, split across
voxels in proportion to approximate path length (additive back-projection).
With ``--crop``, rays still use **global** pixel ``(x,y)`` = (column, row) for
calibration, equal to crop origin plus local index plus 0.5.

**mart**: iterative multiplicative algebraic reconstruction (see
``--mart-iters``, ``--relax``).

**pinv**: linear model ``y ≈ A x`` (same ``A`` weights as accumulate). If
``n_rays * n_voxels`` is below ``--pinv-max-elements``, uses
``numpy.linalg.pinv(A, rcond=...)`` (Moore–Penrose, **truncated SVD** with
relative cutoff ``rcond``). Otherwise **sparse** ``lsmr`` with Tikhonov
``damp`` (defaults to ``rcond`` — not identical to truncated SVD).

Z resolution is controlled by ``--nz`` (number of voxels along Z). X/Y voxel
counts are chosen so voxel spacing matches dz (cubic voxels in world mm).

Outputs: with ``--out vol.npy`` the script also writes ``vol.vtk`` (VTK legacy
BINARY structured points, ParaView-safe) unless ``--no-vtk``. Optional
``--vti`` writes XML ImageData. Use ``--vtk path`` to force a specific ``.vtk``.

Example:
  python MART/mart_test.py \\
    --folder /path/to/experiment_with_Calibration/calib.h5_parent \\
    --images cam0.tif,cam1.tif,cam2.tif,cam3.tif \\
    --nz 80 --out MART/output/mart_volume.npy --crop 512

Or use ``python MART/launch_mart.py`` (edit defaults there) for a one-command run.

Requires repo roots on PYTHONPATH (run from repo root is enough):
  python MART/mart_test.py ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MART_DIR = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
_PTVCALIB_SRC = _REPO_ROOT / "PTVcalib" / "src"
for p in (_PTVCALIB_SRC, _SRC, _MART_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ptv_calib.io.import_calibration import import_calibration  # noqa: E402

from python_4dptv.matching.fit_line import fit3dline  # noqa: E402
from python_4dptv.matching.rays import (  # noqa: E402
    normalize_center_rotate_per_camera,
    normalize_image_size_per_camera,
)

from npy_to_paraview import (  # noqa: E402
    write_vti_image_data_appended,
    write_vtk_legacy_structured_points_binary,
)


def _strided_sample_count(dim: int, stride: int) -> int:
    if dim <= 0 or stride < 1:
        return 0
    return (dim - 1) // stride + 1


class _SimpleProgressBar:
    """Minimal tqdm-like context manager when tqdm is not installed."""

    def __init__(self, total: int, desc: str, file=sys.stderr, mininterval: float = 0.2):
        self.total = max(int(total), 1)
        self.desc = desc
        self.file = file
        self.n = 0
        self._last_line = ""
        self._mininterval = mininterval
        self._last_draw = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._last_line:
            self.file.write("\n")
            self.file.flush()
        return False

    def set_description(self, desc: str, refresh: bool = True) -> None:
        self.desc = desc
        if refresh:
            self._redraw(force=True)

    def _redraw(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_draw) < self._mininterval and self.n < self.total:
            return
        self._last_draw = now
        pct = 100.0 * self.n / self.total
        width = 36
        filled = int(width * self.n / self.total)
        bar = "#" * filled + "-" * (width - filled)
        line = f"\r{self.desc} |{bar}| {pct:5.1f}% ({self.n}/{self.total})"
        self.file.write(line)
        self.file.flush()
        self._last_line = line

    def update(self, n: int = 1) -> None:
        self.n = min(self.n + n, self.total)
        self._redraw(force=self.n >= self.total)


def _progress_ctx(total: int, desc: str, enabled: bool, unit: str = "it"):
    if not enabled or total <= 0:

        class _Null:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def update(self, n: int = 1):
                pass

            def set_description(self, *_a, **_k):
                pass

        return _Null()

    try:
        from tqdm import tqdm

        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            mininterval=0.15,
            smoothing=0.05,
            file=sys.stderr,
            dynamic_ncols=True,
        )
    except ImportError:
        return _SimpleProgressBar(total, desc, sys.stderr, mininterval=0.2)


def _load_grayscale(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import imageio.v3 as iio

        img = np.asarray(iio.imread(path))
    except Exception:
        import cv2

        img = np.asarray(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
    if img.ndim == 3:
        # RGB / BGR → luminance-ish average
        img = np.mean(img.astype(np.float64), axis=-1)
    return img.astype(np.float64)


def _maybe_rotate_image(
    img: np.ndarray, rot: str | None, w_orig: int, h_orig: int
) -> np.ndarray:
    if rot is None:
        return img
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "center_rotate for images requires OpenCV (cv2). "
            "Install opencv-python or run with center_rotate disabled."
        ) from e
    if rot == "cw":
        return np.asarray(
            cv2.rotate(img.astype(np.float32), cv2.ROTATE_90_CLOCKWISE), dtype=np.float64
        )
    if rot == "ccw":
        return np.asarray(
            cv2.rotate(img.astype(np.float32), cv2.ROTATE_90_COUNTERCLOCKWISE),
            dtype=np.float64,
        )
    raise ValueError(rot)


def _calibration_z_extent(calibration) -> tuple[float, float]:
    c0 = calibration[0]
    planes = []
    for layer in c0.calibration:
        planes.append(float(layer["posPlane"]))
    return float(min(planes)), float(max(planes))


def _rotated_image_hw(h0: int, w0: int, rot: str | None) -> tuple[int, int]:
    """Height, width after optional 90° rotation (same convention as ``_maybe_rotate_image``)."""
    if rot is None:
        return h0, w0
    if rot in ("cw", "ccw"):
        return w0, h0
    raise ValueError(rot)


def _parse_crop(s: str | None) -> tuple[int, int] | None:
    """
    Parse ``--crop``: ``512`` → 512×512; ``640,512`` → width×height (center crop).
    """
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) == 1:
        v = int(parts[0])
        if v < 1:
            raise ValueError("--crop size must be >= 1")
        return v, v
    if len(parts) == 2:
        cw, ch = int(parts[0]), int(parts[1])
        if cw < 1 or ch < 1:
            raise ValueError("--crop width and height must be >= 1")
        return cw, ch
    raise ValueError(f"Invalid --crop {s!r}; use e.g. 512 or 640,512")


def _center_crop_window(
    h: int, w: int, crop_w: int, crop_h: int
) -> tuple[int, int, int, int]:
    """
    Center crop in **global** pixel indices (after rotation, same frame as calibration).

    Returns ``(origin_col, origin_row, width, height)`` where ``origin_col`` /
    ``origin_row`` are the top-left of the crop in full-image column/row
    coordinates (0-based). Rays must use ``global_x = origin_col + local_col + 0.5``,
    not ``local_col`` alone.
    """
    cw = min(int(crop_w), w)
    ch = min(int(crop_h), h)
    if cw < 1 or ch < 1:
        raise ValueError("crop dimensions must be positive")
    col0 = (w - cw) // 2
    row0 = (h - ch) // 2
    return col0, row0, cw, ch


def _infer_xy_bounds(
    calibration,
    widths_px: list[int],
    heights_px: list[int],
    crops: list[tuple[int, int, int, int] | None] | None = None,
) -> tuple[float, float, float, float]:
    """
    Union of (x,y) over all cameras: calibration-plane samples at image corners.

    ``widths_px`` / ``heights_px`` are per-camera size **after rotation** (same
    space as ``pixel_to_ray``). If ``crops[cam]`` is ``(col0, row0, cw, ch)``,
    use that rectangle's corners instead of the full frame.
    """
    xs: list[float] = []
    ys: list[float] = []
    ncam = len(calibration)
    if len(widths_px) != ncam or len(heights_px) != ncam:
        raise ValueError("widths_px / heights_px must match number of cameras")
    if crops is None:
        crops = [None] * ncam
    if len(crops) != ncam:
        raise ValueError("crops must match number of cameras")

    for cam_idx, calib in enumerate(calibration):
        w, h = widths_px[cam_idx], heights_px[cam_idx]
        box = crops[cam_idx]
        if box is None:
            corner_pts = [(0.0, 0.0), (w - 1.0, 0.0), (0.0, h - 1.0), (w - 1.0, h - 1.0)]
        else:
            col0, row0, cw, ch = box
            corner_pts = [
                (float(col0), float(row0)),
                (float(col0 + cw - 1), float(row0)),
                (float(col0), float(row0 + ch - 1)),
                (float(col0 + cw - 1), float(row0 + ch - 1)),
            ]
        for x_pix, y_pix in corner_pts:
            xy = np.array([[x_pix, y_pix]], dtype=np.float64)
            xyz = calib.transform_to_real_world(xy)
            xs.extend(xyz[:, 0, :].ravel().tolist())
            ys.extend(xyz[:, 1, :].ravel().tolist())
    pad = 0.02 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def batch_pixel_to_rays(
    calib, xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized pixel → ray for many pixels at once.

    ``xy`` shape ``(N, 2)`` with ``[:,0]`` = global column, ``[:,1]`` = global row
    (pixel centers; include +0.5 in the coordinates if desired).

    Returns ``origins (N,3)``, ``directions (N,3)`` unit vectors.
    """
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (N, 2), got {xy.shape}")
    if xy.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xyz = calib.transform_to_real_world(xy)
    xyz0, dd = fit3dline(xyz)
    nrm = np.linalg.norm(dd, axis=1, keepdims=True)
    bad = nrm.ravel() < 1e-12
    nrm = np.maximum(nrm, 1e-12)
    d_unit = dd / nrm
    if np.any(bad):
        d_unit = d_unit.copy()
        d_unit[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return xyz0.astype(np.float64, copy=False), d_unit.astype(np.float64, copy=False)


def pixel_to_ray(calib, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ray origin (3,) and unit direction (3,) in real-world coordinates (mm).

    ``x``, ``y`` must be **global** pixel coordinates in the same frame as the
    calibration (and as ``rays.py`` / center finding): **x** = column index
    (horizontal, 0 at left), **y** = row index (vertical, 0 at top), typically
    at pixel centers (… + 0.5). For a center crop, pass ``crop_origin_col +
    local_col + 0.5``, **not** ``local_col + 0.5`` alone.
    """
    frame_xy = np.array([[x, y]], dtype=np.float64)
    xyz = calib.transform_to_real_world(frame_xy)
    xyz0, dd = fit3dline(xyz)
    o = xyz0[0].astype(np.float64)
    d = dd[0].astype(np.float64)
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        return o, np.array([0.0, 0.0, 1.0])
    return o, d / nrm


def _slab_enter_exit(
    o: np.ndarray,
    d: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
    clip_t_nonneg: bool = False,
) -> tuple[float, float] | None:
    """
    Line ``o + t * d`` clipped to axis-aligned box ``[bmin, bmax]``.

    Returns ``(t_enter, t_exit)`` with ``t_enter <= t_exit`` along ``+d``.
    If ``clip_t_nonneg``, restrict to the forward half-line ``t >= 0`` (legacy
    camera-half-ray). For volume integration use ``clip_t_nonneg=False`` so
    the full chord inside the box is used.
    """
    d = np.where(np.abs(d) < 1e-14, 1e-14 * np.sign(d + 1e-20), d)
    invd = 1.0 / d
    t0 = (bmin - o) * invd
    t1 = (bmax - o) * invd
    t_enter = np.max(np.minimum(t0, t1))
    t_exit = np.min(np.maximum(t0, t1))
    if not np.isfinite(t_enter) or not np.isfinite(t_exit) or t_exit <= t_enter:
        return None
    if clip_t_nonneg:
        t_enter = max(t_enter, 0.0)
    return float(t_enter), float(t_exit)


def orient_ray_z_increasing_through_box(
    o: np.ndarray,
    d: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return unit ``d`` so that, along the in-box chord of ``o + t d``, ``z``
    is non-decreasing as ``t`` goes from ``t_enter`` to ``t_exit``.

    This matches traversing the calibration volume from low ``z`` (e.g. -25 mm)
    toward high ``z`` (+25 mm) when the line is not parallel to the ``xy`` plane.
    """
    o = np.asarray(o, dtype=np.float64).ravel()[:3]
    d = np.asarray(d, dtype=np.float64).ravel()[:3]
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        return o, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = d / nrm
    seg = _slab_enter_exit(o, d, bmin, bmax, clip_t_nonneg=False)
    if seg is None:
        return o, d
    t0, t1 = seg
    z0 = float((o + t0 * d)[2])
    z1 = float((o + t1 * d)[2])
    if z1 < z0 - 1e-9:
        d = -d
    return o, d


def _aggregate_idx_weights(idx: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum ``w`` for each distinct ``idx`` (1-D, nonnegative int64)."""
    if idx.size == 0:
        return idx.astype(np.int64, copy=False), w.astype(np.float64, copy=False)
    order = np.argsort(idx, kind="mergesort")
    idx_s = idx[order]
    w_s = w[order]
    first = np.empty(idx_s.shape[0], dtype=bool)
    first[0] = True
    first[1:] = idx_s[1:] != idx_s[:-1]
    uniq = idx_s[first]
    boundaries = np.flatnonzero(first)
    ends = np.append(boundaries[1:], idx_s.shape[0])
    sums = np.add.reduceat(w_s, boundaries)
    return uniq.astype(np.int64, copy=False), sums.astype(np.float64, copy=False)


def ray_accumulate_lengths(
    o: np.ndarray,
    d_unit: np.ndarray,
    grid_origin: np.ndarray,
    spacing: np.ndarray,
    dims: tuple[int, int, int],
    dt_scale: float = 0.25,
    beam_sigma_mm: float = 0.0,
    beam_cut_sigmas: float = 4.0,
    beam_max_half_voxels: int = 32,
    beam_chunk_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate weights per voxel by stepping along the ray with
    ``dt = dt_scale * min(spacing)``.

    ``beam_sigma_mm <= 0`` (default): each sample deposits ``dt`` into the voxel
    that contains the sample (line / tube of zero thickness).

    ``beam_sigma_mm > 0`` (**accumulate** only): Gaussian pencil around the ray —
    at each sample ``p`` every voxel center ``c`` in a local index cube receives
    ``dt * exp(-r^2 / (2 \\sigma^2))`` where ``r`` is the transverse distance from
    ``c`` to the line through ``p`` along ``d_unit``. Voxels beyond
    ``beam_cut_sigmas * sigma`` (perpendicular) are skipped. The index half-width
    is capped at ``beam_max_half_voxels``.

    The beam path is vectorized in chunks along the ray; ``beam_chunk_samples``
    sets the maximum samples per chunk (0 = auto from stencil size and a memory
    budget).
    """
    nx, ny, nz = dims
    bmin = grid_origin
    bmax = grid_origin + spacing * np.array([nx, ny, nz], dtype=np.float64)
    seg = _slab_enter_exit(o, d_unit, bmin, bmax, clip_t_nonneg=False)
    if seg is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    t0, t1 = seg
    dt = float(dt_scale * float(np.min(spacing)))
    if dt <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    invs = 1.0 / spacing
    span = t1 - t0
    n_steps = max(1, int(np.ceil(span / dt)))
    t = t0 + dt * np.arange(n_steps, dtype=np.float64)
    t = t[t < t1]
    if t.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    d_u = np.asarray(d_unit, dtype=np.float64).reshape(3)
    d_u = d_u / max(np.linalg.norm(d_u), 1e-12)
    p_all = o.reshape(1, 3) + np.outer(t, d_u)
    ix = ((p_all[:, 0] - grid_origin[0]) * invs[0]).astype(np.int64, copy=False)
    iy = ((p_all[:, 1] - grid_origin[1]) * invs[1]).astype(np.int64, copy=False)
    iz = ((p_all[:, 2] - grid_origin[2]) * invs[2]).astype(np.int64, copy=False)
    valid = (
        (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
        & (iz >= 0)
        & (iz < nz)
    )
    if not np.any(valid):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    sigma = float(beam_sigma_mm)
    if sigma <= 0.0:
        idx = ix[valid] + iy[valid] * nx + iz[valid] * nx * ny
        weights = np.full(idx.size, dt, dtype=np.float64)
        uniq, lens = _aggregate_idx_weights(idx, weights)
        return uniq, lens

    cut2 = (float(beam_cut_sigmas) * sigma) ** 2
    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)
    h = int(
        np.ceil(float(beam_cut_sigmas) * sigma / float(np.min(spacing)))
    )
    h = max(1, min(h, int(beam_max_half_voxels)))
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    ox, oy, oz = float(grid_origin[0]), float(grid_origin[1]), float(grid_origin[2])

    rng = np.arange(-h, h + 1, dtype=np.int64)
    dI, dJ, dK = np.meshgrid(rng, rng, rng, indexing="ij")
    dI = dI.ravel()
    dJ = dJ.ravel()
    dK = dK.ravel()
    m_stencil = int(dI.size)
    # Float work per (sample × stencil): II,JJ,KK,cx,cy,cz,vx..,rp2,gw ≈ 15 arrays
    mem_budget = 6_000_000
    auto_chunk = max(1, mem_budget // max(15 * m_stencil, 1))
    chunk = int(beam_chunk_samples) if int(beam_chunk_samples) > 0 else auto_chunk

    vi = np.flatnonzero(valid)
    idx_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []

    for chunk_start in range(0, vi.size, chunk):
        sl = slice(chunk_start, min(chunk_start + chunk, vi.size))
        vsl = vi[sl]
        B = int(vsl.size)
        if B == 0:
            continue
        p_b = p_all[vsl]
        ix_b = ix[vsl]
        iy_b = iy[vsl]
        iz_b = iz[vsl]

        II = ix_b[:, np.newaxis] + dI[np.newaxis, :]
        JJ = iy_b[:, np.newaxis] + dJ[np.newaxis, :]
        KK = iz_b[:, np.newaxis] + dK[np.newaxis, :]

        inb = (
            (II >= 0)
            & (II < nx)
            & (JJ >= 0)
            & (JJ < ny)
            & (KK >= 0)
            & (KK < nz)
        )
        d0, d1, d2 = float(d_u[0]), float(d_u[1]), float(d_u[2])
        px = p_b[:, 0:1]
        py = p_b[:, 1:2]
        pz = p_b[:, 2:3]
        cx = ox + (II.astype(np.float64, copy=False) + 0.5) * sx
        cy = oy + (JJ.astype(np.float64, copy=False) + 0.5) * sy
        cz = oz + (KK.astype(np.float64, copy=False) + 0.5) * sz
        vx = cx - px
        vy = cy - py
        vz = cz - pz
        t = vx * d0 + vy * d1 + vz * d2
        ex = vx - t * d0
        ey = vy - t * d1
        ez = vz - t * d2
        rp2 = ex * ex + ey * ey + ez * ez
        mask = inb & (rp2 <= cut2)
        if not np.any(mask):
            continue
        gw = dt * np.exp(-rp2 * inv_2sig2)
        idx_b = II + JJ * np.int64(nx) + KK * np.int64(nx * ny)
        idx_parts.append(idx_b[mask].ravel())
        w_parts.append(gw[mask].ravel())

    if not idx_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    idx = np.concatenate(idx_parts)
    weights = np.concatenate(w_parts)
    uniq, lens = _aggregate_idx_weights(idx, weights)
    return uniq, lens


def build_voxel_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Axis-aligned voxels that **exactly** tile ``[xmin,xmax]×[ymin,ymax]×[zmin,zmax]``.

    ``dz = (zmax - zmin) / nz``. Along X and Y the cell count is the minimum
    ``ceil(span / dz)`` so cells are at most ``dz`` wide; ``dx`` and ``dy`` are
    then ``(xmax-xmin)/nx`` and ``(ymax-ymin)/ny``, typically slightly **below**
    ``dz``. There is no padding slab outside the requested bounds (unlike a
    fixed cubic grid with a centered XY origin).
    """
    if nz < 2:
        raise ValueError("nz must be at least 2")
    if zmax <= zmin:
        raise ValueError("zmax must be greater than zmin")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("xmax and ymax must exceed xmin and ymin")
    dz = (zmax - zmin) / nz
    nx = max(1, int(np.ceil((xmax - xmin) / dz)))
    ny = max(1, int(np.ceil((ymax - ymin) / dz)))
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    origin = np.array([xmin, ymin, zmin], dtype=np.float64)
    spacing = np.array([dx, dy, dz], dtype=np.float64)
    return origin, spacing, (nx, ny, nz)


def _build_sparse_ray_matrix(
    y_normed: list[tuple[np.ndarray, np.ndarray, float]],
    grid_origin: np.ndarray,
    spacing: np.ndarray,
    dims: tuple[int, int, int],
    n_vox: int,
    eps_pred: float,
    show_progress: bool,
):
    """
    Forward model ``y ≈ A @ x`` with one row per kept ray.

    Row ``i`` has ``A[i, j] = w_ij / sum_k w_ik`` (same weighting as accumulate),
    ``y_i`` is the per-camera normalized intensity.
    """
    try:
        from scipy.sparse import coo_matrix
    except ImportError as e:
        raise ImportError(
            "method 'pinv' requires scipy. Install with: pip install scipy"
        ) from e

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    y_rows: list[float] = []
    row = 0
    with _progress_ctx(
        len(y_normed), "Pinv: build A", show_progress, unit="ray"
    ) as pbar:
        for o, d, y in y_normed:
            pbar.update(1)
            idxs, w = ray_accumulate_lengths(o, d, grid_origin, spacing, dims)
            if idxs.size == 0:
                continue
            wsum = float(np.sum(w))
            if wsum < eps_pred:
                continue
            wn = (w / wsum).astype(np.float64, copy=False)
            for k in range(idxs.size):
                rows.append(row)
                cols.append(int(idxs[k]))
                data.append(float(wn[k]))
            y_rows.append(float(y))
            row += 1

    if not y_rows:
        raise RuntimeError("No valid rays intersect the volume for pinv.")

    A = coo_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(row, n_vox),
        dtype=np.float64,
    ).tocsr()
    y_vec = np.asarray(y_rows, dtype=np.float64)
    return A, y_vec


def _solve_pinv_or_lsmr(
    A_csr,
    y_vec: np.ndarray,
    rcond: float,
    pinv_max_elements: int,
    lsmr_damp: float | None,
) -> np.ndarray:
    """
    If ``n_rows * n_cols <= pinv_max_elements``, form dense ``A`` and use
    ``numpy.linalg.pinv(..., rcond=...)`` (Moore–Penrose with **relative**
    singular-value truncation, same ``rcond`` semantics as NumPy).

    Otherwise solve damped least squares with ``scipy.sparse.linalg.lsmr``
    (sparse); ``damp`` defaults to ``rcond`` (Tikhonov-style, **not** identical to
    truncated SVD — use a smaller grid / fewer rays to use the dense pinv path).
    """
    n_r, n_c = A_csr.shape
    use_dense = (n_r * n_c) <= pinv_max_elements

    if use_dense:
        A_dense = A_csr.toarray()
        return np.linalg.pinv(A_dense, rcond=rcond, hermitian=False) @ y_vec

    from scipy.sparse.linalg import lsmr

    damp = float(lsmr_damp) if lsmr_damp is not None else float(rcond)
    maxiter = min(50_000, max(2000, 4 * n_c))
    x, *_ = lsmr(
        A_csr,
        y_vec,
        damp=damp,
        atol=1e-10,
        btol=1e-10,
        maxiter=maxiter,
    )
    return x


def mart_update_ray(
    x: np.ndarray,
    idxs: np.ndarray,
    w: np.ndarray,
    y: float,
    relax: float,
    eps: float,
) -> None:
    """In-place MART for one ray: w_j are nonnegative weights (e.g. path lengths)."""
    if idxs.size == 0:
        return
    pred = float(np.sum(w * x[idxs]))
    pred = max(pred, eps)
    ratio = y / pred
    wsum2 = float(np.sum(w * w))
    if wsum2 < eps:
        return
    # x_j *= ratio^(relax * w_j / sum_k w_k^2)
    exponents = relax * w / wsum2
    x[idxs] *= np.power(ratio, exponents)


def run_mart(
    calibration,
    images: list[np.ndarray],
    widths_orig: list[int],
    heights_orig: list[int],
    center_rotate: list,
    grid_origin: np.ndarray,
    spacing: np.ndarray,
    dims: tuple[int, int, int],
    pixel_stride: int,
    intensity_floor: float,
    mart_iters: int,
    relax: float,
    seed: int,
    show_progress: bool = True,
    method: str = "accumulate",
    crop_boxes: list[tuple[int, int, int, int] | None] | None = None,
    pinv_rcond: float = 1e-3,
    pinv_max_elements: int = 25_000_000,
    pinv_lsmr_damp: float | None = None,
    accumulate_beam_sigma_mm: float = 0.0,
    accumulate_beam_chunk_samples: int = 0,
    black_pixel_max: float = -1.0,
) -> np.ndarray:
    if method not in ("accumulate", "mart", "pinv"):
        raise ValueError(
            f"method must be 'accumulate', 'mart', or 'pinv', got {method!r}"
        )
    if accumulate_beam_sigma_mm > 0.0 and method != "accumulate":
        raise ValueError(
            "accumulate_beam_sigma_mm > 0 is only supported for method='accumulate'"
        )
    ncam = len(images)
    if crop_boxes is None:
        crop_boxes = [None] * ncam
    if len(crop_boxes) != ncam:
        raise ValueError("crop_boxes must have one entry per camera")
    rng = np.random.default_rng(seed)
    nx, ny, nz = dims
    n_vox = nx * ny * nz
    bmin = np.asarray(grid_origin, dtype=np.float64).reshape(3)
    bmax = bmin + np.asarray(spacing, dtype=np.float64).reshape(3) * np.array(
        [nx, ny, nz], dtype=np.float64
    )

    # (cam_idx, cropped image, calib, global_col_offset, global_row_offset)
    prepared: list[tuple[int, np.ndarray, object, int, int]] = []
    for cam_idx, img in enumerate(images):
        rot = center_rotate[cam_idx]
        h0, w0 = heights_orig[cam_idx], widths_orig[cam_idx]
        img_r = _maybe_rotate_image(img, rot, w0, h0)
        calib = calibration[cam_idx]
        flat = img_r.ravel()
        bg = np.percentile(flat, 5.0)
        img_c = np.clip(img_r - bg, 0.0, None)
        global_col_offset = global_row_offset = 0
        box = crop_boxes[cam_idx]
        if box is not None:
            c0, r0, cw, ch = box
            img_c = img_c[r0 : r0 + ch, c0 : c0 + cw]
            global_col_offset, global_row_offset = c0, r0
        prepared.append(
            (cam_idx, img_c, calib, global_col_offset, global_row_offset)
        )

    total_samples = sum(
        _strided_sample_count(im.shape[0], pixel_stride)
        * _strided_sample_count(im.shape[1], pixel_stride)
        for _, im, _, _, _ in prepared
    )

    rays: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    with _progress_ctx(
        total_samples, "Rays (pixels)", show_progress, unit="sample"
    ) as pbar:
        for cam_idx, img_c, calib, gcol0, grow0 in prepared:
            h, w = img_c.shape[0], img_c.shape[1]
            rows_i = np.arange(0, h, pixel_stride, dtype=np.int64)
            cols_i = np.arange(0, w, pixel_stride, dtype=np.int64)
            grid_r, grid_c = np.meshgrid(rows_i, cols_i, indexing="ij")
            vals = img_c[grid_r, grid_c].astype(np.float64, copy=False).ravel()
            global_x = (gcol0 + grid_c.astype(np.float64) + 0.5).ravel()
            global_y = (grow0 + grid_r.astype(np.float64) + 0.5).ravel()
            mask = vals >= intensity_floor
            if black_pixel_max >= 0.0:
                mask &= vals > float(black_pixel_max)
            pbar.update(grid_r.size)
            if not np.any(mask):
                continue
            xy = np.column_stack([global_x[mask], global_y[mask]])
            vals_kept = vals[mask]
            xyz0, d_unit = batch_pixel_to_rays(calib, xy)
            for i in range(vals_kept.size):
                o_i, d_i = orient_ray_z_increasing_through_box(
                    xyz0[i], d_unit[i], bmin, bmax
                )
                rays.append((cam_idx, float(vals_kept[i]), o_i, d_i))

    if not rays:
        raise RuntimeError(
            "No rays kept: relax --intensity-floor / --black-pixel-max, "
            "increase stride, or check images."
        )

    # Per-camera normalization of projection data to unit mean (stability)
    sums = {}
    counts = {}
    for cam_idx, val, _, _ in rays:
        sums[cam_idx] = sums.get(cam_idx, 0.0) + val
        counts[cam_idx] = counts.get(cam_idx, 0) + 1
    y_normed: list[tuple[np.ndarray, np.ndarray, float]] = []
    for cam_idx, val, o, d in rays:
        mean = sums[cam_idx] / max(counts[cam_idx], 1)
        y_normed.append((o, d, val / max(mean, 1e-12)))

    order = np.arange(len(y_normed))
    eps_pred = 1e-9
    eps_x = 1e-12

    n_rays = len(y_normed)

    if method == "pinv":
        A, y_vec = _build_sparse_ray_matrix(
            y_normed,
            grid_origin,
            spacing,
            dims,
            n_vox,
            eps_pred,
            show_progress,
        )
        dense_elems = int(A.shape[0]) * int(A.shape[1])
        use_dense = dense_elems <= int(pinv_max_elements)
        print(
            f"pinv: {A.shape[0]} rays × {n_vox} voxels, nnz={A.nnz}, "
            f"{'dense np.linalg.pinv' if use_dense else 'sparse lsmr'} "
            f"(rcond={pinv_rcond})",
            flush=True,
        )
        x = _solve_pinv_or_lsmr(
            A, y_vec, pinv_rcond, pinv_max_elements, pinv_lsmr_damp
        )
        np.clip(x, 0.0, None, out=x)
        return x.reshape((nx, ny, nz))

    x = np.zeros(n_vox, dtype=np.float64) if method == "accumulate" else np.ones(
        n_vox, dtype=np.float64
    )

    if method == "accumulate":
        # x was created as zeros above; only in-place += (no multiplicative step).
        with _progress_ctx(n_rays, "Accumulate rays", show_progress, unit="ray") as pbar:
            for k in range(n_rays):
                pbar.update(1)
                o, d, y = y_normed[k]
                idxs, w = ray_accumulate_lengths(
                    o,
                    d,
                    grid_origin,
                    spacing,
                    dims,
                    beam_sigma_mm=accumulate_beam_sigma_mm,
                    beam_chunk_samples=accumulate_beam_chunk_samples,
                )
                if idxs.size == 0:
                    continue
                wsum = float(np.sum(w))
                if wsum < eps_pred:
                    continue
                x[idxs] += y * (w / wsum)
        return x.reshape((nx, ny, nz))

    mart_total = mart_iters * n_rays
    with _progress_ctx(
        mart_total, f"MART 1/{mart_iters}", show_progress, unit="ray"
    ) as pbar:
        for it in range(mart_iters):
            pbar.set_description(f"MART {it + 1}/{mart_iters}", refresh=False)
            rng.shuffle(order)
            for k in order:
                pbar.update(1)
                o, d, y = y_normed[k]
                idxs, w = ray_accumulate_lengths(o, d, grid_origin, spacing, dims)
                if idxs.size == 0:
                    continue
                mart_update_ray(x, idxs, w, y, relax, eps_pred)
            np.clip(x, eps_x, None, out=x)

    return x.reshape((nx, ny, nz))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="4-camera voxel volume from images (accumulate, MART, or pinv)"
    )
    p.add_argument(
        "--folder",
        type=str,
        default=str(_REPO_ROOT / "raw_data" / "tracking_tests"),
        help="Experiment folder where import_calibration finds calib.h5",
    )
    p.add_argument(
        "--images",
        type=str,
        required=True,
        help="Comma-separated paths to raw images, one per camera in calib order",
    )
    p.add_argument(
        "--nz",
        type=int,
        required=True,
        help="Number of voxels along Z; dz=(zmax-zmin)/nz; dx,dy ≤ dz so XY fits "
        "xmin..xmax and ymin..ymax with no extra margin",
    )
    p.add_argument(
        "--zmin",
        type=float,
        default=None,
        help="Volume Z minimum (mm); must be used together with --zmax",
    )
    p.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="Volume Z maximum (mm); must be used together with --zmin",
    )
    p.add_argument("--pixel-stride", type=int, default=8, help="Subsample pixels (>=1)")
    p.add_argument(
        "--intensity-floor",
        type=float,
        default=0.0,
        help="Skip pixels below this value (after 5%% background subtraction)",
    )
    p.add_argument(
        "--black-pixel-max",
        type=float,
        default=1.0,
        metavar="V",
        help="After background subtraction, treat pixels with value <= V as black "
        "(skipped). Use -1 to disable. Default 1: only intensities strictly above 1 "
        "are used as signal (so 0 and 1 are black).",
    )
    p.add_argument(
        "--crop",
        type=str,
        default=None,
        metavar="W[,H]",
        help="Center crop after rotation; rays still use full-frame pixel "
        "(column,row) + 0.5 for calibration, not 0..W-1 inside the crop only.",
    )
    p.add_argument(
        "--method",
        choices=("accumulate", "mart", "pinv"),
        default="accumulate",
        help="accumulate: zero + sum along rays; mart: multiplicative iterations; "
        "pinv: min ||A x - y|| via Moore-Penrose when small, else damped lsmr",
    )
    p.add_argument(
        "--accumulate-beam-sigma",
        type=float,
        default=0.0,
        metavar="MM",
        help="For method accumulate: transverse Gaussian beam width σ in mm "
        "(same units as grid). 0 = infinitesimal tube (default). "
        "Each ray step spreads weight with exp(-r²/(2σ²)) in the plane perpendicular "
        "to the ray, truncated at ~4σ.",
    )
    p.add_argument(
        "--accumulate-beam-chunk-samples",
        type=int,
        default=0,
        metavar="N",
        help="When using --accumulate-beam-sigma: max ray samples per vectorized "
        "chunk (0 = auto from stencil size and a memory budget). Increase on large "
        "RAM if profiling shows chunk overhead.",
    )
    p.add_argument(
        "--pinv-rcond",
        type=float,
        default=1e-3,
        help="Relative SVD truncation for numpy.linalg.pinv (pinv method); "
        "singular values <= rcond * smax are discarded. Also default damp for lsmr fallback.",
    )
    p.add_argument(
        "--pinv-max-elements",
        type=int,
        default=25_000_000,
        help="If n_rays*n_voxels <= this, use dense pinv; else sparse lsmr (pinv only)",
    )
    p.add_argument(
        "--pinv-lsmr-damp",
        type=float,
        default=None,
        help="Optional Tikhonov damp for sparse lsmr fallback (pinv). "
        "Default: same as --pinv-rcond.",
    )
    p.add_argument(
        "--mart-iters",
        type=int,
        default=4,
        help="Full passes over all rays (mart method only)",
    )
    p.add_argument(
        "--relax",
        type=float,
        default=0.15,
        help="MART relaxation in (0,1] (mart method only)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--center-rotate",
        type=str,
        default=None,
        help="Same as rays.Rays: None, 'pairwise', or 'cw,cw,ccw,ccw' per camera",
    )
    p.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Sensor width before rotation (required if any center_rotate)",
    )
    p.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Sensor height before rotation (required if any center_rotate)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional path to save voxel volume as .npy (C-order shape nx,ny,nz)",
    )
    p.add_argument(
        "--vtk",
        type=str,
        default="",
        help="Explicit path for VTK legacy STRUCTURED_POINTS binary (.vtk). "
        "If omitted with --out, writes <out>.vtk next to the .npy.",
    )
    p.add_argument(
        "--no-vtk",
        action="store_true",
        help="Do not write auto .vtk beside --out (or beside --vti when no --out)",
    )
    p.add_argument(
        "--vti",
        type=str,
        default="",
        help="Optional VTK XML ImageData (.vti), appended raw format (scalar 'mart')",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars (stderr)",
    )
    args = p.parse_args(argv)
    try:
        crop_spec = _parse_crop(args.crop)
    except ValueError as e:
        p.error(str(e))

    calib_arr = import_calibration(args.folder)
    ncam = calib_arr.shape[0]
    image_paths = [s.strip() for s in args.images.split(",") if s.strip()]
    if len(image_paths) != ncam:
        raise ValueError(
            f"Expected {ncam} image paths (calibration cameras), got {len(image_paths)}"
        )

    center_rotate = normalize_center_rotate_per_camera(args.center_rotate, ncam)
    w_list, h_list = normalize_image_size_per_camera(
        args.image_width, args.image_height, ncam, center_rotate
    )

    images = [_load_grayscale(path) for path in image_paths]
    widths_orig: list[int] = []
    heights_orig: list[int] = []
    for cam_idx, (im, path) in enumerate(zip(images, image_paths)):
        heights_orig.append(int(im.shape[0]))
        widths_orig.append(int(im.shape[1]))
        if w_list is not None and center_rotate[cam_idx]:
            exp_w, exp_h = w_list[cam_idx], h_list[cam_idx]
            if exp_w != widths_orig[-1] or exp_h != heights_orig[-1]:
                raise ValueError(
                    f"Image {path} size {widths_orig[-1]}x{heights_orig[-1]} does not match "
                    f"declared sensor size {exp_w}x{exp_h} for rotation"
                )

    widths_px: list[int] = []
    heights_px: list[int] = []
    crop_boxes: list[tuple[int, int, int, int] | None] = []
    for cam_idx in range(ncam):
        h0, w0 = heights_orig[cam_idx], widths_orig[cam_idx]
        hr, wr = _rotated_image_hw(h0, w0, center_rotate[cam_idx])
        widths_px.append(wr)
        heights_px.append(hr)
        if crop_spec is None:
            crop_boxes.append(None)
        else:
            crop_boxes.append(
                _center_crop_window(hr, wr, crop_spec[0], crop_spec[1])
            )

    if (args.zmin is None) ^ (args.zmax is None):
        p.error("use both --zmin and --zmax together, or neither")
    if args.zmin is not None:
        zmin, zmax = float(args.zmin), float(args.zmax)
        if zmin >= zmax:
            p.error("--zmin must be less than --zmax")
    else:
        zmin, zmax = _calibration_z_extent(calib_arr)
    xmin, xmax, ymin, ymax = _infer_xy_bounds(calib_arr, widths_px, heights_px, crop_boxes)
    origin, spacing, dims = build_voxel_grid(xmin, xmax, ymin, ymax, zmin, zmax, args.nz)

    print(
        f"Grid: dims={dims} spacing={spacing} mm origin={origin} "
        f"Z=[{zmin:.3f},{zmax:.3f}] mm"
    )
    if float(args.black_pixel_max) >= 0.0:
        print(
            f"Signal pixels: intensity >= {float(args.intensity_floor):.4g} and "
            f"> {float(args.black_pixel_max):.4g} after 5% background subtraction.",
            flush=True,
        )
    if crop_spec is not None:
        print(
            "Crop active: calibration uses global pixel (x=col, y=row) with "
            "origin offset per camera (see lines below).",
            flush=True,
        )
        for i, box in enumerate(crop_boxes):
            if box is not None:
                c0, r0, cw, ch = box
                print(
                    f"Camera {i} center crop: global origin (col,row)=({c0},{r0}) "
                    f"size={cw}×{ch} (sensor after rotate {widths_px[i]}×{heights_px[i]})",
                    flush=True,
                )

    if args.method == "accumulate":
        print(
            "Method 'accumulate': voxel volume starts at all zeros; updates are "
            "additive (+=) only.",
            flush=True,
        )
        if float(args.accumulate_beam_sigma) > 0.0:
            print(
                f"  Gaussian beam σ={float(args.accumulate_beam_sigma):.4g} mm "
                "(transverse to each ray segment).",
                flush=True,
            )
    elif args.method == "pinv":
        print(
            "Method 'pinv': solve y ≈ A @ x with numpy.linalg.pinv (truncated SVD "
            "via rcond) when small enough, else scipy.sparse.linalg.lsmr.",
            flush=True,
        )

    if float(args.accumulate_beam_sigma) > 0.0 and args.method != "accumulate":
        p.error("--accumulate-beam-sigma is only valid with --method accumulate")

    vol = run_mart(
        calib_arr,
        images,
        widths_orig,
        heights_orig,
        center_rotate,
        origin,
        spacing,
        dims,
        max(1, args.pixel_stride),
        args.intensity_floor,
        args.mart_iters,
        args.relax,
        args.seed,
        show_progress=not args.no_progress,
        method=args.method,
        crop_boxes=crop_boxes,
        pinv_rcond=args.pinv_rcond,
        pinv_max_elements=args.pinv_max_elements,
        pinv_lsmr_damp=args.pinv_lsmr_damp,
        accumulate_beam_sigma_mm=float(args.accumulate_beam_sigma),
        accumulate_beam_chunk_samples=int(args.accumulate_beam_chunk_samples),
        black_pixel_max=float(args.black_pixel_max),
    )

    print(
        f"Voxel volume shape (nx,ny,nz)={vol.shape}, "
        f"min={vol.min():.4g} max={vol.max():.4g} mean={vol.mean():.4g}"
    )

    vtk_path: Path | None = None
    if args.vtk.strip():
        vtk_path = Path(args.vtk.strip())
    elif not args.no_vtk:
        if args.out:
            vtk_path = Path(args.out).with_suffix(".vtk")
        elif args.vti:
            vtk_path = Path(args.vti).with_suffix(".vtk")

    if vtk_path is not None:
        vtk_path.parent.mkdir(parents=True, exist_ok=True)
        write_vtk_legacy_structured_points_binary(
            vtk_path,
            vol,
            origin,
            spacing,
            scalar_name="mart",
            title=vtk_path.stem,
        )
        print(f"Wrote VTK legacy BINARY (ParaView): {vtk_path}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, vol)
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.npz")
        np.savez(
            meta_path,
            grid_origin=origin,
            grid_spacing=spacing,
            dims=np.array(dims),
            zmin=zmin,
            zmax=zmax,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        print(f"Wrote {out_path} and {meta_path}")

    if args.vti:
        vti_path = Path(args.vti)
        vti_path.parent.mkdir(parents=True, exist_ok=True)
        write_vti_image_data_appended(
            vti_path, vol, origin, spacing, scalar_name="mart"
        )
        print(f"Wrote VTK XML ImageData (.vti): {vti_path}")


if __name__ == "__main__":
    main()
