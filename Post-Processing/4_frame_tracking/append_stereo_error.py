"""Append stereomatching error (RMS + per-ray + xy/z split) onto tracks.h5part.

Joins each tracked particle to its STM match in ``rays_out_cpp.h5`` by exact
(x, y, z). The scalar ``error`` is the STM RMS (``xyze`` row 3). Per-ray
components ``error_0`` … ``error_{maxcams-1}`` are the perpendicular distances
from the triangulated point to each contributing ray (same slot order as
``intensity_0`` …), recomputed from ``rays.h5`` via ``camrayids``.

``error_xy`` / ``error_z`` split that same residual vector into the lab (x, y)
plane and the depth (z) axis:

    r_i = (P - A_i) - ((P - A_i) · d̂_i) d̂_i
    error_xy = rms_i(sqrt(r_ix^2 + r_iy^2))
    error_z  = rms_i(|r_iz|)

so ``error^2 ≈ error_xy^2 + error_z^2``.
"""

from __future__ import annotations

import os
import re

import h5py
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ERROR_XY_DESC = (
    "RMS of in-plane (x, y) components of ray-point residuals "
    "(perpendicular miss of each contributing ray)"
)
ERROR_Z_DESC = (
    "RMS of depth (z) components of ray-point residuals "
    "(perpendicular miss of each contributing ray)"
)


def _xyz_void(x, y, z):
    coords = np.ascontiguousarray(np.column_stack((x, y, z)))
    return coords.view(np.dtype((np.void, coords.dtype.itemsize * 3))).reshape(-1)


def match_xyz(tx, ty, tz, sx, sy, sz, max_dist=0.0):
    """Return STM index for each tracked (x, y, z), or -1 if unmatched.

    Exact bit-match by default. If ``max_dist`` > 0, fall back to nearest STM
    point within that distance (mm) for rows that did not match exactly.
    """
    n = int(np.asarray(tx).size)
    out = np.full(n, -1, dtype=np.int64)
    if n == 0 or np.asarray(sx).size == 0:
        return out
    sv = _xyz_void(sx, sy, sz)
    tv = _xyz_void(tx, ty, tz)
    order = np.argsort(sv, kind="mergesort")
    sorted_sv = sv[order]
    pos = np.searchsorted(sorted_sv, tv)
    pos = np.minimum(pos, len(sorted_sv) - 1)
    cand = order[pos]
    hit = sv[cand] == tv
    out[hit] = cand[hit]
    if max_dist and float(max_dist) > 0 and not np.all(hit):
        from scipy.spatial import cKDTree

        tree = cKDTree(np.column_stack((sx, sy, sz)))
        dist, idx = tree.query(np.column_stack((tx, ty, tz)), k=1)
        near = (~hit) & np.isfinite(dist) & (dist <= float(max_dist))
        out[near] = idx[near]
    return out


def infer_rays_h5(run_dir, rays_filename="rays.h5"):
    """Find rays.h5 for a run directory (same folder, or data → processed_data)."""
    same = os.path.join(run_dir, rays_filename)
    if os.path.isfile(same):
        return same
    replaced = run_dir.replace(f"{os.sep}data{os.sep}", f"{os.sep}processed_data{os.sep}")
    alt = os.path.join(replaced, rays_filename)
    if alt != same and os.path.isfile(alt):
        return alt
    return None


def infer_tracks_parquet(case_dir, tracks_name="tracks.h5part"):
    """Best-effort path to the combined case parquet (TTI_* or {case}_*)."""
    case = os.path.basename(os.path.normpath(case_dir))
    stem = os.path.splitext(os.path.basename(tracks_name))[0]
    names = []
    if case.startswith("TNTI_"):
        names.append(f"TTI_{case[5:]}_{stem}.parquet")
    names.append(f"{case}_{stem}.parquet")
    for name in names:
        path = os.path.join(case_dir, name)
        if os.path.isfile(path):
            return path
    if os.path.isdir(case_dir):
        for name in sorted(os.listdir(case_dir)):
            if not name.endswith(".parquet") or "STB" in name:
                continue
            if name.endswith(f"_{stem}.parquet"):
                return os.path.join(case_dir, name)
    return None


def _replace_dataset(grp, name, data):
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=data)


def _perp_residual(P, A, D):
    """Perpendicular residual vector from each ray (origin A, dir D) to point P."""
    nrm = np.linalg.norm(D, axis=1, keepdims=True)
    Dh = np.divide(D, nrm, out=np.zeros_like(D), where=nrm > 0)
    pa = P - A
    proj = np.sum(pa * Dh, axis=1, keepdims=True)
    return pa - proj * Dh


def _perp_dist(P, A, D):
    return np.linalg.norm(_perp_residual(P, A, D), axis=1)


def residual_rms_xy_z(residuals):
    """RMS of (x, y) and z residual components over camera slots.

    ``residuals`` is (n_particles, n_slots, 3) with NaN unused slots.
    """
    with np.errstate(all="ignore"):
        xy2 = residuals[:, :, 0] ** 2 + residuals[:, :, 1] ** 2
        z2 = residuals[:, :, 2] ** 2
        valid = np.isfinite(xy2)
        n_valid = np.sum(valid, axis=1)
        xy_sum = np.nansum(xy2, axis=1)
        z_sum = np.nansum(z2, axis=1)
        error_xy = np.sqrt(np.divide(xy_sum, n_valid, out=np.full_like(xy_sum, np.nan), where=n_valid > 0))
        error_z = np.sqrt(np.divide(z_sum, n_valid, out=np.full_like(z_sum, np.nan), where=n_valid > 0))
    return error_xy, error_z


def _camera_origins_dirs(rays_file, frame_key, n_cameras):
    origins = []
    dirs = []
    for cam_id in range(n_cameras):
        cam = f"Camera {cam_id}"
        if rays_file is None or cam not in rays_file or frame_key not in rays_file[cam]:
            origins.append(None)
            dirs.append(None)
            continue
        g = rays_file[cam][frame_key]
        origins.append(np.asarray(g["xyz0"][()], dtype=np.float64))
        dirs.append(np.asarray(g["dd"][()], dtype=np.float64))
    return origins, dirs


def per_ray_residuals(xyze, camrayids, stm_index, origins, dirs):
    """Residual vectors (n, maxcams, 3) for each STM camrayids slot."""
    n = int(stm_index.size)
    maxcams = int(camrayids.shape[0] // 2)
    residuals = np.full((n, maxcams, 3), np.nan, dtype=np.float64)
    good = stm_index >= 0
    if not np.any(good) or all(o is None for o in origins):
        return residuals
    for slot in range(maxcams):
        si = np.maximum(stm_index, 0)
        camid = np.where(good, camrayids[2 * slot, si], -1)
        rayid = np.where(good, camrayids[2 * slot + 1, si], -1)
        for cam in range(len(origins)):
            if origins[cam] is None:
                continue
            n_rays = origins[cam].shape[0]
            sel = (camid == cam) & (rayid >= 0) & (rayid < n_rays)
            if not np.any(sel):
                continue
            rid = rayid[sel]
            si_sel = stm_index[sel]
            P = np.column_stack([xyze[0, si_sel], xyze[1, si_sel], xyze[2, si_sel]])
            residuals[sel, slot] = _perp_residual(
                P, origins[cam][rid], dirs[cam][rid]
            )
    return residuals


def per_ray_errors(xyze, camrayids, stm_index, origins, dirs):
    """Perpendicular distance to each STM camrayids slot for matched particles."""
    residuals = per_ray_residuals(xyze, camrayids, stm_index, origins, dirs)
    with np.errstate(all="ignore"):
        return np.linalg.norm(residuals, axis=2)


def _step_keys(tracks_file):
    keys = [k for k in tracks_file.keys() if k.startswith("Step#")]
    keys.sort(key=lambda k: int(re.match(r"Step#(\d+)$", k).group(1)))
    return keys


def _write_error_dataset(grp, name, data, description):
    _replace_dataset(grp, name, data)
    grp[name].attrs["units"] = "mm"
    grp[name].attrs["description"] = description


def _frame_error_components(tx, ty, tz, xyze, camrayids, origins, dirs, max_dist=0.0):
    """Return error, error_xy, error_z, per-ray distances for one frame."""
    n = int(np.asarray(tx).size)
    err = np.full(n, np.nan, dtype=np.float64)
    err_xy = np.full(n, np.nan, dtype=np.float64)
    err_z = np.full(n, np.nan, dtype=np.float64)
    maxcams = int(camrayids.shape[0] // 2) if camrayids.size else 0
    comps = np.full((n, maxcams), np.nan, dtype=np.float64)
    if n == 0 or xyze.size == 0:
        return err, err_xy, err_z, comps
    idx = match_xyz(tx, ty, tz, xyze[0], xyze[1], xyze[2], max_dist=max_dist)
    good = idx >= 0
    if xyze.shape[0] >= 4:
        err[good] = xyze[3, idx[good]]
    residuals = per_ray_residuals(xyze, camrayids, idx, origins, dirs)
    comps = np.linalg.norm(residuals, axis=2)
    xy, zz = residual_rms_xy_z(residuals)
    err_xy[good] = xy[good]
    err_z[good] = zz[good]
    return err, err_xy, err_z, comps


def h5part_has_steps(tracks_path):
    if not os.path.isfile(tracks_path) or os.path.getsize(tracks_path) < 1024:
        return False
    try:
        with h5py.File(tracks_path, "r") as f:
            return any(k.startswith("Step#") for k in f.keys())
    except OSError:
        return False


def append_stereo_error_to_parquet(
    parquet_path,
    case_dir,
    *,
    stm_filename="rays_out_cpp.h5",
    rays_filename="rays.h5",
    show_progress=True,
    max_dist=0.0,
):
    """Compute error / error_xy / error_z from parquet (x,y,z) and write them back."""
    import pandas as pd
    import pyarrow.parquet as pq

    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(parquet_path)

    tbl = pq.read_table(parquet_path, columns=["run", "frame", "x", "y", "z"])
    run = np.asarray(tbl.column("run").to_pylist(), dtype=object)
    frame = np.asarray(tbl.column("frame").to_numpy())
    tx = np.asarray(tbl.column("x").to_numpy(), dtype=np.float64)
    ty = np.asarray(tbl.column("y").to_numpy(), dtype=np.float64)
    tz = np.asarray(tbl.column("z").to_numpy(), dtype=np.float64)
    n = int(tx.size)
    err = np.full(n, np.nan, dtype=np.float64)
    err_xy = np.full(n, np.nan, dtype=np.float64)
    err_z = np.full(n, np.nan, dtype=np.float64)
    per_ray = None
    n_matched = 0

    df = pd.DataFrame({"run": run, "frame": frame, "i": np.arange(n)})
    run_groups = list(df.groupby("run", sort=False))
    run_iter = run_groups
    if show_progress and tqdm is not None:
        run_iter = tqdm(run_groups, desc=os.path.basename(parquet_path), unit="run")

    for run_label, g_run in run_iter:
        run_dir = os.path.join(case_dir, str(run_label))
        stm_path = os.path.join(run_dir, stm_filename)
        if not os.path.isfile(stm_path):
            continue
        try:
            stm_probe = h5py.File(stm_path, "r")
            stm_probe.close()
        except OSError as exc:
            print(f"  skipping {run_label}: cannot read {stm_filename} ({exc})")
            continue
        rays_path = infer_rays_h5(run_dir, rays_filename)
        use_rays = bool(rays_path) and os.path.isfile(rays_path)
        if use_rays and os.path.getmtime(rays_path) > os.path.getmtime(stm_path) + 3600:
            print(
                f"  skipping residual split for {run_label}: "
                f"rays.h5 is newer than {stm_filename}"
            )
            use_rays = False
        rays_file = h5py.File(rays_path, "r") if use_rays else None
        try:
            with h5py.File(stm_path, "r") as stm:
                frame_groups = list(g_run.groupby("frame", sort=True))
                fr_iter = frame_groups
                if show_progress and tqdm is not None:
                    fr_iter = tqdm(
                        frame_groups,
                        desc=str(run_label),
                        unit="frame",
                        leave=False,
                    )
                for fr, g in fr_iter:
                    rows = g["i"].to_numpy()
                    frame_key = f"frame{int(fr):05d}"
                    if frame_key not in stm:
                        continue
                    xyze = np.asarray(stm[frame_key]["xyze"][()], dtype=np.float64)
                    camrayids = np.asarray(stm[frame_key]["camrayids"][()])
                    maxcams = int(camrayids.shape[0] // 2)
                    origins, dirs = _camera_origins_dirs(rays_file, frame_key, maxcams)
                    e, exy, ez, comps = _frame_error_components(
                        tx[rows],
                        ty[rows],
                        tz[rows],
                        xyze,
                        camrayids,
                        origins,
                        dirs,
                        max_dist=max_dist,
                    )
                    if comps.size:
                        with np.errstate(all="ignore"):
                            n_valid = np.sum(np.isfinite(comps), axis=1)
                            recomputed = np.sqrt(
                                np.nansum(np.square(comps), axis=1)
                                / np.maximum(n_valid, 1)
                            )
                            recomputed = np.where(n_valid > 0, recomputed, np.nan)
                        disagree = (
                            np.isfinite(e)
                            & np.isfinite(recomputed)
                            & (np.abs(recomputed - e) > (1e-3 + 0.05 * np.abs(e)))
                        )
                        if np.any(disagree):
                            exy = np.where(disagree, np.nan, exy)
                            ez = np.where(disagree, np.nan, ez)
                            comps = comps.copy()
                            comps[disagree] = np.nan
                    err[rows] = e
                    err_xy[rows] = exy
                    err_z[rows] = ez
                    n_matched += int(np.sum(np.isfinite(e)))
                    if per_ray is None:
                        per_ray = np.full((n, comps.shape[1]), np.nan, dtype=np.float64)
                    elif comps.shape[1] > per_ray.shape[1]:
                        extra = np.full(
                            (n, comps.shape[1] - per_ray.shape[1]),
                            np.nan,
                            dtype=np.float64,
                        )
                        per_ray = np.concatenate([per_ray, extra], axis=1)
                    per_ray[rows, : comps.shape[1]] = comps
        finally:
            if rays_file is not None:
                rays_file.close()

    extra = {
        "error": err,
        "error_xy": err_xy,
        "error_z": err_z,
        "error_attr_units": np.full(n, "mm", dtype=object),
        "error_attr_description": np.full(
            n, "STM RMS stereomatching error (xyze row 3)", dtype=object
        ),
        "error_xy_attr_units": np.full(n, "mm", dtype=object),
        "error_xy_attr_description": np.full(n, ERROR_XY_DESC, dtype=object),
        "error_z_attr_units": np.full(n, "mm", dtype=object),
        "error_z_attr_description": np.full(n, ERROR_Z_DESC, dtype=object),
    }
    if per_ray is not None:
        for slot in range(per_ray.shape[1]):
            extra[f"error_{slot}"] = per_ray[:, slot]
            extra[f"error_{slot}_attr_units"] = np.full(n, "mm", dtype=object)
            extra[f"error_{slot}_attr_description"] = np.full(
                n,
                "Perpendicular distance from match point to contributing ray "
                f"slot {slot} (same order as intensity_{slot})",
                dtype=object,
            )
    append_columns_to_parquet(parquet_path, extra)
    return {
        "parquet_path": parquet_path,
        "n_rows": n,
        "n_matched": n_matched,
        "n_miss": int(n - n_matched),
    }


def append_stereo_error_to_h5part(
    tracks_path,
    stm_path,
    rays_path=None,
    *,
    show_progress=True,
):
    """Write ``error``, ``error_*``, ``error_xy``, and ``error_z`` into every Step#."""
    if not os.path.isfile(tracks_path):
        raise FileNotFoundError(tracks_path)
    if not os.path.isfile(stm_path):
        raise FileNotFoundError(stm_path)
    rays_file = None
    if rays_path:
        if not os.path.isfile(rays_path):
            raise FileNotFoundError(rays_path)
        rays_file = h5py.File(rays_path, "r")

    n_matched = 0
    n_rows = 0
    n_miss = 0
    try:
        with h5py.File(stm_path, "r") as stm, h5py.File(tracks_path, "a") as tracks:
            step_keys = _step_keys(tracks)
            iterator = step_keys
            if show_progress and tqdm is not None:
                iterator = tqdm(
                    step_keys,
                    desc=os.path.basename(os.path.dirname(tracks_path)),
                    unit="step",
                    leave=True,
                )
            for sk in iterator:
                frame = int(re.match(r"Step#(\d+)$", sk).group(1))
                frame_key = f"frame{frame:05d}"
                grp = tracks[sk]
                tx = np.asarray(grp["x"][()], dtype=np.float64)
                ty = np.asarray(grp["y"][()], dtype=np.float64)
                tz = np.asarray(grp["z"][()], dtype=np.float64)
                n = int(tx.size)
                n_rows += n
                err = np.full(n, np.nan, dtype=np.float64)
                err_xy = np.full(n, np.nan, dtype=np.float64)
                err_z = np.full(n, np.nan, dtype=np.float64)
                if frame_key not in stm:
                    _write_error_dataset(
                        grp, "error", err, "STM RMS stereomatching error (xyze row 3)"
                    )
                    _write_error_dataset(grp, "error_xy", err_xy, ERROR_XY_DESC)
                    _write_error_dataset(grp, "error_z", err_z, ERROR_Z_DESC)
                    continue
                xyze = np.asarray(stm[frame_key]["xyze"][()], dtype=np.float64)
                camrayids = np.asarray(stm[frame_key]["camrayids"][()])
                idx = match_xyz(tx, ty, tz, xyze[0], xyze[1], xyze[2])
                good = idx >= 0
                n_matched += int(np.sum(good))
                n_miss += int(np.sum(~good))
                if xyze.shape[0] >= 4:
                    err[good] = xyze[3, idx[good]]
                maxcams = int(camrayids.shape[0] // 2)
                origins, dirs = _camera_origins_dirs(rays_file, frame_key, maxcams)
                residuals = per_ray_residuals(xyze, camrayids, idx, origins, dirs)
                comps = np.linalg.norm(residuals, axis=2)
                xy, zz = residual_rms_xy_z(residuals)
                err_xy[good] = xy[good]
                err_z[good] = zz[good]
                _write_error_dataset(
                    grp, "error", err, "STM RMS stereomatching error (xyze row 3)"
                )
                _write_error_dataset(grp, "error_xy", err_xy, ERROR_XY_DESC)
                _write_error_dataset(grp, "error_z", err_z, ERROR_Z_DESC)
                for slot in range(comps.shape[1]):
                    _write_error_dataset(
                        grp,
                        f"error_{slot}",
                        comps[:, slot],
                        "Perpendicular distance from match point to contributing ray "
                        f"slot {slot} (same order as intensity_{slot})",
                    )
                if show_progress and tqdm is not None and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(matched=n_matched, miss=n_miss, refresh=False)
    finally:
        if rays_file is not None:
            rays_file.close()

    return {
        "tracks_path": tracks_path,
        "n_rows": n_rows,
        "n_matched": n_matched,
        "n_miss": n_miss,
    }


def append_stereo_error_for_run(
    run_dir,
    *,
    tracks_name="tracks.h5part",
    stm_filename="rays_out_cpp.h5",
    rays_filename="rays.h5",
    rays_path=None,
    show_progress=True,
):
    tracks_path = os.path.join(run_dir, tracks_name)
    stm_path = os.path.join(run_dir, stm_filename)
    if rays_path is None:
        rays_path = infer_rays_h5(run_dir, rays_filename)
    return append_stereo_error_to_h5part(
        tracks_path,
        stm_path,
        rays_path=rays_path,
        show_progress=show_progress,
    )


def load_h5part_columns(path, names):
    """Load 1-D datasets from every Step# in the same order as combine_tracks."""
    chunks = {name: [] for name in names}
    n_rows = 0
    with h5py.File(path, "r") as f:
        for sk in _step_keys(f):
            grp = f[sk]
            dsnames = [
                k for k in grp.keys() if isinstance(grp[k], h5py.Dataset)
            ]
            if not dsnames:
                continue
            lens = [grp[k].shape[0] for k in dsnames if grp[k].ndim >= 1]
            if not lens:
                continue
            n = min(lens)
            n_rows += n
            for name in names:
                if name in grp and grp[name].ndim >= 1 and grp[name].shape[0] >= n:
                    chunks[name].append(np.asarray(grp[name][:n]))
                else:
                    chunks[name].append(np.full(n, np.nan, dtype=np.float64))
    out = {}
    for name, parts in chunks.items():
        if not parts:
            out[name] = np.array([], dtype=np.float64)
        else:
            out[name] = np.concatenate(parts)
    return out, n_rows


def _parquet_run_order(parquet_path):
    import pyarrow.parquet as pq

    runs = pq.read_table(parquet_path, columns=["run"]).column("run").to_pylist()
    order = []
    counts = {}
    for run in runs:
        if run not in counts:
            order.append(run)
            counts[run] = 0
        counts[run] += 1
    return order, counts, len(runs)


def append_columns_to_parquet(parquet_path, extra, *, compression="snappy"):
    """Rewrite ``parquet_path`` adding/replacing 1-D columns (streaming by row group)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows
    for name, arr in extra.items():
        if int(np.asarray(arr).shape[0]) != n_rows:
            raise ValueError(
                f"Column {name!r} length {np.asarray(arr).shape[0]} "
                f"!= parquet rows {n_rows} for {parquet_path}"
            )

    tmp_path = parquet_path + ".tmp"
    writer = None
    offset = 0
    try:
        iterator = range(pf.num_row_groups)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                desc=os.path.basename(parquet_path),
                unit="rg",
            )
        for i in iterator:
            tbl = pf.read_row_group(i)
            n = tbl.num_rows
            drop = [name for name in extra if name in tbl.column_names]
            if drop:
                tbl = tbl.drop(drop)
            for name, arr in extra.items():
                chunk = np.asarray(arr[offset : offset + n])
                if np.issubdtype(chunk.dtype, np.number):
                    col = pa.array(chunk, type=pa.float64())
                else:
                    col = pa.array(chunk, type=pa.string())
                tbl = tbl.append_column(name, col)
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path, tbl.schema, compression=compression
                )
            writer.write_table(tbl)
            offset += n
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        if writer is not None:
            writer.close()
    if offset != n_rows:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Wrote {offset} parquet rows, expected {n_rows} for {parquet_path}"
        )
    if os.path.isfile(tmp_path):
        os.replace(tmp_path, parquet_path)
    elif not os.path.isfile(parquet_path):
        raise FileNotFoundError(tmp_path)
    return parquet_path


def update_parquet_error_columns(
    case_dir,
    parquet_path=None,
    *,
    tracks_name="tracks.h5part",
    column_names=None,
):
    """Copy error columns from per-run h5part files onto the combined parquet."""
    import pyarrow.parquet as pq

    if parquet_path is None:
        parquet_path = infer_tracks_parquet(case_dir, tracks_name)
    if parquet_path is None or not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"No tracks parquet under {case_dir}")

    if column_names is None:
        column_names = [
            "error",
            "error_xy",
            "error_z",
            "error_0",
            "error_1",
            "error_2",
            "error_3",
        ]

    order, counts, n_parquet = _parquet_run_order(parquet_path)
    parts = {name: [] for name in column_names}
    check_x = []
    for run in order:
        h5_path = os.path.join(case_dir, run, tracks_name)
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(h5_path)
        cols, n_h5 = load_h5part_columns(h5_path, list(column_names) + ["x"])
        if n_h5 != counts[run]:
            raise RuntimeError(
                f"{run}: h5part has {n_h5} rows but parquet has {counts[run]} "
                f"in {parquet_path}"
            )
        check_x.append(cols["x"])
        for name in column_names:
            parts[name].append(cols[name])

    extra = {name: np.concatenate(parts[name]) for name in column_names}
    xs = np.concatenate(check_x)
    pq_x = pq.read_table(parquet_path, columns=["x"]).column("x").to_numpy()
    if xs.shape[0] != pq_x.shape[0] or not np.allclose(xs, pq_x, equal_nan=True):
        n_bad = int(np.sum(~np.isclose(xs, pq_x, equal_nan=True)))
        raise RuntimeError(
            f"Parquet/h5part x mismatch for {parquet_path}: {n_bad} differing rows"
        )

    extra["error_xy_attr_units"] = np.full(n_parquet, "mm", dtype=object)
    extra["error_xy_attr_description"] = np.full(n_parquet, ERROR_XY_DESC, dtype=object)
    extra["error_z_attr_units"] = np.full(n_parquet, "mm", dtype=object)
    extra["error_z_attr_description"] = np.full(n_parquet, ERROR_Z_DESC, dtype=object)
    extra["error_attr_units"] = np.full(n_parquet, "mm", dtype=object)
    extra["error_attr_description"] = np.full(
        n_parquet, "STM RMS stereomatching error (xyze row 3)", dtype=object
    )

    append_columns_to_parquet(parquet_path, extra)
    return {
        "parquet_path": parquet_path,
        "n_rows": n_parquet,
        "runs": order,
        "columns": column_names,
    }
