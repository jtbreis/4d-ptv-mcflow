import multiprocessing
import os
import sys
import tempfile

import h5py
import numpy as np

from ptv_calib.io.import_calibration import import_calibration

from ..io.read_h5 import (
    _sorted_frame_keys,
    count_h5_center_frames,
    iter_read_h5_centers,
    read_center_frame_from_group,
    read_h5_centers,
)
from .fit_line import fit3dline
from ..plotting.ray_testing import plot_rays
from ..utils.structure import Filenames, Folders


def _split_frame_ranges(n_frames: int, n_parts: int) -> list[tuple[int, int]]:
    """Split [0, n_frames) into n_parts contiguous half-open ranges (roughly equal size)."""
    if n_frames <= 0 or n_parts <= 0:
        return []
    n_parts = min(n_parts, n_frames)
    base = n_frames // n_parts
    rem = n_frames % n_parts
    ranges = []
    start = 0
    for i in range(n_parts):
        length = base + (1 if i < rem else 0)
        end = start + length
        if length > 0:
            ranges.append((start, end))
        start = end
    return ranges


def _effective_frame_count(n_src: int, max_frames: int | None) -> int:
    """Frames to process: min(n_src, max_frames) if max_frames is set, else n_src."""
    if max_frames is None:
        return n_src
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1 when set")
    return min(n_src, max_frames)


def _chunks_per_camera(n_workers: int, ncameras: int) -> int:
    """Ceil(n_workers / ncameras); at least 1 chunk budget per camera when parallelizing."""
    if n_workers < 1 or ncameras < 1:
        return 1
    return max(1, (n_workers + ncameras - 1) // ncameras)


def _progress_interval(total_frames: int) -> int:
    """Print roughly every 2% of frames (~50 lines max for long runs)."""
    if total_frames <= 0:
        return 1
    return max(1, total_frames // 50)


def _normalize_one_center_rotate_token(tok: str):
    t = tok.strip().lower()
    if t in ("", "0", "none", "no"):
        return None
    if t in ("cw", "clockwise", "90cw", "rot90_cw"):
        return "cw"
    if t in ("ccw", "counterclockwise", "counter-clockwise", "90ccw", "rot90_ccw"):
        return "ccw"
    raise ValueError(f"Unknown center rotation token {tok!r}; use cw, ccw, or none")


def normalize_center_rotate_per_camera(spec, ncameras: int) -> list:
    """
    Build a length-ncameras list of None | 'cw' | 'ccw'.

    spec: None -> all None.
    str 'pairwise' -> cameras 0–1: 90° CW, 2–3: 90° CCW (requires ncameras == 4).
    str 'cw,ccw,...' -> comma-separated per camera (same order as Camera 0..N-1).
    list/tuple -> same tokens, length must match ncameras.
    """
    if spec is None:
        return [None] * ncameras
    if isinstance(spec, (list, tuple)):
        if len(spec) != ncameras:
            raise ValueError(
                f"center_rotate list length {len(spec)} != ncameras {ncameras}"
            )
        out_list = []
        for x in spec:
            if x is None:
                out_list.append(None)
                continue
            out_list.append(_normalize_one_center_rotate_token(str(x)))
        return out_list
    if not isinstance(spec, str):
        raise TypeError(f"center_rotate spec must be str, list, or None, got {type(spec)}")
    s = spec.strip().lower()
    if not s:
        return [None] * ncameras
    if s == "pairwise":
        if ncameras != 4:
            raise ValueError(
                'center_rotate="pairwise" requires exactly 4 cameras '
                "(cams 0–1: 90° clockwise, 2–3: 90° counter-clockwise)"
            )
        return ["cw", "cw", "ccw", "ccw"]
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != ncameras:
        raise ValueError(
            f"center_rotate comma-list has {len(parts)} entries, need {ncameras} "
            f"(one per camera 0..{ncameras - 1})"
        )
    return [_normalize_one_center_rotate_token(p) for p in parts]


def normalize_image_size_per_camera(image_width, image_height, ncameras, rotate_per_cam: list):
    """
    Scalar image_width / image_height broadcast to all cameras; or pass a list/tuple of
    length ncameras for per-camera resolution (only entries for cameras with rotation
    must be positive).
    Returns (w_list, h_list) each length ncameras, or (None, None) if no camera rotates.
    """
    if not any(r is not None for r in rotate_per_cam):
        return None, None
    if image_width is None or image_height is None:
        raise ValueError(
            "image_width and image_height are required when center_rotate applies "
            "to any camera"
        )

    def expand_dim(val, name):
        if isinstance(val, (list, tuple)):
            if len(val) != ncameras:
                raise ValueError(
                    f"{name} list length {len(val)} must equal ncameras ({ncameras})"
                )
            return [int(v) for v in val]
        return [int(val)] * ncameras

    w_list = expand_dim(image_width, "image_width")
    h_list = expand_dim(image_height, "image_height")
    for i in range(ncameras):
        if rotate_per_cam[i] is None:
            continue
        if w_list[i] < 1 or h_list[i] < 1:
            raise ValueError(
                f"image width/height for camera {i} must be positive when "
                "its center_rotate is set"
            )
    return w_list, h_list


def _rotate_centers_xy_90(frame_data: dict, w: int, h: int, direction: str) -> dict:
    """
    Remap particle (x, y) as if the image were rotated 90° in place (OpenCV-style axes:
    x column left→right, y row top→bottom). Original image size is w×h; after rotation
    the pixel grid is h×w, and this returns coordinates in that rotated grid.
    """
    if direction not in ("cw", "ccw"):
        raise ValueError(direction)
    x = np.asarray(frame_data["x"], dtype=np.float64)
    y = np.asarray(frame_data["y"], dtype=np.float64)
    wf = float(w)
    hf = float(h)
    if direction == "cw":
        xn = hf - 1.0 - y
        yn = x
    else:
        xn = y
        yn = wf - 1.0 - x
    out = dict(frame_data)
    out["x"] = xn
    out["y"] = yn
    return out


def _dataset_kwargs(arr):
    """Compress large ray arrays to cut file size; small frames stay uncompressed."""
    if arr is None or arr.size == 0:
        return {}
    if arr.size >= 4096:
        return {"compression": "gzip", "compression_opts": 3, "shuffle": True}
    return {}


def _compute_frame_arrays(calibration_cam, frame_data, store_xyz_for_plotting=False):
    """Ray arrays for one frame (usable from worker processes)."""
    frame_xy = np.column_stack([frame_data["x"], frame_data["y"]])
    xyz = calibration_cam.transform_to_real_world(frame_xy)
    xyz0, dd = fit3dline(xyz)
    n_rays = len(frame_data["x"])
    diameter = np.full(n_rays, np.nan, dtype=np.float64)
    intensity = np.full(n_rays, np.nan, dtype=np.float64)
    mass = np.full(n_rays, np.nan, dtype=np.float64)
    for key, arr in (
        ("diameter", diameter),
        ("intensity", intensity),
        ("mass", mass),
    ):
        if key in frame_data:
            arr[:] = np.asarray(frame_data[key], dtype=np.float64)
    xyz_plot = xyz if store_xyz_for_plotting else None
    return xyz0, dd, diameter, intensity, mass, xyz_plot


def _write_frame_group_h5(grp, xyz0, dd, diameter, intensity, mass):
    grp.create_dataset("xyz0", data=xyz0, **_dataset_kwargs(xyz0))
    grp.create_dataset("dd", data=dd, **_dataset_kwargs(dd))
    grp.create_dataset("diameter", data=diameter, **_dataset_kwargs(diameter))
    grp.create_dataset("intensity", data=intensity,
                       **_dataset_kwargs(intensity))
    grp.create_dataset("mass", data=mass, **_dataset_kwargs(mass))


def _stream_camera_chunk_worker(args):
    """
    Process a contiguous [start, end) frame range for one camera; write partial rays.h5.
    """
    calib_path, cam_idx, center_file, start, end, out_partial_path, flush_every, rot_dir, img_w, img_h = args
    n_local = end - start
    n_src = count_h5_center_frames(center_file)
    step = _progress_interval(n_local)
    calib = import_calibration(calib_path)
    calibration_cam = calib[cam_idx]
    print(
        f"[rays] Camera {cam_idx} frames [{start},{end}) / {n_src} total "
        f"(h5 flush every {flush_every})",
        flush=True,
    )
    n_written = 0
    with h5py.File(out_partial_path, "w") as fout:
        camgrp = fout.create_group(f"Camera {cam_idx}")
        with h5py.File(center_file, "r") as fcent:
            frame_keys = _sorted_frame_keys(fcent)
            if len(frame_keys) != n_src:
                raise RuntimeError(
                    f"Camera {cam_idx}: center file {center_file!r} has "
                    f"{len(frame_keys)} frame group(s) but count_h5_center_frames "
                    f"reported {n_src}; refusing parallel read."
                )
            for seq_idx in range(start, end):
                grp_in = fcent[frame_keys[seq_idx]]
                frame_data = read_center_frame_from_group(grp_in)
                if rot_dir:
                    frame_data = _rotate_centers_xy_90(
                        frame_data, img_w, img_h, rot_dir)
                xyz0, dd, diameter, intensity, mass, _ = _compute_frame_arrays(
                    calibration_cam, frame_data, store_xyz_for_plotting=False
                )
                # Output names match sequential streaming: 0..N-1 by sorted order.
                grp = camgrp.create_group(f"frame{seq_idx:05d}")
                _write_frame_group_h5(grp, xyz0, dd, diameter, intensity, mass)
                n_written += 1
                if flush_every >= 1 and (seq_idx + 1) % flush_every == 0:
                    fout.flush()
                if (
                    n_written == 1
                    or n_written == n_local
                    or n_written % step == 0
                ):
                    pct = 100.0 * (start + n_written) / \
                        n_src if n_src else 100.0
                    print(
                        f"[rays] Camera {cam_idx} chunk: local {n_written}/{n_local} "
                        f"(global ~{pct:.1f}%)  seq {seq_idx}",
                        flush=True,
                    )
        fout.flush()
    print(
        f"[rays] Camera {cam_idx} chunk [{start},{end}): done, {n_written} frames",
        flush=True,
    )
    return cam_idx, start, end, n_written, out_partial_path


def _merge_ray_chunks(final_path, partials_per_camera: list, ncameras: int):
    """Merge ordered partial files per camera into one rays.h5 (frame groups in order)."""
    with h5py.File(final_path, "w") as fout:
        fout.attrs["ncameras"] = ncameras
        for cam_idx in range(ncameras):
            camgrp = fout.create_group(f"Camera {cam_idx}")
            for ppath in partials_per_camera[cam_idx]:
                with h5py.File(ppath, "r") as fin:
                    src = fin[f"Camera {cam_idx}"]
                    for name in sorted(
                        src.keys(),
                        key=lambda k: int(k.replace("frame", ""))
                        if k.replace("frame", "").isdigit()
                        else 0,
                    ):
                        fin.copy(f"Camera {cam_idx}/{name}", camgrp, name)


class Rays():
    def __init__(
        self,
        path,
        calibration_folder=None,
        store_xyz_for_plotting=False,
        center_rotate=None,
        image_width=None,
        image_height=None,
    ):
        """
        path: folder containing Centers/ and where rays will be read/written.
        calibration_folder: if set, load calib.h5 from this folder (e.g. test
            Calibration_Before calibration on points from Calibration_After).
        store_xyz_for_plotting: if True, keep full per-plane XYZ (large RAM);
            required only for plot_rays(). Default False to avoid OOM on dense data.
        center_rotate: optional per-camera remap of center (x,y) as if that camera's
            image were rotated 90° before calibration. Prefer a list/tuple of length
            ncameras with entries None | 'cw' | 'ccw' (camera index matches calib order).
            Alternatively: None (no remap); str 'pairwise' (4 cams: 0–1 CW, 2–3 CCW);
            or comma-separated cw/ccw/none per camera.
        image_width, image_height: original image size in pixels before rotation.
            Integers apply to every camera that has a non-None center_rotate; or pass
            list/tuple of length ncameras for per-camera width/height. Required when
            any camera uses center_rotate.
        """
        self.path = path
        calib_path = calibration_folder if calibration_folder is not None else path
        self._calib_path = calib_path
        self.calibration = import_calibration(calib_path)
        self.ncameras = self.calibration.shape[0]
        self.store_xyz_for_plotting = store_xyz_for_plotting

        self._center_rotate = normalize_center_rotate_per_camera(
            center_rotate, self.ncameras
        )
        self._image_w_list, self._image_h_list = normalize_image_size_per_camera(
            image_width, image_height, self.ncameras, self._center_rotate
        )

        self._center_files = self._discover_center_files()

        # Filled by compute_rays(stream_to_disk=False) or when store_xyz_for_plotting
        self.centers = None
        self.XYZ = None
        self.xyz0 = None
        self.dd = None
        self.diameter = None
        self.intensity = None
        self.mass = None

        self._output_written = False

    def _discover_center_files(self):
        folder = self.path + Folders.CENTERS.value
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Centers folder not found: {folder}")
        files = sorted(f for f in os.listdir(folder) if f.endswith(".h5"))
        if len(files) != self.ncameras:
            raise ValueError(
                f"Expected {self.ncameras} Centers/*.h5 files, found {len(files)} in {folder}"
            )
        return [os.path.join(folder, f) for f in files]

    def load_centers(self):
        """Load all center frames into memory (legacy; avoid for large datasets)."""
        self.centers = np.empty(self.ncameras, dtype=object)
        for idx, fp in enumerate(self._center_files):
            self.centers[idx] = read_h5_centers(fp)

    def _process_frame_arrays(self, cam_idx, frame_data):
        """Return xyz0, dd, optional XYZ for plotting, and scalar arrays for HDF5."""
        rot = self._center_rotate[cam_idx]
        if rot:
            frame_data = _rotate_centers_xy_90(
                frame_data,
                self._image_w_list[cam_idx],
                self._image_h_list[cam_idx],
                rot,
            )
        return _compute_frame_arrays(
            self.calibration[cam_idx], frame_data, self.store_xyz_for_plotting
        )

    def _write_frame_group(self, grp, xyz0, dd, diameter, intensity, mass):
        _write_frame_group_h5(grp, xyz0, dd, diameter, intensity, mass)

    def _compute_rays_stream_to_disk(self, n_workers=1, flush_every=1, max_frames=None):
        """Read centers → compute → rays.h5. All frames (or first max_frames); optional parallel chunks."""
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")
        out_path = self.path + Filenames.RAYS.value
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        calib_path = self._calib_path

        if any(self._center_rotate):
            parts = [
                f"cam{i}:{self._center_rotate[i]}@{self._image_w_list[i]}x{self._image_h_list[i]}"
                for i in range(self.ncameras)
                if self._center_rotate[i]
            ]
            print(
                "[rays] Center (x,y) remap (90° as image rotation): " + "; ".join(parts),
                flush=True,
            )

        if n_workers > 1:
            n_src_per_cam = [
                count_h5_center_frames(self._center_files[c]) for c in range(self.ncameras)
            ]
            if len(set(n_src_per_cam)) != 1:
                raise RuntimeError(
                    "All cameras must have the same number of center frames for parallel "
                    f"ray compute; got per-camera counts {n_src_per_cam}"
                )
            n_src = n_src_per_cam[0]
            if n_src == 0:
                raise RuntimeError(
                    "No center frames found in Centers/*.h5; cannot compute rays.")
            n_eff = _effective_frame_count(n_src, max_frames)
            if max_frames is not None and n_eff < n_src:
                print(
                    f"[rays] Limiting parallel ray compute to first {n_eff} of {n_src} "
                    "frames per camera",
                    flush=True,
                )
            chunks_per_cam = _chunks_per_camera(n_workers, self.ncameras)
            tasks = []
            all_partial_paths = []
            for cam_idx in range(self.ncameras):
                n_parts = min(chunks_per_cam, n_eff)
                for start, end in _split_frame_ranges(n_eff, n_parts):
                    fd, ppath = tempfile.mkstemp(
                        suffix=f"_cam{cam_idx}_f{start}_{end}_rays.h5", dir=out_dir
                    )
                    os.close(fd)
                    all_partial_paths.append(ppath)
                    tasks.append(
                        (
                            calib_path,
                            cam_idx,
                            self._center_files[cam_idx],
                            start,
                            end,
                            ppath,
                            flush_every,
                            self._center_rotate[cam_idx],
                            self._image_w_list[cam_idx]
                            if self._image_w_list is not None
                            else 0,
                            self._image_h_list[cam_idx]
                            if self._image_h_list is not None
                            else 0,
                        )
                    )

            if not tasks:
                raise RuntimeError(
                    "No parallel ray tasks (empty frame ranges).")

            n_proc = min(n_workers, len(tasks))
            print(
                f"Streaming rays (parallel): {len(tasks)} chunk(s), pool size {n_proc}, "
                f"~{chunks_per_cam} chunk(s) per camera, h5 flush every {flush_every} frame(s)",
                flush=True,
            )
            try:
                _mp_ctx = multiprocessing.get_context(
                    "spawn" if sys.platform == "win32" else "fork"
                )
                with _mp_ctx.Pool(processes=n_proc) as pool:
                    results = pool.map(_stream_camera_chunk_worker, tasks)

                partials_per_camera = [[] for _ in range(self.ncameras)]
                for cam_idx, start, end, n_written, path in results:
                    if n_written != end - start:
                        raise RuntimeError(
                            f"Chunk incomplete for camera {cam_idx} [{start},{end}): "
                            f"expected {end - start} frames, got {n_written}"
                        )
                    partials_per_camera[cam_idx].append((start, path))
                for cam_idx in range(self.ncameras):
                    partials_per_camera[cam_idx].sort(key=lambda x: x[0])
                    partials_per_camera[cam_idx] = [p[1]
                                                    for p in partials_per_camera[cam_idx]]

                print(f"Merging chunk partials -> {out_path}", flush=True)
                _merge_ray_chunks(out_path, partials_per_camera, self.ncameras)
                print("Merge done.", flush=True)
            finally:
                for p in all_partial_paths:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
        else:
            print(
                f"Processing cameras sequentially (streaming write), "
                f"h5 flush every {flush_every} frame(s)",
                flush=True,
            )
            n_src0 = count_h5_center_frames(self._center_files[0])
            n_eff_seq = _effective_frame_count(n_src0, max_frames)
            if max_frames is not None and n_eff_seq < n_src0:
                print(
                    f"[rays] Limiting sequential ray compute to first {n_eff_seq} of "
                    f"{n_src0} frames per camera",
                    flush=True,
                )
            with h5py.File(out_path, "w") as fout:
                fout.attrs["ncameras"] = self.ncameras
                for cam_idx in range(self.ncameras):
                    fpath = self._center_files[cam_idx]
                    n_src = count_h5_center_frames(fpath)
                    n_eff = _effective_frame_count(n_src, max_frames)
                    step = _progress_interval(n_eff)
                    print(
                        f"Camera {cam_idx} (streaming): {n_eff} frames"
                        + (f" (of {n_src} in centers)" if n_eff < n_src else ""),
                        flush=True,
                    )
                    camgrp = fout.create_group(f"Camera {cam_idx}")
                    for frame_idx, frame_data in enumerate(iter_read_h5_centers(fpath)):
                        if frame_idx >= n_eff:
                            break
                        xyz0, dd, diameter, intensity, mass, xyz_plot = (
                            self._process_frame_arrays(cam_idx, frame_data)
                        )
                        grp = camgrp.create_group(f"frame{frame_idx:05d}")
                        self._write_frame_group(
                            grp, xyz0, dd, diameter, intensity, mass)
                        done = frame_idx + 1
                        if flush_every >= 1 and done % flush_every == 0:
                            fout.flush()
                        if (
                            done == 1
                            or done == n_eff
                            or done % step == 0
                        ):
                            pct = 100.0 * done / n_eff if n_eff else 100.0
                            print(
                                f"  Camera {cam_idx}: {done}/{n_eff} "
                                f"({pct:.1f}%)  frame {frame_idx}",
                                flush=True,
                            )
                        del xyz0, dd, diameter, intensity, mass, xyz_plot, frame_data
                    fout.flush()
                    print(
                        f"  Camera {cam_idx}: finished, {n_eff} frames written",
                        flush=True,
                    )

        self._output_written = True

    def _compute_rays_accumulate(self, max_frames=None):
        """Keep xyz0/dd in memory for write_rays() later; never loads all centers at once."""
        self.xyz0 = np.empty(self.ncameras, dtype=object)
        self.dd = np.empty(self.ncameras, dtype=object)
        self.diameter = np.empty(self.ncameras, dtype=object)
        self.intensity = np.empty(self.ncameras, dtype=object)
        self.mass = np.empty(self.ncameras, dtype=object)
        if self.store_xyz_for_plotting:
            self.XYZ = np.empty(self.ncameras, dtype=object)

        for cam_idx in range(self.ncameras):
            fpath = self._center_files[cam_idx]
            n_src = count_h5_center_frames(fpath)
            n_eff = _effective_frame_count(n_src, max_frames)
            step = _progress_interval(n_eff)
            print(
                f"Camera {cam_idx} (accumulate in RAM): {n_eff} frames"
                + (f" (of {n_src} in centers)" if n_eff < n_src else ""),
                flush=True,
            )
            xyz0_list, dd_list = [], []
            d_list, i_list, m_list = [], [], []
            xyz_list = [] if self.store_xyz_for_plotting else None

            n_kept = 0
            for frame_idx, frame_data in enumerate(iter_read_h5_centers(fpath)):
                if n_kept >= n_eff:
                    break
                xyz0, dd, diameter, intensity, mass, xyz_plot = self._process_frame_arrays(
                    cam_idx, frame_data
                )
                xyz0_list.append(xyz0)
                dd_list.append(dd)
                d_list.append(diameter)
                i_list.append(intensity)
                m_list.append(mass)
                if self.store_xyz_for_plotting:
                    xyz_list.append(xyz_plot)
                n_kept += 1
                if (
                    n_kept == 1
                    or n_kept == n_eff
                    or n_kept % step == 0
                ):
                    pct = 100.0 * n_kept / n_eff if n_eff else 100.0
                    print(
                        f"  Camera {cam_idx}: {n_kept}/{n_eff} "
                        f"({pct:.1f}%)  frame {frame_idx}",
                        flush=True,
                    )

            self.xyz0[cam_idx] = np.asarray(xyz0_list, dtype=object)
            self.dd[cam_idx] = np.asarray(dd_list, dtype=object)
            self.diameter[cam_idx] = np.asarray(d_list, dtype=object)
            self.intensity[cam_idx] = np.asarray(i_list, dtype=object)
            self.mass[cam_idx] = np.asarray(m_list, dtype=object)
            if self.store_xyz_for_plotting:
                self.XYZ[cam_idx] = np.asarray(xyz_list, dtype=object)
            print(
                f"  Camera {cam_idx}: finished accumulate, {n_kept} frames kept",
                flush=True,
            )

    def process_camera(self, cam_idx: int = 0, max_frames=None):
        """Process one camera using in-memory self.centers (call load_centers() first)."""
        if self.centers is None:
            raise RuntimeError(
                "load_centers() must be called before process_camera()")
        n_all = len(self.centers[cam_idx])
        n_frames = _effective_frame_count(n_all, max_frames)
        step = _progress_interval(n_frames)
        print(
            f"Camera {cam_idx} (in-memory centers): {n_frames} frames"
            + (f" (of {n_all})" if n_frames < n_all else ""),
            flush=True,
        )
        if self.store_xyz_for_plotting:
            self.XYZ[cam_idx] = np.empty(n_frames, dtype=object)
        self.xyz0[cam_idx] = np.empty(n_frames, dtype=object)
        self.dd[cam_idx] = np.empty(n_frames, dtype=object)
        self.diameter[cam_idx] = np.empty(n_frames, dtype=object)
        self.intensity[cam_idx] = np.empty(n_frames, dtype=object)
        self.mass[cam_idx] = np.empty(n_frames, dtype=object)

        for frame_idx, frame_data in enumerate(self.centers[cam_idx][:n_frames]):
            done = frame_idx + 1
            if done == 1 or done == n_frames or done % step == 0:
                pct = 100.0 * done / n_frames if n_frames else 100.0
                print(
                    f"  Camera {cam_idx}: frame {done}/{n_frames} ({pct:.1f}%)",
                    flush=True,
                )
            xyz0, dd, diameter, intensity, mass, xyz_plot = (
                self._process_frame_arrays(cam_idx, frame_data)
            )
            self.xyz0[cam_idx][frame_idx] = xyz0
            self.dd[cam_idx][frame_idx] = dd
            self.diameter[cam_idx][frame_idx] = diameter
            self.intensity[cam_idx][frame_idx] = intensity
            self.mass[cam_idx][frame_idx] = mass
            if self.store_xyz_for_plotting:
                self.XYZ[cam_idx][frame_idx] = xyz_plot

    def compute_rays(self, stream_to_disk=True, n_workers=1, flush_every=1, max_frames=None):
        """
        Compute ray origins and directions from centers (all frames).

        stream_to_disk=True (default): write rays.h5 in a single streaming pass
        (minimal RAM — does not load all centers or store full XYZ multi-plane arrays).

        flush_every: after this many frames are written per camera, flush the HDF5 file
        to push OS buffers to disk (default 1 = every frame). Larger values reduce
        flush overhead; 1 maximizes how often data leaves process memory buffers.

        n_workers: use up to this many worker processes. Frame ranges are split into about
        ceil(n_workers/ncameras) chunks per camera (each chunk runs in parallel), then
        partial HDF5 files are merged. With one camera, all workers can serve that camera's
        chunks. Ignored when stream_to_disk=False (use 1).

        max_frames: if set, only the first N frames per camera are processed (rays.h5
        contains frame00000 … frame{N-1:05d}).

        stream_to_disk=False: fill xyz0/dd in memory for a later write_rays(); still
        streams center files frame-by-frame unless you called load_centers().
        """
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")

        if stream_to_disk and self.store_xyz_for_plotting:
            raise ValueError(
                "stream_to_disk=True is incompatible with store_xyz_for_plotting=True; "
                "use stream_to_disk=False and then write_rays(), or disable plotting storage."
            )
        if stream_to_disk:
            self._compute_rays_stream_to_disk(
                n_workers, flush_every, max_frames=max_frames)
        else:
            if n_workers > 1:
                raise ValueError("n_workers>1 requires stream_to_disk=True")
            if self.centers is not None:
                if self.xyz0 is None:
                    self.xyz0 = np.empty(self.ncameras, dtype=object)
                    self.dd = np.empty(self.ncameras, dtype=object)
                    self.diameter = np.empty(self.ncameras, dtype=object)
                    self.intensity = np.empty(self.ncameras, dtype=object)
                    self.mass = np.empty(self.ncameras, dtype=object)
                    if self.store_xyz_for_plotting:
                        self.XYZ = np.empty(self.ncameras, dtype=object)
                for cam_idx in range(self.ncameras):
                    self.process_camera(cam_idx=cam_idx, max_frames=max_frames)
            else:
                self._compute_rays_accumulate(max_frames=max_frames)

    def find_rays(calibration, x_px, y_px):
        nplanes = calibration["n_planes"]

    def plot_rays(self, nrays: int = 10, frame: int = 0, cameras: list[int] = None):
        if cameras is None:
            cameras = [0]
        cameras = list(cameras)
        if not self.store_xyz_for_plotting or self.XYZ is None:
            raise RuntimeError(
                "plot_rays() requires store_xyz_for_plotting=True at construction "
                "and compute_rays(stream_to_disk=False)."
            )
        plot_rays(self.XYZ, xyz0=self.xyz0,
                  dd=self.dd[:], nrays=nrays, cameras=cameras, frame=frame)

    def write_rays(self):
        """Write rays.h5 from memory. No-op if compute_rays(stream_to_disk=True) already wrote the file."""
        if self._output_written:
            return
        if self.xyz0 is None:
            raise RuntimeError(
                "No ray data to write; call compute_rays() first.")

        out_path = self.path + Filenames.RAYS.value
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with h5py.File(out_path, "w") as fout:
            fout.attrs["ncameras"] = self.ncameras
            for cam in range(self.ncameras):
                camgrp = fout.create_group(f"Camera {cam}")
                n_frames = len(self.xyz0[cam])
                for frame_idx in range(n_frames):
                    grp = camgrp.create_group(f"frame{frame_idx:05d}")
                    self._write_frame_group(
                        grp,
                        self.xyz0[cam][frame_idx],
                        self.dd[cam][frame_idx],
                        self.diameter[cam][frame_idx],
                        self.intensity[cam][frame_idx],
                        self.mass[cam][frame_idx],
                    )
        self._output_written = True
