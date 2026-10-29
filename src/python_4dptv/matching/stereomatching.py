import subprocess
import h5py
import numpy as np
import os

from ..utils.structure import Filenames

from ..plotting.plot_particles import plot_particles_xyze

try:
    from ..io.h5_metadata import write_h5_run_metadata
except ImportError:
    write_h5_run_metadata = None


_PARTICLE_PROP_KEYS = ("diameter", "intensity", "mass")


def enrich_stereo_output_with_particle_props(
    path,
    *,
    rays_path=None,
    write_particle_props=True,
    write_diameter=True,
    write_intensity=True,
    write_mass=True,
):
    """
    Add diameter, intensity, and/or mass to rays_out_cpp.h5 using camrayids
    to look up values from rays.h5. Writes values from all contributing
    cameras per match: shape (n_matches, maxcams), NaN where a camera did not contribute.

    Set ``write_particle_props=False`` to skip enrichment entirely. Individual
    ``write_diameter`` / ``write_intensity`` / ``write_mass`` flags select which
    datasets are written (defaults: all on). Disabled props are removed if present.
    """
    if not write_particle_props:
        return

    props_to_write = [
        key
        for key, enabled in (
            ("diameter", write_diameter),
            ("intensity", write_intensity),
            ("mass", write_mass),
        )
        if enabled
    ]

    rays_path = rays_path if rays_path is not None else path + Filenames.RAYS.value
    stm_path = path + Filenames.STM.value
    if not os.path.isfile(rays_path) or not os.path.isfile(stm_path):
        return

    if not props_to_write:
        # Master on but all datasets off: strip any existing prop datasets.
        try:
            with h5py.File(stm_path, "r+") as s:
                for fk in s.keys():
                    if not fk.startswith("frame"):
                        continue
                    grp = s[fk]
                    for key in _PARTICLE_PROP_KEYS:
                        if key in grp:
                            del grp[key]
        except OSError:
            pass
        return

    with h5py.File(rays_path, "r") as r:
        cam_names = sorted(
            [k for k in r.keys() if k.startswith("Camera")],
            key=lambda k: int(k.split()[-1]) if k.split()[-1].isdigit() else 0,
        )
        if not cam_names:
            return
        frame_keys = sorted(
            r[cam_names[0]].keys(),
            key=lambda k: int(k.replace("frame", "")) if k.replace("frame", "").isdigit() else 0,
        )
        has_props = "diameter" in r[cam_names[0]][frame_keys[0]]
        if not has_props:
            return

        rays_props = {}
        for cam in cam_names:
            rays_props[cam] = {}
            for fk in frame_keys:
                g = r[cam][fk]
                rays_props[cam][fk] = {
                    key: np.asarray(g[key][()]) for key in props_to_write
                }

    try:
        stm_file = h5py.File(stm_path, "r+")
    except OSError as e:
        if "truncated" in str(e).lower() or "eof" in str(e).lower():
            raise OSError(
                "STM output file is truncated or corrupted (incomplete write). "
                "The C++ STM process may have been interrupted, run out of memory, or crashed. "
                "Try running with fewer threads or check STM output above."
            ) from e
        raise

    with stm_file as s:
        stm_frame_keys = sorted(
            [k for k in s.keys() if k.startswith("frame")],
            key=lambda k: int(k.replace("frame", "")) if k.replace("frame", "").isdigit() else 0,
        )
        for fk in stm_frame_keys:
            if fk not in rays_props[cam_names[0]]:
                continue
            grp = s[fk]
            xyze = grp["xyze"]
            n_matches = xyze.shape[1]
            # camrayids: row 0 = camera 0, row 1 = ray id; row 2 = camera 1, row 3 = ray id; ...
            camrayids = grp["camrayids"][()]
            maxcams = camrayids.shape[0] // 2

            arrays = {
                key: np.full((n_matches, maxcams), np.nan, dtype=np.float64)
                for key in props_to_write
            }

            for i in range(n_matches):
                for j in range(maxcams):
                    camid = int(camrayids[2 * j, i])
                    rayid = int(camrayids[2 * j + 1, i])
                    if camid < 0 or rayid < 0:
                        continue
                    cam_name = f"Camera {camid}"
                    if cam_name not in rays_props or fk not in rays_props[cam_name]:
                        continue
                    d = rays_props[cam_name][fk]
                    ref = next(iter(d.values()))
                    if rayid < len(ref):
                        for key in props_to_write:
                            arrays[key][i, j] = d[key][rayid]

            for key in _PARTICLE_PROP_KEYS:
                if key in grp:
                    del grp[key]
            for key in props_to_write:
                grp.create_dataset(key, data=arrays[key])


class StereoMatching():
    def __init__(
        self,
        path,
        mincameras,
        maxdistance,
        multiplematchesperraydistance,
        maxmatchesperray,
        nvoxels: list[int],
        boundingbox: list[float],
        *,
        rays_input_path=None,
        run_metadata=None,
    ):
        self.path = path
        self.rays_input_path = rays_input_path if rays_input_path is not None else path
        self.run_metadata = run_metadata
        self.frames = self.read_number_of_frames(self.rays_input_path + Filenames.RAYS.value)
        self.filename = self.rays_input_path + Filenames.RAYS.value
        self.output = self.path
        self.mincameras = mincameras
        self.maxdistance = maxdistance
        self.multiplematchesperraydistance = multiplematchesperraydistance
        self.maxmatchesperray = maxmatchesperray
        self.nx = nvoxels[0]
        self.ny = nvoxels[1]
        self.nz = nvoxels[2]
        self.minX = boundingbox[0]
        self.maxX = boundingbox[1]
        self.minY = boundingbox[2]
        self.maxY = boundingbox[3]
        self.minZ = boundingbox[4]
        self.maxZ = boundingbox[5]
        self.boundingbox = boundingbox

    def read_number_of_frames(self, filename):
        with h5py.File(filename, 'r') as f:
            groups = list(f.keys())
            return len(f[groups[0]].keys())

    def run_stereomatching(
        self,
        nthreads=8,
        nframes=None,
        timing=False,
        spatial_boxes=0,
        spatial_overlap_cells=1,
        frame_parallelism=0,
        cylinder=False,
        cylinder_max_shells=32,
        cylinder_voxels=None,
        cylinder_rank=None,
        cylinder_h5_aux=None,
        write_particle_props=True,
        write_diameter=True,
        write_intensity=True,
        write_mass=True,
        resume=False,
    ):
        """Run the C++ STM binary.

        ``spatial_boxes`` sets ``--spatial-boxes`` (sub-volume count; 0 = single full volume).
        ``spatial_overlap_cells`` sets ``--spatial-overlap`` (halo in global voxels per face; default 1).
        Use ``OMP_NUM_THREADS`` >= ``spatial_boxes`` if you want all boxes to run in parallel.

        ``frame_parallelism`` sets ``--frame-parallelism`` (max frames processed at once; 0 = use
        ``OMP_NUM_THREADS`` for the outer frame loop, same as not passing the flag).

        ``resume=True`` passes ``--resume``: keep an existing ``rays_out_cpp.h5`` and skip frames that
        already have complete match groups (``xyze`` + ``camrayids``).

        Cylinder mode (uses ``diameter`` from ``rays.h5`` when ``cylinder=True``): passes
        ``--cylinder-voxels``, ``--cylinder-rank``, and ``--cylinder-h5-aux`` unless overridden by
        the individual ``cylinder_*`` arguments. ``cylinder_max_shells`` maps to ``--cylinder-max-shells``.

        After a successful STM run, particle properties from ``rays.h5`` are optionally written
        into ``rays_out_cpp.h5``. Use ``write_particle_props=False`` to skip that step, or
        ``write_diameter`` / ``write_intensity`` / ``write_mass`` to select datasets (all default on).
        """
        if nframes is not None:
            self.frames = nframes

        self._last_stm_run = {
            "nthreads": nthreads,
            "timing": timing,
            "spatial_boxes": spatial_boxes,
            "spatial_overlap_cells": spatial_overlap_cells,
            "frame_parallelism": frame_parallelism,
            "max_frames": nframes,
            "write_particle_props": write_particle_props,
            "write_diameter": write_diameter,
            "write_intensity": write_intensity,
            "write_mass": write_mass,
            "resume": resume,
        }

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = f'{nthreads}'
        # argv list (do not str.split): paths/spaces safe; STM --bb expects six floats (CLI uses vector<double>)
        cmd = [
            './STMCpp/STM',
            '-i', self.filename,
            '-o', self.output,
            '-f', str(self.frames),
            '-c', str(self.mincameras),
            '-d', str(self.maxdistance),
            '-s', str(self.multiplematchesperraydistance),
            '-m', str(self.maxmatchesperray),
            '-x', str(self.nx),
            '-y', str(self.ny),
            '-z', str(self.nz),
            '-b',
            str(self.minX), str(self.maxX), str(self.minY), str(self.maxY), str(self.minZ), str(self.maxZ),
            '--hdf5',
        ]
        if timing:
            cmd.append('--timing')
        if resume:
            cmd.append('--resume')
        if spatial_boxes and int(spatial_boxes) > 0:
            cmd.extend(['--spatial-boxes', str(int(spatial_boxes))])
            cmd.extend(['--spatial-overlap', str(int(spatial_overlap_cells))])
        if frame_parallelism and int(frame_parallelism) > 0:
            cmd.extend(['--frame-parallelism', str(int(frame_parallelism))])

        use_vox = cylinder_voxels if cylinder_voxels is not None else bool(cylinder)
        use_rank = cylinder_rank if cylinder_rank is not None else bool(cylinder)
        use_h5 = cylinder_h5_aux if cylinder_h5_aux is not None else bool(cylinder)
        if use_vox:
            cmd.append('--cylinder-voxels')
        if use_rank:
            cmd.append('--cylinder-rank')
        if use_h5:
            cmd.append('--cylinder-h5-aux')
        if use_vox or use_rank or use_h5:
            cmd.extend(['--cylinder-max-shells', str(int(cylinder_max_shells))])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,           # automatically decode bytes to string
            bufsize=1,            # line-buffered
            env=env
        )

        # Continuously read lines as they are printed
        for line in proc.stdout:
            print(line, end="")  # already has newline

        # Wait for the process to finish
        proc.wait()
        if proc.returncode != 0:
            rc = proc.returncode
            hint = "Check the output above for errors."
            if rc == -9 or rc == 137:  # SIGKILL (137 = 128+9)
                hint = (
                    "Process was SIGKILL'd (-9): often the OOM killer (out of RAM). "
                    "Try fewer OMP threads (--threads / N_THREADS), a smaller voxel grid (nvoxels), "
                    "or run on a machine with more memory; see dmesg for 'Killed process' / 'oom-kill'."
                )
            raise RuntimeError(
                f"Stereo matching (STM) process exited with code {rc}. {hint}"
            )

        # Carry diameter, intensity, mass from centers through to STM output (optional)
        enrich_stereo_output_with_particle_props(
            self.path,
            rays_path=self.rays_input_path + Filenames.RAYS.value,
            write_particle_props=write_particle_props,
            write_diameter=write_diameter,
            write_intensity=write_intensity,
            write_mass=write_mass,
        )
        self._write_stm_metadata(nframes)

    def _write_stm_metadata(self, nframes: int) -> None:
        if write_h5_run_metadata is None:
            return
        stm_path = self.path + Filenames.STM.value
        if not os.path.isfile(stm_path):
            return
        meta = dict(self.run_metadata or {})
        meta.setdefault("job_id", "stereo_matching")
        params = dict(meta.get("parameters") or {})
        params.update({
            "min_cameras": self.mincameras,
            "max_distance": self.maxdistance,
            "multiple_matches_per_ray_distance": self.multiplematchesperraydistance,
            "max_matches_per_ray": self.maxmatchesperray,
            "nvoxels": [self.nx, self.ny, self.nz],
            "bounding_box": self.boundingbox,
            "frames_requested": nframes,
            "input_rays_path": os.path.abspath(self.filename),
            "output_dir": os.path.abspath(self.path),
        })
        if getattr(self, "_last_stm_run", None):
            params.update(self._last_stm_run)
        meta["parameters"] = params
        write_h5_run_metadata(stm_path, meta)

    def plot_matches(self, external_path=None):
        if external_path is None:
            external_path = self.path
        with h5py.File(self.path + Filenames.STM.value, "r") as f:
            for frame_idx, frame in enumerate(f.values()):
                XYZe = frame['xyze']
                plot_particles_xyze(XYZe, self.boundingbox,
                                    external_path, frame_idx)

    def plot_matches_frame(self, eval_frame):
        with h5py.File(self.path + Filenames.STM.value, "r") as f:
            for frame_idx, frame in enumerate(f.values()):
                if eval_frame != frame_idx:
                    continue
                XYZe = frame['xyze']
                plot_particles_xyze(XYZe, self.boundingbox,
                                    self.path, frame_idx)
