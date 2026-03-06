import subprocess
import h5py
import numpy as np
import os

from ..utils.structure import Filenames

from ..plotting.plot_particles import plot_particles_xyze


def enrich_stereo_output_with_particle_props(path):
    """
    Add diameter, intensity, and mass to rays_out_cpp.h5 using camrayids
    to look up values from rays.h5. Writes values from all contributing
    cameras per match: shape (n_matches, maxcams), NaN where a camera did not contribute.
    """
    rays_path = path + Filenames.RAYS.value
    stm_path = path + Filenames.STM.value
    if not os.path.isfile(rays_path) or not os.path.isfile(stm_path):
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
                    "diameter": np.asarray(g["diameter"][()]),
                    "intensity": np.asarray(g["intensity"][()]),
                    "mass": np.asarray(g["mass"][()]),
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
            camrayids = grp["camrayids"][()]
            maxcams = camrayids.shape[0] // 2

            diameter = np.full((n_matches, maxcams), np.nan, dtype=np.float64)
            intensity = np.full((n_matches, maxcams), np.nan, dtype=np.float64)
            mass = np.full((n_matches, maxcams), np.nan, dtype=np.float64)

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
                    if rayid < len(d["diameter"]):
                        diameter[i, j] = d["diameter"][rayid]
                        intensity[i, j] = d["intensity"][rayid]
                        mass[i, j] = d["mass"][rayid]

            if "diameter" in grp:
                del grp["diameter"]
            if "intensity" in grp:
                del grp["intensity"]
            if "mass" in grp:
                del grp["mass"]
            grp.create_dataset("diameter", data=diameter)
            grp.create_dataset("intensity", data=intensity)
            grp.create_dataset("mass", data=mass)


class StereoMatching():
    def __init__(self, path, mincameras, maxdistance, multiplematchesperraydistance, maxmatchesperray, nvoxels: list[int], boundingbox: list[float]):
        self.path = path
        self.frames = self.read_number_of_frames(path + Filenames.RAYS.value)
        self.filename = self.path + Filenames.RAYS.value
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

    def run_stereomatching(self, nthreads=8, nframes=None):
        if nframes is not None:
            self.frames = nframes

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = f'{nthreads}'
        run_command = f'./STMCpp/STM -i {self.filename} -o {self.output} -f {self.frames} -c {self.mincameras} -d {self.maxdistance} -s {self.multiplematchesperraydistance} -m {self.maxmatchesperray} -x {self.nx} -y {self.ny} -z {self.nz} -b {self.minX} {self.maxX} {self.minY} {self.maxY} {self.minZ} {self.maxZ} --hdf5'

        # Launch the process
        proc = subprocess.Popen(
            # replace with your command
            run_command.split(),
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
            raise RuntimeError(
                f"Stereo matching (STM) process exited with code {proc.returncode}. "
                "Check the output above for errors."
            )

        # Carry diameter, intensity, mass from centers through to STM output
        enrich_stereo_output_with_particle_props(self.path)

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
