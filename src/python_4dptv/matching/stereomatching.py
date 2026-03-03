import subprocess
import h5py
import os

from ..utils.structure import Filenames

from ..plotting.plot_particles import plot_particles_xyze


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
