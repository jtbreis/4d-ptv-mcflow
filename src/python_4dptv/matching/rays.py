import numpy as np
import os
import matplotlib.pyplot as plt

from ptv_calib.io.import_calibration import import_calibration

from ..io.read_h5 import read_h5_centers
from .fit_line import fit3dline
from ..plotting.ray_testing import plot_rays


class Rays():
    def __init__(self, path):
        self.path = path
        self.calibration = import_calibration(path)
        self.ncameras = self.calibration.shape[0]
        self.centers = np.empty(self.ncameras, dtype=object)
        self.XYZ = np.empty(self.ncameras, dtype=object)
        self.xyz0 = np.empty(self.ncameras, dtype=object)
        self.dd = np.empty(self.ncameras, dtype=object)

        self.load_centers()

    def load_centers(self):
        files = sorted(os.listdir(self.path))
        for idx, file in enumerate(files):
            self.centers[idx] = read_h5_centers(os.path.join(self.path, file))

    def process_camera(self, cam_idx: int = 0):
        for frame_xy in self.centers[cam_idx]:
            self.XYZ[cam_idx] = self.calibration[cam_idx].transform_to_real_world(
                frame_xy)
            self.xyz0[cam_idx], self.dd[cam_idx] = fit3dline(
                self.XYZ[cam_idx])
        return

    def compute_rays(self):
        for cam_idx in range(self.ncameras):
            self.process_camera(cam_idx=cam_idx)

    def find_rays(calibration, x_px, y_px):
        nplanes = calibration['n_planes']

    def plot_rays(self, nrays: int = 10, cameras: list[int] = [0]):
        plot_rays(self.XYZ, xyz0=self.xyz0, dd=self.dd,
                  nrays=nrays, cameras=cameras)
