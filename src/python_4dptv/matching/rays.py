import numpy as np
import os
import matplotlib.pyplot as plt

from ptv_calib.io.import_calibration import import_calibration

from ..io.read_h5 import read_h5_centers
from ..io.write_h5 import write_h5_multiple_camera
from .fit_line import fit3dline
from ..plotting.ray_testing import plot_rays
from ..utils.structure import Filenames, Folders


class Rays():
    def __init__(self, path, calibration_folder=None):
        """
        path: folder containing Centers/ and where rays will be read/written.
        calibration_folder: if set, load calib.h5 from this folder (e.g. test
            Calibration_Before calibration on points from Calibration_After).
        """
        self.path = path
        calib_path = calibration_folder if calibration_folder is not None else path
        self.calibration = import_calibration(calib_path)
        self.ncameras = self.calibration.shape[0]
        self.centers = np.empty(self.ncameras, dtype=object)
        self.XYZ = np.empty(self.ncameras, dtype=object)
        self.xyz0 = np.empty(self.ncameras, dtype=object)
        self.dd = np.empty(self.ncameras, dtype=object)

        self.load_centers()

    def load_centers(self):
        folder = self.path + Folders.CENTERS.value
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Centers folder not found: {folder}")
        files = sorted(f for f in os.listdir(folder) if f.endswith(".h5"))
        for idx, file in enumerate(files):
            self.centers[idx] = read_h5_centers(
                os.path.join(folder, file))

    def process_camera(self, cam_idx: int = 0):
        self.XYZ[cam_idx] = np.empty(
            self.centers[cam_idx].shape[0], dtype=object)
        self.xyz0[cam_idx] = np.empty(
            self.centers[cam_idx].shape[0], dtype=object)
        self.dd[cam_idx] = np.empty(
            self.centers[cam_idx].shape[0], dtype=object)
        for frame_idx, frame_xy in enumerate(self.centers[cam_idx]):
            print(f'Frame {frame_idx} out of {len(self.centers[cam_idx])}')
            self.XYZ[cam_idx][frame_idx] = self.calibration[cam_idx].transform_to_real_world(
                frame_xy)
            self.xyz0[cam_idx][frame_idx], self.dd[cam_idx][frame_idx] = fit3dline(
                self.XYZ[cam_idx][frame_idx])
        return

    def compute_rays(self):
        for cam_idx in range(self.ncameras):
            print(f'Processing Camera {cam_idx}')
            self.process_camera(cam_idx=cam_idx)

    def find_rays(calibration, x_px, y_px):
        nplanes = calibration['n_planes']

    def plot_rays(self, nrays: int = 10, frame: int = 0, cameras: list[int] = [0]):
        plot_rays(self.XYZ, xyz0=self.xyz0, dd=self.dd[:],
                  nrays=nrays, cameras=cameras, frame=frame)

    def write_rays(self):
        data = {}
        metadata = {'ncameras': self.ncameras}
        for cam in range(self.ncameras):
            data_cam = {}
            for frame in range(self.xyz0[cam].shape[0]):
                data_cam[frame] = {
                    'xyz0': self.xyz0[cam][frame],
                    'dd': self.dd[cam][frame]
                }
            data[f'Camera {cam}'] = data_cam
        write_h5_multiple_camera(os.path.join(
            self.path + Filenames.RAYS.value), data, metadata)
