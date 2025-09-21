import numpy as np
import os

from ptv_calib.io.import_calibration import import_calibration

from ..io.read_h5 import read_h5_centers


class Rays():
    def __init__(self, path):
        self.path = path
        self.calibration = import_calibration(path)
        self.centers = self.load_centers()

    def load_centers(self):
        self.centers = np.empty(self.calibration.shape[0], dtype=object)
        files = sorted(os.listdir(self.path))
        for idx, file in enumerate(files):
            self.centers[idx] = read_h5_centers(os.path.join(self.path, file))

    def process_camera(self, ncamera: int = 0):
        return

    def compute_rays(self):
        return

    def find_rays(calibration, x_px, y_px):
        nplanes = calibration['n_planes']
