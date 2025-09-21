import h5py
import numpy as np


def read_h5_centers(filename):
    with h5py.File(filename, "r") as f:
        frames = np.empty(len(f.keys()), dtype=object)
        for frame_idx, frame in enumerate(f.keys()):
            grp = f[frame]

            x = grp['x'][()]
            y = grp['y'][()]
            frames[frame_idx] = np.array([x, y])
        return frames
