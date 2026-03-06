import h5py
import numpy as np


def read_h5_centers(filename):
    """Read center-finding HDF5. Returns list of dicts per frame with 'x', 'y'
    and optionally 'mass', 'diameter', 'intensity' for stereomatching.
    Uses trackpy names from the file: size -> diameter (2*size), signal -> intensity, mass."""
    with h5py.File(filename, "r") as f:
        frame_keys = sorted(
            (k for k in f.keys() if k.startswith("frame")),
            key=lambda k: int(k.replace("frame", "")) if k.replace("frame", "").isdigit() else 0,
        )
        frames = []
        for frame_key in frame_keys:
            grp = f[frame_key]
            x = np.asarray(grp["x"][()])
            y = np.asarray(grp["y"][()])
            out = {"x": x, "y": y}
            if "mass" in grp:
                out["mass"] = np.asarray(grp["mass"][()])
            if "diameter" in grp:
                out["diameter"] = np.asarray(grp["diameter"][()])
            elif "size" in grp:
                out["diameter"] = 2.0 * np.asarray(grp["size"][()])
            if "intensity" in grp:
                out["intensity"] = np.asarray(grp["intensity"][()])
            elif "signal" in grp:
                out["intensity"] = np.asarray(grp["signal"][()])
            frames.append(out)
        return frames
