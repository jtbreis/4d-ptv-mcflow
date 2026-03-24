import h5py
import numpy as np


def _sorted_frame_keys(f):
    return sorted(
        (k for k in f.keys() if k.startswith("frame")),
        key=lambda k: int(k.replace("frame", "")) if k.replace(
            "frame", "").isdigit() else 0,
    )


def read_center_frame_from_group(grp):
    """Read one center-finding frame from an HDF5 group (frame#####)."""
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
    return out


def count_h5_center_frames(filename):
    """Count frame##### groups without loading particle data (fast progress totals)."""
    with h5py.File(filename, "r") as f:
        return sum(1 for k in f.keys() if k.startswith("frame"))


def iter_read_h5_centers(filename):
    """Yield center dicts one frame at a time (memory-efficient for large movies)."""
    with h5py.File(filename, "r") as f:
        for frame_key in _sorted_frame_keys(f):
            yield read_center_frame_from_group(f[frame_key])


def read_h5_centers(filename):
    """Read center-finding HDF5. Returns list of dicts per frame with 'x', 'y'
    and optionally 'mass', 'diameter', 'intensity' for stereomatching.
    Uses trackpy names from the file: size -> diameter (2*size), signal -> intensity, mass."""
    return list(iter_read_h5_centers(filename))
