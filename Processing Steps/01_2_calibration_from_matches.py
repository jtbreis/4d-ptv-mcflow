# %%
"""
Run calibration from precomputed match files in two sets (set A and set B).
No preprocessing or matching steps; only reads matches, runs calibration, writes
the combined matches and calib, then sets the Matches folder to read-only.
"""
import os
import stat
import numpy as np
from ptv_calib.calibrate import Calibration
from ptv_calib.io.output import read_h5_matches
from ptv_calib.utils.structure import Folders

os.chdir('/workspaces/4d-ptv-mcflow')


def set_folder_readonly(path: str) -> None:
    """Set directory and all its contents to read-only (555)."""
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in filenames:
            os.chmod(os.path.join(dirpath, name), stat.S_IRUSR | stat.S_IRGRP |
                     stat.S_IROTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.chmod(dirpath, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH |
                 stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# %%
# Match file sets: each entry is (path, n_cameras) — use first n_cameras from that file.
# Number of cameras across sets must equal len(cameras).
# path, use first 3 cameras
set_a = ('/workspaces/4d-ptv-mcflow/data/julian/PTV_above/Calibration/Matches/matches_cam1to2.h5', 2)
# path, use first 1 camera
set_b = ('/workspaces/4d-ptv-mcflow/data/julian/PTV_above/Calibration/Matches/matches_cam3to4.h5', 2)
match_file_sets = [set_a, set_b]

output_path = '/workspaces/4d-ptv-mcflow/data/julian/PTV_above/Calibration'
calibration_grid_path = 'PTVcalib/calibration_targets/TSI_5mm_backlight_nX39_nY39.csv'

cameras = [1, 2, 3, 4]
calibration_method = '4d-ptv'
grid_spacing = 5.0  # mm
z_min = -25  # mm
z_max = 25  # mm
target_point_diameter = 20

# --- Which layers to use for calibration (optional) ---
n_planes_to_use = None  # e.g. 21 to use that many evenly spaced planes
# e.g. [0, 5, 10, ...] to use exactly these layer indices (overrides n_planes_to_use)
layers_to_use = None
# If both are None, all layers from the match files are used.

# Set Matches folder to read-only after writing (True/False)
set_matches_folder_readonly = True

# %%
# Read and merge matches from all sets
n_cameras_total = sum(n for _, n in match_file_sets)
assert n_cameras_total == len(cameras), (
    f"Total cameras from sets ({n_cameras_total}) must equal len(cameras) ({len(cameras)})"
)

n_planes_total = None
parts = []  # list of (n_cams, data array)
for path, n_cameras in match_file_sets:
    data = read_h5_matches(path)
    assert data.shape[0] >= n_cameras, (
        f"File {path} has {data.shape[0]} cameras, need at least {n_cameras}"
    )
    if n_planes_total is None:
        n_planes_total = data.shape[1]
    else:
        assert data.shape[1] == n_planes_total, (
            f"Plane count mismatch: {path} has {data.shape[1]} planes, expected {n_planes_total}"
        )
    parts.append((n_cameras, data[:n_cameras, :]))

matched_points = np.empty((n_cameras_total, n_planes_total), dtype=object)
offset = 0
for n_cams, part in parts:
    matched_points[offset: offset + n_cams, :] = part
    offset += n_cams

# Select layer indices to use for calibration
full_z_planes = np.linspace(z_min, z_max, n_planes_total)
if layers_to_use is not None:
    layer_indices = np.asarray(layers_to_use, dtype=int)
elif n_planes_to_use is not None:
    layer_indices = np.linspace(
        0, n_planes_total - 1, n_planes_to_use, dtype=int)
else:
    layer_indices = np.arange(n_planes_total)

matched_points = matched_points[:, layer_indices]
z_planes = full_z_planes[layer_indices]
n_planes = len(layer_indices)
print(
    f"Using {n_planes} planes for calibration (layer indices: {layer_indices.tolist()})")

# %%
# Create calibration (same config as 01_calibration, no folder_path needed for calibration step)
calibration = Calibration(
    cameras=cameras,
    folder_path='',  # unused when only running calibration from matches
    output_path=output_path,
    calibration_grid_path=calibration_grid_path,
    grid_spacing=grid_spacing,
    target_point_diameter=target_point_diameter,
    z_min=z_min,
    z_max=z_max,
    n_planes=n_planes,
    calibration_method=calibration_method,
)
calibration.set_custom_zplanes(z_planes.tolist())
calibration.matched_points = matched_points

# %%
calibration.perform_calibration()

# %%
calibration.write_calibration()
calibration.write_matches()
calibration.write_calibration_test_files()

# %%
if set_matches_folder_readonly:
    matches_folder = output_path.rstrip('/') + Folders.MATCHES.value
    set_folder_readonly(matches_folder)
    print(f"Set read-only: {matches_folder}")

# %%
