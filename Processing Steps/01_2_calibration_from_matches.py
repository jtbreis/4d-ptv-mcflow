# %%
"""
Run calibration from precomputed match files (cameras 1-3 in one file, camera 4 in another).
No preprocessing or matching steps; only reads matches, runs calibration, and saves calib.h5.
"""
import os
import numpy as np
from ptv_calib.calibrate import Calibration
from ptv_calib.io.output import read_h5_matches

os.chdir('/workspaces/4d-ptv-mcflow')

# %%
# Paths to the two match files
matches_cam1to3_path = 'data/julian/PTV_below/Calibration/Matches/matches_cam1to3.h5'
matches_cam4_path = 'data/julian/PTV_below/Calibration/Matches/matches_cam4.h5'

output_path = 'data/julian/PTV_below'
calibration_grid_path = 'PTVcalib/calibration_targets/TSI_5mm_backlight_nX39_nY39.csv'

cameras = [1, 2, 3, 4]
calibration_method = '4d-ptv'
grid_spacing = 5.0  # mm
z_min = -25  # mm
z_max = 25  # mm
target_point_diameter = 20

# --- Which layers to use for calibration (optional) ---
# Option A: use a specific number of planes, evenly spaced over all layers
n_planes_to_use = None  # e.g. 21 to use 21 evenly spaced planes

# Option B: explicitly list layer indices to use (overrides n_planes_to_use if set)
layers_to_use = None  # e.g. [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] or list(range(0, 51, 5))

# If both are None, all layers from the match files are used.

# %%
# Read matches from both files.
# File 1 may contain 4 cameras; we use only cameras 1–3 (first 3 rows).
# File 2 contains camera 4.
matches_cam1to3_raw = read_h5_matches(matches_cam1to3_path)  # shape (n_cams, n_planes), n_cams >= 3
matches_cam4 = read_h5_matches(matches_cam4_path)            # shape (1, n_planes)

matches_cam1to3 = matches_cam1to3_raw[:3, :]  # use only cameras 1–3
n_planes_1 = matches_cam1to3.shape[1]
n_planes_2 = matches_cam4.shape[1]
assert matches_cam1to3_raw.shape[0] >= 3, "matches_cam1to3 must contain at least 3 cameras"
assert matches_cam4.shape[0] == 1, "Expected 1 camera in matches_cam4"
assert n_planes_1 == n_planes_2, (
    f"Plane count mismatch: cam1-3 has {n_planes_1} planes, cam4 has {n_planes_2}"
)
n_planes_total = n_planes_1

# Merge into (4, n_planes_total)
matched_points = np.empty((4, n_planes_total), dtype=object)
matched_points[0:3, :] = matches_cam1to3
matched_points[3:4, :] = matches_cam4

# Select layer indices to use for calibration
full_z_planes = np.linspace(z_min, z_max, n_planes_total)
if layers_to_use is not None:
    layer_indices = np.asarray(layers_to_use, dtype=int)
elif n_planes_to_use is not None:
    layer_indices = np.linspace(0, n_planes_total - 1, n_planes_to_use, dtype=int)
else:
    layer_indices = np.arange(n_planes_total)

matched_points = matched_points[:, layer_indices]
z_planes = full_z_planes[layer_indices]
n_planes = len(layer_indices)
print(f"Using {n_planes} planes for calibration (layer indices: {layer_indices.tolist()})")

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

# Optional: write merged matches and test files for consistency
calibration.write_matches()
# calibration.write_calibration_test_files()

# %%
