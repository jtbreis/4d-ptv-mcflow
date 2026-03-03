# %%
"""
Find the optimal number of calibration layers by running calibration on different
subsets of layers, then running rays + stereomatching. Error is the mean distance
(mm) between stereo-matched 3D particle positions and the known calibration grid
(Centers). Plots error vs number of layers and reports the optimal count.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from ptv_calib.calibrate import Calibration
from ptv_calib.io.output import read_h5_matches, write_h5_test_files, write_h5_calibration
from ptv_calib.utils.structure import create_folder_structure
from python_4dptv.matching.rays import Rays
from python_4dptv.matching.stereomatching import StereoMatching
from python_4dptv.matching.stereomatching_error import compute_stereomatching_error_by_layers

os.chdir('/workspaces/4d-ptv-mcflow')

# %%
# --- Configuration ---
# Path to existing matches file (e.g. from 01_2 or after running calibration pipeline)
matches_path = 'data/julian/PTV_center/Calibration_Before/Matches/matches.h5'
output_path = 'data/julian/PTV_center'
calibration_grid_path = 'PTVcalib/calibration_targets/TSI_5mm_backlight_nX39_nY39.csv'

cameras = [1, 2, 3, 4]
calibration_method = '4d-ptv'
grid_spacing = 5.0  # mm
z_min = -20  # mm
z_max = 20  # mm
target_point_diameter = 20

# StereoMatching parameters (used for ray intersection)
mincameras = 3
maxdistance = 0.2
multiplematchesperraydistance = 1
maxmatchesperray = 2
nvoxels = [600, 800, 500]
boundingbox = [-50, 50, -30, 30, -20, 20]

# Range of number of layers to test
min_layers = 3
max_layers = None  # None = use half of available layers; set to n_planes_total to use all
# Optional: test only these exact counts (overrides min/max)
# n_layers_to_try = [3, 5, 7, 11, 15, 21, 31]
n_layers_to_try = None

# Output plot path (relative to output_path or absolute)
plot_save_path = None  # e.g. 'Calibration/error_vs_n_layers.pdf'

# %%
# Load full matched points
matched_points_full = read_h5_matches(matches_path)
n_cameras, n_planes_total = matched_points_full.shape
full_z_planes = np.linspace(z_min, z_max, n_planes_total)
print(f"Loaded matches: {n_cameras} cameras, {n_planes_total} layers")

# Prepare output folder: calibration structure + Centers (all layers) for rays/stereomatching.
# All rays and stereomatching I/O lives under Calibration/Tests.
output_path = output_path.rstrip('/')
create_folder_structure(output_path)
tests_dir = os.path.join(output_path, 'Calibration', 'Tests')
centers_camera_dir = os.path.join(tests_dir, 'Centers')
os.makedirs(centers_camera_dir, exist_ok=True)
# Write Centers from all layers once (known 3D grid for stereomatching error)
write_h5_test_files(matched_points_full, centers_camera_dir + '/')
print(f"Wrote Centers ({n_planes_total} layers) to {centers_camera_dir}")

# Decide which n_layers values to try
if n_layers_to_try is not None:
    n_layers_list = [k for k in n_layers_to_try if 1 <= k <= n_planes_total]
else:
    max_n = min(n_planes_total, max_layers) if max_layers is not None else n_planes_total // 2
    n_layers_list = list(range(min_layers, max_n + 1))
if not n_layers_list:
    raise ValueError("No layer counts to try. Check min_layers, max_layers, n_layers_to_try.")

print(f"Testing n_layers: {n_layers_list}")

# %%
# For each n_layers: calibrate, write calib, run rays + stereomatching, compute error on cal and in-between layers
errors_cal = []
errors_in_between = []
for n_layers in n_layers_list:
    layer_indices = np.linspace(0, n_planes_total - 1, n_layers, dtype=int)
    mp_subset = matched_points_full[:, layer_indices]
    z_planes = full_z_planes[layer_indices]

    cal = Calibration(
        cameras=cameras,
        folder_path='',
        output_path=output_path,
        calibration_grid_path=calibration_grid_path,
        grid_spacing=grid_spacing,
        target_point_diameter=target_point_diameter,
        z_min=z_min,
        z_max=z_max,
        n_planes=n_layers,
        calibration_method=calibration_method,
    )
    cal.set_custom_zplanes(z_planes.tolist())
    cal.matched_points = mp_subset
    cal.perform_calibration()
    cal.write_calibration()

    # Rays and StereoMatching read/write under Calibration/Tests (centers, rays, STM output)
    rays = Rays(tests_dir, calibration_folder=output_path)
    rays.compute_rays()
    rays.write_rays()

    stereomatching = StereoMatching(
        tests_dir, mincameras, maxdistance,
        multiplematchesperraydistance, maxmatchesperray, nvoxels, boundingbox
    )
    stereomatching.run_stereomatching()

    err_cal, err_in_between = compute_stereomatching_error_by_layers(
        tests_dir, layer_indices, remove_offset=False
    )
    errors_cal.append(err_cal)
    errors_in_between.append(err_in_between)
    print(f"  n_layers={n_layers} -> stereo error on cal layers = {err_cal:.4f} mm, on in-between = {err_in_between:.4f} mm")

errors_cal = np.array(errors_cal)
errors_in_between = np.array(errors_in_between)

# %%
# Plot stereomatching error (mm) on calibration layers and on in-between layers
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_layers_list, errors_cal, 'o-', color='#2e86ab', linewidth=2, markersize=8,
        label='On calibration layers')
ax.plot(n_layers_list, errors_in_between, 's-', color='#e94f37', linewidth=2, markersize=8,
        label='On in-between layers')
ax.set_xlabel('Number of calibration layers', fontsize=12)
ax.set_ylabel('Mean stereomatching error (mm)', fontsize=12)
ax.set_title('Particle position error after stereomatching vs number of layers')
ax.grid(True, alpha=0.3)

idx_min = np.nanargmin(errors_in_between)
ax.axvline(n_layers_list[idx_min], color='#e94f37', linestyle='--', alpha=0.7,
           label=f'Min in-between at {n_layers_list[idx_min]} layers')
ax.legend()

if plot_save_path:
    save_path = plot_save_path if os.path.isabs(plot_save_path) else os.path.join(output_path, plot_save_path)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# Always save the optimal_n_layers plot in calibration/validation/
validation_dir = os.path.join(output_path, 'Calibration', 'validation')
os.makedirs(validation_dir, exist_ok=True)
validation_plot_path = os.path.join(validation_dir, 'optimal_n_layers.pdf')
plt.savefig(validation_plot_path)
print(f"Plot saved to {validation_plot_path}")
plt.show()

# %%
optimal_n_layers = n_layers_list[idx_min]
print(f"Optimal number of layers (lowest error on in-between): {optimal_n_layers} "
      f"(in-between = {errors_in_between[idx_min]:.4f} mm, cal layers = {errors_cal[idx_min]:.4f} mm)")

# Re-run calibration for optimal n_layers and write to calib.h5
layer_indices_opt = np.linspace(0, n_planes_total - 1, optimal_n_layers, dtype=int)
mp_opt = matched_points_full[:, layer_indices_opt]
z_planes_opt = full_z_planes[layer_indices_opt]
cal_opt = Calibration(
    cameras=cameras,
    folder_path='',
    output_path=output_path,
    calibration_grid_path=calibration_grid_path,
    grid_spacing=grid_spacing,
    target_point_diameter=target_point_diameter,
    z_min=z_min,
    z_max=z_max,
    n_planes=optimal_n_layers,
    calibration_method=calibration_method,
)
cal_opt.set_custom_zplanes(z_planes_opt.tolist())
cal_opt.matched_points = mp_opt
cal_opt.perform_calibration()
calib_h5_path = os.path.join(output_path, 'Calibration', 'calib.h5')
os.makedirs(os.path.dirname(calib_h5_path), exist_ok=True)
write_h5_calibration(cal_opt.calibration, calib_h5_path)
print(f"Optimal calibration written to {calib_h5_path}")
