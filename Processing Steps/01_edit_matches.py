# %%
import numpy as np
from ptv_calib.calibrate import Calibration
import os
os.chdir('/workspaces/4d-ptv-mcflow')

# %load_ext autoreload
# %autoreload 2
# %%

cameras = [1, 2, 3, 4]
folder_path = '/workspaces/4d-ptv-mcflow/raw_data/2025-11-10-ParticleTracking_BelowTTI/Calibration_Before'
output_path = 'data/julian/PTV_below'
calibration_grid_path = 'PTVcalib/calibration_targets/TSI_5mm_backlight_nX39_nY39.csv'

calibration_method = '4d-ptv'
grid_spacing = 5.0  # in mm
z_min = -25  # in mm
z_max = 25  # in mm

target_point_diameter = 15
number_of_planes = 6

# %%
calibration = Calibration(cameras=cameras, folder_path=folder_path, output_path=output_path, calibration_grid_path=calibration_grid_path,
                          grid_spacing=grid_spacing, target_point_diameter=target_point_diameter, z_min=z_min, z_max=z_max, n_planes=number_of_planes, calibration_method=calibration_method)

# %%
# Points to remove: (camera_index, X, Y, Z) in mm — Z is the plane depth
points_to_remove = [
    (2, 40.0, 5.0, -15.0),
    (2, 40.0, 0.0, -15.0),
    (2, 40.0, -5.0, -15.0),
]

# %%
calibration.load_matches_from_file()

# %%
calibration.remove_matched_points(
    points_to_remove, grid_tolerance_mm=0.5, redraw=True)
