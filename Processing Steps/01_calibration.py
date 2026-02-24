# %%
import numpy as np
from ptv_calib.calibrate import Calibration
import os
os.chdir('/workspaces/4d-ptv-mcflow')

%load_ext autoreload
%autoreload 2
# %%

cameras = [1, 2, 3, 4]
folder_path = '/workspaces/4d-ptv-mcflow/raw_data/2025-11-10-ParticleTracking_BelowTTI/Calibration_Before'
output_path = 'data/julian/PTV_below'
calibration_grid_path = 'PTVcalib/calibration_targets/TSI_5mm_backlight_nX39_nY39.csv'

calibration_method = '4d-ptv'
grid_spacing = 5.0  # in mm
z_min = -25  # in mm
z_max = 25  # in mm

target_point_diameter = 20
number_of_planes = 6

# %%
calibration = Calibration(cameras=cameras, folder_path=folder_path, output_path=output_path, calibration_grid_path=calibration_grid_path,
                          grid_spacing=grid_spacing, target_point_diameter=target_point_diameter, z_min=z_min, z_max=z_max, n_planes=number_of_planes, calibration_method=calibration_method)
calibration.set_plotting_mode('Normal')
# %%
calibration.preprocess_images(enhance_contrast='None')
# %% Optional: interactively remove spurious points (Jupyter / Run Cell)
# For each (camera, plane) an image is shown; type an index (or "5,6,7") to remove points, or Enter for next.
# Single image: calibration.interactively_remove_detected_points(cam_idx=0, plane_idx=0)
for cam_idx in range(calibration.ncameras):
    for plane_idx in range(calibration.n_planes):
        print(f"--- Camera {cam_idx}, Plane {plane_idx} ---")
        calibration.interactively_remove_detected_points(cam_idx, plane_idx)

# %%
calibration.match_calibration_grid(center_find_method='TSI-backlight')
# %%
calibration.write_matches()
calibration.write_calibration_test_files()
# %%
calibration.perform_calibration()

# %%
calibration.write_calibration()

# %%
