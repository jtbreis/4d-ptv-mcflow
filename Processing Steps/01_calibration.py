import numpy as np
from ptv_calib.calibrate import Calibration
import os
os.chdir('/workspaces/4d-ptv-mcflow')
# %%

cameras = [1, 2, 3, 4]
folder_path = 'raw_data/julian/2025-09-11-ParticleTracking/Calibration_Before'
output_path = 'data/julian/PTV_center'
calibration_grid_path = 'PTVcalib/calibration_targets/TSI_5mm_backlight_nX39_nY39.csv'

calibration_method = '4d-ptv'
grid_spacing = 5.0  # in mm
z_min = -20  # in mm
z_max = 20  # in mm

target_point_diameter = 15
number_of_planes = 9

# %%
calibration = Calibration(cameras=cameras, folder_path=folder_path, output_path=output_path, calibration_grid_path=calibration_grid_path,
                          grid_spacing=grid_spacing, target_point_diameter=target_point_diameter, z_min=z_min, z_max=z_max, n_planes=number_of_planes, calibration_method=calibration_method)
calibration.set_plotting_mode('Normal')
# %%
calibration.preprocess_images(enhance_contrast='None')
# %%
calibration.match_calibration_grid(center_find_method='TSI-backlight')
# %%
calibration.write_matches()
calibration.write_calibration_test_files()
# %%
calibration.perform_calibration()

# %%
calibration.write_calibration()
