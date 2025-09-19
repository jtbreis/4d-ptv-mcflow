# %%
# Rename Calibration Files
import os
from ptv_calib.utils.rename_images import rename_images
import glob

# %%
folder = '/workspaces/4d-ptv-mcflow/raw_data/julian/2025-09-11-ParticleTracking/Calibration_After/Camera1'

rename_images(folder_path=folder)

# %%
# Rename Run Files for Camera 1, Camera 2, Camera 3, Camera 4

run_folder = '/workspaces/4d-ptv-mcflow/raw_data/julian/2025-09-11-ParticleTracking/TTI_opposing_gravity'
num_cameras = 4

run_folders = [f.path for f in os.scandir(run_folder) if f.is_dir()]
for folder in run_folders:
    for cam_num in range(1, num_cameras+1):
        pattern = os.path.join(folder, f'Camera {cam_num}_*.cine')
        files = glob.glob(pattern)
        for old_filename in files:
            new_filename = os.path.join(folder, f'Camera{cam_num}.cine')
            if os.path.exists(old_filename):
                os.rename(old_filename, new_filename)
