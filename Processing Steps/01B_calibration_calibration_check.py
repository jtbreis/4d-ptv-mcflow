# %%
import os
import subprocess
from python_4dptv.matching.rays import Rays
from python_4dptv.matching.stereomatching import StereoMatching
from python_4dptv.matching.stereomatching_error import evaluate_stereomatching_error

os.chdir('/workspaces/4d-ptv-mcflow')
# Set Folder
folder = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center/Calibration_Before'
# Ensure output folders exist
os.makedirs(os.path.join(folder, 'Error', 'after_matching_plots'), exist_ok=True)
os.makedirs(os.path.join(folder, 'Error', 'planes'), exist_ok=True)
# Set StereoMatching Parameters
mincameras = 3
maxdistance = 0.15
multiplematchesperraydistance = 2
maxmatchesperray = 1
# Number of Voxels in direction [nx, ny, nz]
nvoxels = [400, 400, 280]
# Bounding Box [minX, maxX, minY, maxY, minZ, maxZ]
boundingbox = [-50, 50, -30, 30, -20, 20]
# %% Run Ray Computation
test_folder = os.path.join(folder, 'Tests')
rays = Rays(test_folder)
rays.compute_rays()
rays.write_rays()

# %% Run Stereomatching you need to run 'cd STMCpp && make' first / only once
stereomatching = StereoMatching(test_folder, mincameras, maxdistance,
                                multiplematchesperraydistance, maxmatchesperray, nvoxels, boundingbox)
stereomatching.run_stereomatching()

# %%
evaluate_stereomatching_error(folder, boundingbox)

# %% Test Calibration_Before on points from Calibration_After (offset removed; target moved between sets)
# Same plots as above (histogram, mean error vs Z, per-Z error plots) saved with _after suffix.
folder_calibration_after = os.path.join(os.path.dirname(folder), 'Calibration_After', 'Tests')
def _has_centers(path):
    """True if path has a Centers folder (path/Centers or path/Tests/Centers, etc.)."""
    for sub in ("/Centers", "/Tests/Centers", "/Calibration/Tests/Centers/Camera"):
        p = path.rstrip(os.sep) + sub
        if os.path.isdir(p) and any(f.endswith(".h5") for f in os.listdir(p)):
            return True
    return False

if os.path.isdir(folder_calibration_after):
    if not _has_centers(folder_calibration_after):
        print(f'Calibration_After: no Centers folder found under {folder_calibration_after}')
    else:
        # Ensure output folders exist for Calibration_After
        os.makedirs(os.path.join(folder, 'Error', 'after_matching_plots'), exist_ok=True)
        # Use calibration from Calibration_Before, points (Centers) from Calibration_After
        rays_after = Rays(folder_calibration_after, calibration_folder=folder)
        rays_after.compute_rays()
        rays_after.write_rays()
        stereomatching_after = StereoMatching(folder_calibration_after, mincameras, maxdistance,
                                             multiplematchesperraydistance, maxmatchesperray, nvoxels, boundingbox)
        stereomatching_after.run_stereomatching()
        # Same evaluation as Before: histogram, mean error vs Z location, per-Z error plots (offset removed)
        stereomatching_after.plot_matches(os.path.join(folder,'Error', 'after_matching_plots'))
else:
    print(f'Calibration_After folder not found: {folder_calibration_after}')# %%

# %%
