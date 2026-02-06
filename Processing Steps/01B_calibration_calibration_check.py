# %%
import os
import subprocess
from python_4dptv.matching.rays import Rays
from python_4dptv.matching.stereomatching import StereoMatching
from python_4dptv.matching.stereomatching_error import evaluate_stereomatching_error

os.chdir('/workspaces/4d-ptv-mcflow')
# Set Folder
folder = '/workspaces/4d-ptv-mcflow/data/julian/PTV_below/Calibration/Tests'
# Set StereoMatching Parameters
mincameras = 3
maxdistance = 0.2
multiplematchesperraydistance = 1
maxmatchesperray = 1
# Number of Voxels in direction [nx, ny, nz]
nvoxels = [600, 800, 500]
# Bounding Box [minX, maxX, minY, maxY, minZ, maxZ]
boundingbox = [-30, 30, -50, 50, -25, 25]
# %% Run Ray Computation
rays = Rays(folder)
rays.compute_rays()
rays.write_rays()

# %% Run Stereomatching you need to run 'cd STMCpp && make' first / only once
stereomatching = StereoMatching(folder, mincameras, maxdistance,
                                multiplematchesperraydistance, maxmatchesperray, nvoxels, boundingbox)
stereomatching.run_stereomatching()

# %%
stereomatching.plot_matches()
# %%
evaluate_stereomatching_error(folder, boundingbox)

# %%
