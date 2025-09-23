# %%
import os
import subprocess
from python_4dptv.matching.rays import Rays
from python_4dptv.matching.stereomatching import StereoMatching

os.chdir('/workspaces/4d-ptv-mcflow')
# Set Folder
folder = 'data/julian/PTV_center/Calibration/Tests'
# Set StereoMatching Parameters
mincameras = 3
maxdistance = 0.05
multiplematchesperraydistance = 1
maxmatchesperray = 2
# Number of Voxels in direction [nx, ny, nz]
nvoxels = [400, 400, 250]
# Bounding Box [minX, maxX, minY, maxY, minZ, maxZ]
boundingbox = [-50, 50, -35, 35, -20, 20]
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
