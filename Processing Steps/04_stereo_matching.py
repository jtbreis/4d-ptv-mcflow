# %%
import os
from python_4dptv.matching.stereomatching import StereoMatching
# %load_ext autoreload
# %autoreload 2

os.chdir('/workspaces/4d-ptv-mcflow')
folder = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center/TTI_aligned_with_gravity/Run1'
# Set StereoMatching Parameters
mincameras = 3
maxdistance = 0.2
multiplematchesperraydistance = 2
maxmatchesperray = 1
# Number of Voxels in direction [nx, ny, nz]
nvoxels = [600, 600, 500]
# Bounding Box [minX, maxX, minY, maxY, minZ, maxZ]
boundingbox = [-50, 50, -35, 35, -20, 20]

# %% Run Stereomatching you need to run 'cd STMCpp && make' first / only once
stereomatching = StereoMatching(folder, mincameras, maxdistance,
                                multiplematchesperraydistance, maxmatchesperray, nvoxels, boundingbox)
stereomatching.run_stereomatching(40)
stereomatching.plot_matches()
