# %%
import os
from python_4dptv.matching.stereomatching import StereoMatching
# %load_ext autoreload
# %autoreload 2

os.chdir('/workspaces/4d-ptv-mcflow')
folder = '/workspaces/4d-ptv-mcflow/raw_data/low_threshold/PTV_center/TTI_aligned_with_gravity/Run2'
# Set StereoMatching Parameters
mincameras = 3
maxdistance = 0.15
multiplematchesperraydistance = 0.5
maxmatchesperray = 4
# Number of Voxels in direction [nx, ny, nz]
nvoxels = [550, 350, 200]
# Bounding Box [minX, maxX, minY, maxY, minZ, maxZ]
boundingbox = [-55, 55, -35, 35, -20, 20]

# %% Run Stereomatching you need to run 'cd STMCpp && make' first / only once
stereomatching = StereoMatching(folder, mincameras, maxdistance,
                                multiplematchesperraydistance, maxmatchesperray, nvoxels, boundingbox)
stereomatching.run_stereomatching(nthreads=12, nframes=4)
# stereomatching.plot_matches()
