# %%
import os
from python_4dptv.matching.stereomatching import StereoMatching
# %load_ext autoreload
# %autoreload 2

os.chdir('/workspaces/4d-ptv-mcflow')
folder = '/workspaces/4d-ptv-mcflow/raw_data/low_threshold/PTV_above/TTI_no_gravity/Run1'
# Set StereoMatching Parameters
mincameras = 3
maxdistance = 0.07
multiplematchesperraydistance = 0.001
maxmatchesperray = 100
# Number of Voxels in direction [nx, ny, nz]
nvoxels = [350*2, 550*2, 250*2]
# Bounding Box [minX, maxX, minY, maxY, minZ, maxZ]
boundingbox = [-35, 35, -55, 55, -25, 25]

# %% Run Stereomatching you need to run 'cd STMCpp && make' first / only once
stereomatching = StereoMatching(folder, mincameras, maxdistance,
                                multiplematchesperraydistance, maxmatchesperray, nvoxels, boundingbox)
stereomatching.run_stereomatching(
    nthreads=32,
    nframes=12,
    timing=True,
    # set >0 to match C++ --spatial-boxes (e.g. 16); use nthreads >= spatial_boxes for full parallelism
    spatial_boxes=16,
    spatial_overlap_cells=1,
    # max frames at once (C++ --frame-parallelism); omit or 0 = use OMP_NUM_THREADS (= nthreads here)
    frame_parallelism=2,
)
# stereomatching.plot_matches()
