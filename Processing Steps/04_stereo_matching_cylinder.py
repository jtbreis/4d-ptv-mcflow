# %%
"""
Stereo matching with finite-diameter rays (cylinder-style STM).

Requires ``rays.h5`` with a ``diameter`` dataset per frame (from compute_rays).
The C++ STM is invoked with --cylinder-voxels, --cylinder-rank, and --cylinder-h5-aux;
see STMCpp/STM_main.cpp and ``StereoMatching.run_stereomatching(..., cylinder=True)``.
"""
import os
from python_4dptv.matching.stereomatching import StereoMatching
# %load_ext autoreload
# %autoreload 2

os.chdir('/workspaces/4d-ptv-mcflow')
folder = '/workspaces/4d-ptv-mcflow/data/high_threshold/PTV_above/TTI_no_gravity/Run1'
# Set StereoMatching Parameters
mincameras = 3
maxdistance = 0.07
multiplematchesperraydistance = 0.001
maxmatchesperray = 100
# Number of Voxels in direction [nx, ny, nz]
nvoxels = [350 * 2, 550 * 2, 250 * 2]
# Bounding Box [minX, maxX, minY, maxY, minZ, maxZ]
boundingbox = [-35, 35, -55, 55, -25, 25]

# %% Run Stereomatching you need to run 'cd STMCpp && make' first / only once
stereomatching = StereoMatching(folder, mincameras, maxdistance,
                                multiplematchesperraydistance, maxmatchesperray, nvoxels, boundingbox)
stereomatching.run_stereomatching(
    nthreads=32,
    nframes=12,
    timing=True,
    spatial_boxes=16,
    spatial_overlap_cells=1,
    frame_parallelism=2,
    cylinder=True,
    cylinder_max_shells=32,
    # particle props from rays.h5 → rays_out_cpp.h5 (defaults: all on)
    # write_particle_props=False,  # skip diameter/intensity/mass entirely
    # write_mass=False,            # or disable individual datasets
)
# stereomatching.plot_matches()
