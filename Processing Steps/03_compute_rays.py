# %%
import os
from python_4dptv.matching.rays import Rays
# %load_ext autoreload
# %autoreload 2

os.chdir('/workspaces/4d-ptv-mcflow')

folder = '/workspaces/4d-ptv-mcflow/raw_data/low_threshold/PTV_below/TTI_aligned_with_gravity/Run1'

rays = Rays(folder)
# flush_every=N: call HDF5 flush every N written frames (default 1 = every frame).
# n_workers>1: parallel frame chunks per camera, then merge partial HDF5 files.
rays.compute_rays(flush_every=50, n_workers=24)
# %%
# rays.plot_rays(cameras=[0, 1, 2, 3], nrays=10)
# %%
rays.write_rays()

# %%
