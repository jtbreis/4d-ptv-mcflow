# %%
from python_4dptv.matching.rays import Rays
import os

os.chdir('/workspaces/4d-ptv-mcflow')

folder = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center/TTI_aligned_with_gravity/Run1'

rays = Rays(folder)
rays.compute_rays()
rays.plot_rays(cameras=[0, 1], nrays=5)
# %%
