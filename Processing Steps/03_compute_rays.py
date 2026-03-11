# %%
import os
from python_4dptv.matching.rays import Rays
# %load_ext autoreload
# %autoreload 2

os.chdir('/workspaces/4d-ptv-mcflow')

folder = '/workspaces/4d-ptv-mcflow/data/tracking_test_threshold1'

rays = Rays(folder)
rays.compute_rays()
# %%
# rays.plot_rays(cameras=[0, 1, 2, 3], nrays=10)
# %%
rays.write_rays()

# %%
