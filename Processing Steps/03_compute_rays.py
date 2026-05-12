# %%
import os
from python_4dptv.matching.rays import Rays
# %load_ext autoreload
# %autoreload 2

os.chdir('/workspaces/4d-ptv-mcflow')

folder = '/workspaces/4d-ptv-mcflow/raw_data/low_threshold/PTV_above/TTI_no_gravity/Run1'

# Optional: only the first N frames per camera (None = all).
max_frames = 12

# Optional: per-camera center (x,y) remap, as if that camera's image were rotated 90°.
# Use None for no remap on any camera. Otherwise pass a list with one entry per camera
# in calib order (Camera 0, 1, …): None | 'cw' | 'ccw'. Example (four cameras):
center_rotate = ['cw', 'cw', 'ccw', 'ccw']
# center_rotate = None

# Image width/height in pixels before rotation (required if any camera uses rotation).
# One int applies to every camera that rotates; or pass a list per camera if sizes differ.
image_width = 2560
image_height = 1600

rays = Rays(
    folder,
    center_rotate=center_rotate,
    image_width=image_width,
    image_height=image_height,
)
# flush_every=N: call HDF5 flush every N written frames (default 1 = every frame).
# n_workers>1: parallel frame chunks per camera, then merge partial HDF5 files.
rays.compute_rays(flush_every=50, n_workers=1, max_frames=max_frames)
# %%
# rays.plot_rays(cameras=[0, 1, 2, 3], nrays=10)
# %%
rays.write_rays()

# %%
