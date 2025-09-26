# %%
from python_4be_eti.plotting.plot_tracks import plot_tracks

filename = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center/TTI_aligned_with_gravity/Run1/tracks.h5'
frame_range = [0, 1, 2, 3]

plot_tracks(filename, frame_range)

# %%
