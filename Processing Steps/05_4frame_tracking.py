# %%
from python_4be_eti.perform_tracking import FourFrameTracking
from python_4be_eti.utils.basic_utils import create_h5_file

# directory containing the file with particle information (location, area, frame number)
# you must include the full path to the directory here
folder = '/workspaces/4d-ptv-mcflow/data/tracking_test_threshold1'
filename = 'rays_out_cpp.h5'
run = 'run1'

create_h5_file(folder)

# 2d or 3d tracking?
dimension = '3d'

# Set box sizes for track initialization (using max)
box_size_initial_x_max = 3.5
box_size_initial_x_low = 1.0
box_size_initial_y_max = 0.2
box_size_initial_y_low = -0.6
box_size_initial_z = 0.5

# Set box size used after a track is initialized (using min)
box_size = 0.25

# %%
# Set write_failed_tracks=True to also write particles for which tracking failed (id=0, dataset 'tracked'=False per step)
tracking = FourFrameTracking(
    folder,
    filename=filename,
    box_size_x=box_size_initial_x_max,
    box_size_initial_x_lo=box_size_initial_x_low,
    box_size_initial_x_hi=box_size_initial_x_max,
    box_size_y=box_size_initial_y_max,
    box_size_initial_y_lo=box_size_initial_y_low,
    box_size_initial_y_hi=box_size_initial_y_max,
    box_size_z=box_size_initial_z,
    box_size_track=box_size,
    dt=1e-3,
    rep_rate=10,
    write_paraview=True,
    write_failed_tracks=True,
    use_bspline=True,
    export_bspline_paraview=True
)

# %%
tracking.run_tracking(2)

# %%
