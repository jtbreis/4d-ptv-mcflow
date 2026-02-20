# %%
from python_4be_eti.perform_tracking import FourFrameTracking
from python_4be_eti.utils.basic_utils import create_h5_file

# directory containing the file with particle information (location, area, frame number)
# you must include the full path to the directory here
folder = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center/TTI_opposing_gravity/Run4'
filename = 'rays_out_cpp.h5'
run = 'run4'

create_h5_file(folder)

# 2d or 3d tracking?
dimension = '3d'
# box size in x direction for track initialization (a good initial guess is the expected
# maximum displacement of the particles in the x direction between frames)
box_size_initial_x = 3.5
# box size in y direction for track initialization
box_size_initial_y = 1
# box size in z direction for track initialization
box_size_initial_z = 1
# box size used after a track is initialized (this should be as small as possible to
# eliminate spurious track)
box_size = 1
# %%
tracking = FourFrameTracking(folder, filename=filename, box_size_x=box_size_initial_x,
                             box_size_y=box_size_initial_y, box_size_z=box_size_initial_z, box_size_track=box_size, dt=1e-3, rep_rate=10, write_paraview=True)

# %%
tracking.run_tracking(20)

# %%
