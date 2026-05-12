# %%
from python_4be_eti.perform_tracking import FourFrameTracking
from python_4be_eti.utils.basic_utils import create_h5_file
import os

# Numba fast path for ``no_previous_tracks_3d`` (must be set before importing ``python_4be_eti``).
# In Jupyter, restart the kernel or run this cell first after changing ``use_numba``.
use_numba = True
os.environ["4BE_ETI_NUMBA"] = "1" if use_numba else "0"


# directory containing the file with particle information (location, area, frame number)
# you must include the full path to the directory here
folder = "/workspaces/4d-ptv-mcflow/raw_data/low_threshold/PTV_above/TTI_no_gravity/Run1"
filename = "rays_out_cpp.h5"
run = "run1"

create_h5_file(folder)

# 2d or 3d tracking?
dimension = "3d"

# Initial x search on frame+1: offsets from seed; window is [x0+min(lo,hi), x0+max(lo,hi)].
# Here: x in [x0+1, x0+3.5] (forward in +x only).
box_size_initial_x_lo = 0.0
box_size_initial_x_hi = 3.5
box_size_initial_x_max = max(
    abs(box_size_initial_x_lo), abs(box_size_initial_x_hi))
box_size_initial_y_max = 1.5
box_size_initial_y_lo = -1.5
box_size_initial_y_hi = box_size_initial_y_max
box_size_initial_z = 1.5

# Set box size used after a track is initialized (using min)
box_size = 0.07

# Optional 3D meshgrid caps (None = package defaults 1000 / 2000); increase for very dense frames.
max_candidates_mesh = 3000
max_targets_mesh = 3000

# Print mesh-stage sizes for the first N new-track (no_previous_tracks_3d) attempts with frame+1 candidates (0 = off).
# When > 0, track initialization uses NumPy so sizes match the mesh-debug logic (slower).
debug_mesh_match_prints = 0

# %%
# Time: ``dt`` = seconds between consecutive stereo frames; ``rep_rate`` = Hz at which each
# 4-frame block is sampled. Step ``Time`` = (frame//4)/rep_rate + (frame%4)*dt (seconds).
# Positions stay in ``position_unit`` (e.g. mm); vx,vy,vz are m/s; ax,ay,az are m/s^2.
# Set write_failed_tracks=True to also write particles for which tracking failed (id=0, dataset 'tracked'=False per step)
tracking = FourFrameTracking(
    folder,
    filename=filename,
    box_size_x=box_size_initial_x_max,
    box_size_initial_x_lo=box_size_initial_x_lo,
    box_size_initial_x_hi=box_size_initial_x_hi,
    box_size_y=box_size_initial_y_max,
    box_size_initial_y_lo=box_size_initial_y_lo,
    box_size_initial_y_hi=box_size_initial_y_hi,
    box_size_z=box_size_initial_z,
    box_size_track=box_size,
    dt=1e-3,
    rep_rate=10,
    position_unit="mm",
    write_paraview=True,
    write_failed_tracks=False,
    use_bspline=True,
    export_bspline_paraview=True,
    # print ``frame: particle i/n`` every N particles; 0 = off (default in package is 1000)
    particle_progress_interval=10000,
    max_candidates_mesh=max_candidates_mesh,
    max_targets_mesh=max_targets_mesh,
    debug_mesh_match_prints=debug_mesh_match_prints,
)

# %%
# Number of worker processes; set max_frame_ranges to an integer to only process the first N frame ranges, or None for all.
workers = 3
max_frame_ranges = 1
tracking.run_tracking(workers, max_frame_ranges=max_frame_ranges)

# %%
