# %%
from mcflow_plotting.turbulence.pdf import plot_pdf, plot_pdf_log
from mcflow_plotting.turbulence.velocity import plot_rms_velocity
from mcflow_plotting.turbulence.velocity import plot_mean_vel_time
import numpy as np

case = 'TTI_aligned_with_gravity'
run = 'Run4'
filename = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/{run}/tracks.h5'
output_path = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}'

samples = load_tracks(filename, 1e-3, 10)

# %%
track_lengths = [track.track_length for track in samples]
mean_track_length = np.mean(track_lengths)
print(f"Mean track length in samples: {mean_track_length}")
print(f"Total number of tracks is {len(track_lengths)}")
# %%
time = [track.time for track in samples]
# %%
velocity_magnitudes = [np.mean(track.vmag) for track in samples]
plot_pdf(velocity_magnitudes, scale=1000,
         variable='||V||2', output=output_path)
plot_mean_vel_time(velocity_magnitudes, time, scale=1000,
                   variable='||V||^2', output=output_path)
# %% VELOCITY X
velocity_x = [np.mean(track.vx) for track in samples]
plot_pdf(velocity_x, scale=1000, variable='V_x', output=output_path)
plot_mean_vel_time(velocity_x, time, scale=1000,
                   variable='V_x', output=output_path)
# %% VELOCITY Y
velocity_y = [np.mean(track.vy) for track in samples]
plot_pdf(velocity_y, scale=1000, variable='V_y', output=output_path)
plot_mean_vel_time(velocity_y, time, scale=1000,
                   variable='V_y', output=output_path)

# %% VELOCITY Z
velocity_z = [np.mean(track.vz) for track in samples]
plot_pdf(velocity_z, scale=1000, variable='V_z', output=output_path)
plot_mean_vel_time(velocity_z, time, scale=1000,
                   variable='V_z', output=output_path)

# %%
# acceleration_magnitude = [track.amag for track in samples]
# plot_pdf_log(acceleration_magnitude)

# %%
