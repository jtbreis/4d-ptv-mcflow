# %%
from mcflow_plotting.turbulence.pdf import plot_pdf, plot_normalized_pdf, plot_pdf_log
from mcflow_plotting.turbulence.velocity import plot_rms_velocity
from mcflow_plotting.turbulence.velocity import plot_mean_vel_time
import numpy as np
import pandas as pd

# Enable autoreload for interactive development
%load_ext autoreload
%autoreload 2

filename = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center/TTI_opposing_gravity/TTI_opposing_gravity_tracks.parquet'
df = pd.read_parquet(filename)

# %% TODO add a method to plot the evolution of the data from every frame
# %%
velocity_magnitudes = df['vmag_0']
velocity_x = df['vx_0']
time = df['time'] if 'time' in df.columns else np.arange(len(df))
output_path = 'output'  # Adjust as needed
samples = df.itertuples()
plot_pdf([velocity_magnitudes, velocity_x], labels=['Mag', 'X'], scale=1000,
         variable=r'\left|\left| V \right|\right|^2')
plot_normalized_pdf(velocity_magnitudes, scale=1000)
# plot_mean_vel_time(velocity_magnitudes, time, scale=1000,
#                    variable='||V||^2', output=output_path)
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
