# %%
from mcflow_plotting.turbulence.pdf import plot_pdf, plot_normalized_pdf
from mcflow_plotting.turbulence.velocity import plot_rms_velocity
from mcflow_plotting.turbulence.velocity import plot_mean_vel_time
import numpy as np
import pandas as pd
from IPython import get_ipython

try:
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('load_ext', 'autoreload')
        ip.run_line_magic('autoreload', '2')
except Exception:
    # Not running in an IPython environment; skip autoreload
    pass

case = 'TTI_opposing_gravity'
filename = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/{case}_tracks.parquet'
# Adjust as needed
output_path = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/'

df = pd.read_parquet(filename)
time = df['time'] if 'time' in df.columns else np.arange(len(df))
# %% VELOCITY MAGNITUDE
velocity_magnitudes = df[['vmag_0', 'vmag_1', 'vmag_2']].mean(axis=1)
plot_normalized_pdf(velocity_magnitudes,
                    variable=r'\left| \left | v \right| \right| ^2', scale=1000, output=output_path, name='vmag')

# %% VELOCITY X
velocity_x = df[['vx_0', 'vx_1', 'vx_2']].mean(axis=1)
plot_normalized_pdf(velocity_x,
                    variable=r'v_\mathrm{x}', scale=1000, output=output_path, name='vx')

# %% VELOCITY Y
velocity_y = df[['vy_0', 'vy_1', 'vy_2']].mean(axis=1)
plot_normalized_pdf(velocity_y,
                    variable=r'v_\mathrm{y}', scale=1000, output=output_path, name='vy')

# %% VELOCITY Z
velocity_z = df[['vz_0', 'vz_1', 'vz_2']].mean(axis=1)
plot_normalized_pdf(velocity_z,
                    variable=r'v_\mathrm{z}', scale=1000, output=output_path, name='vz')

# %% All Velocities
plot_pdf([velocity_magnitudes, velocity_x, velocity_y, velocity_z], labels=[r'$ \left|\left| v \right|\right|^2 $', '$v_\mathrm{x}$', '$v_\mathrm{y}$', '$v_\mathrm{z}$'], scale=1000,
         variable=r'v_\mathrm{i}', xlim=[-2, 3.5], figsize=(6, 4), output=output_path, name='vi')

# %% ACCELERATION MAGNITUDE
acceleration_magnitudes = df[['amag_0', 'amag_1']].mean(axis=1)
plot_normalized_pdf(acceleration_magnitudes,
                    variable=r'\left| \left | a \right| \right| ^2', scale=1000, output=output_path, name='amag', log=True)

# %% ACCELERATION X
acceleration_x = df[['ax_0', 'ax_1']].mean(axis=1)
plot_normalized_pdf(acceleration_x,
                    variable=r'a_\mathrm{x}', unit='m/{s}^2', scale=1000, output=output_path, name='ax', log=True)

# %% ACCELERATION Y
acceleration_y = df[['ay_0', 'ay_1']].mean(axis=1)
plot_normalized_pdf(acceleration_y,
                    variable=r'a_\mathrm{y}', scale=1000, output=output_path, name='ay', log=True)
# %% ACCELERATION Z
acceleration_z = df[['az_0', 'az_1']].mean(axis=1)
plot_normalized_pdf(acceleration_z,
                    variable=r'a_\mathrm{z}', scale=1000, output=output_path, name='az', log=True)

# %% All Accelerations
plot_pdf([acceleration_magnitudes, acceleration_x, acceleration_y, acceleration_z], labels=[r'$ \left|\left| a \right|\right|^2 $', '$a_\mathrm{x}$', '$a_\mathrm{y}$', '$a_\mathrm{z}$'], scale=1000,
         variable=r'a_\mathrm{i}', xlim=[-100, 100], figsize=(6, 4), output=output_path, name='ai', log=True)

# %%
