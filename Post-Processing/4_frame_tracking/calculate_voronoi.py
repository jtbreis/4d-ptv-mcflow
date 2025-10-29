# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mcflow_plotting.inertial_particles.voronoi import plot_voronoi, compute_voronoi
from IPython import get_ipython

try:
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('load_ext', 'autoreload')
        ip.run_line_magic('autoreload', '2')
except Exception:
    # Not running in an IPython environment; skip autoreload
    pass

# %% LOAD CASE
case = 'TTI_opposing_gravity'
filename = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/{case}_tracks.parquet'
# Adjust as needed
output_path = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/'

df = pd.read_parquet(filename)

# %%
plt.figure()
X_positions = df[['X_0', 'X_1', 'X_2', 'X_3']].dropna().values.flatten()
Y_positions = df[['Y_0', 'Y_1', 'Y_2', 'Y_3']].dropna().values.flatten()
Z_positions = df[['Z_0', 'Z_1', 'Z_2', 'Z_3']].dropna().values.flatten()
plt.hist(X_positions, alpha=0.7, bins=100, label='X')
plt.hist(Y_positions, alpha=0.7, bins=100, label='Y')
plt.hist(Z_positions, alpha=0.7, bins=100, label='Z')
plt.xlabel('position (mm)')
plt.ylabel('Count')
plt.title('Distribution of points in directions')
plt.legend()
plt.grid(True)
plt.show()

print('X: mean =', np.mean(X_positions), ' std =', np.std(X_positions))
print('Y: mean =', np.mean(Y_positions), ' std =', np.std(Y_positions))
print('Z: mean =', np.mean(Z_positions), ' std =', np.std(Z_positions))

xbb = [np.mean(X_positions) - np.std(X_positions),
       np.mean(X_positions) + np.std(X_positions)]
# Uncomment if Y should be a Gaussian distribution
# ybb = [np.mean(Y_positions) + np.std(Y_positions),
#    np.mean(Y_positions) - np.std(Y_positions)]
ybb = [np.min(Y_positions), np.max(Y_positions)]
zbb = [np.mean(Z_positions) - np.std(Z_positions),
       np.mean(Z_positions) + np.std(Z_positions)]

# %%
# read points from dataframe columns x_0, y_0, z_0
# select rows for run 1 at time 0 and extract x_0, y_0, z_0 as an (N,3) array
sel = df[(df['run'] == 1) & (df['time'] == 0)]
points = sel[['X_0', 'Y_0', 'Z_0']].dropna().to_numpy()


plot_voronoi(points, [xbb[0], xbb[1], ybb[0], ybb[1], zbb[0], zbb[1]])

# %%
for time_value in df['time'].unique():
    sel = df[(df['run'] == 1) & (df['time'] == time_value)]
    for i in range(4):

        points = sel[[f'X_{i}', f'Y_{i}', f'Z_{i}']].dropna().to_numpy()

        volumes, normalized_volumes = compute_voronoi(
            points, [xbb[0], xbb[1], ybb[0], ybb[1], zbb[0], zbb[1]])

        # Add volumes to the original dataframe for run 1 at the current time
        sel_indices = sel.index
        df.loc[sel_indices, f'voronoi_volume_{i}'] = volumes
        df.loc[sel_indices,
               f'voronoi_volume_normalized{i}'] = normalized_volumes

# %%
# Write the updated DataFrame back to a parquet file
output_file = f'{output_path}/{case}_tracks_with_voronoi.parquet'
df.to_parquet(output_file)

# %%
