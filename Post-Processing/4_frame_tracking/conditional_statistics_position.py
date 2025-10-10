# %%

import matplotlib.pyplot as plt
from mcflow_plotting.turbulence.pdf import plot_pdf
import h5py
import numpy as np
import pandas as pd

filename = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center/TTI_opposing_gravity/TTI_opposing_gravity_tracks.parquet'

num_bins = 10
bin_edges = np.linspace(-35, 35, num_bins + 1)
binned_data_vy = [[] for _ in range(num_bins)]
binned_data_y = [[] for _ in range(num_bins)]

# %%
df = pd.read_parquet(filename)
for index, row in df.iterrows():
    y = row['Y']
    vy = row['vy_1']
    # Bin each vy value according to its corresponding y value
    bin_idx = np.digitize(y, bin_edges) - 1
    if 0 <= bin_idx < num_bins:
        binned_data_vy[bin_idx].append(vy)
        binned_data_y[bin_idx].append(y)

# %% VELOCITY Y
samples = binned_data_vy[0]
velocity_y = [data_point for data_point in samples]
plot_pdf(velocity_y, scale=1000, variable='V_y')

# %%
samples = binned_data_vy[-1]
velocity_y = [data_point for data_point in samples]
plot_pdf(velocity_y, scale=1000, variable='V_y')

# %% ACCELERATION X
samples = df['amean']
acceleration_x = [data_point for data_point in samples]
plot_pdf(acceleration_x, scale=1000, variable='A_x')

# %%
means = [np.mean(bin) if bin else np.nan for bin in binned_data_vy]
stds = [np.std(bin) if bin else np.nan for bin in binned_data_vy]
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.errorbar(bin_centers, means, yerr=stds, fmt='o', capsize=5)
plt.xlabel('Y position (bin center)')
plt.ylabel('Mean V_y ± Std Dev')
plt.title('Mean and Std Dev of V_y across Y bins')
plt.grid(True)
plt.show()

# %%
