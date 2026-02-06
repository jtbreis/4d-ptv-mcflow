# %%

import matplotlib.pyplot as plt
from mcflow_plotting.turbulence.pdf import plot_pdf, plot_conditional_pdf, plot_conditional_normalized_pdf
import h5py
import numpy as np
import pandas as pd

case = 'TTI_no_gravity'
filename = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/{case}_tracks_with_voronoi.parquet'

num_bins = 6
bin_edges = np.linspace(-35, 35, num_bins + 1)
binned_data_vy = [[] for _ in range(num_bins)]
binned_data_y = [[] for _ in range(num_bins)]

# %%
df = pd.read_parquet(filename)
# Assign each row to a bin based on 'Y'

# %% VELOCITY Y
position_y = df['Y']
velocity_y = df[['vx_0', 'vx_1', 'vx_2']].mean(axis=1)
plot_conditional_pdf(position_y, velocity_y, bin_edges,
                     scale=1000, variable='V_\mathrm{{y}}', condition_label='Y')
plot_conditional_normalized_pdf(position_y, velocity_y, bin_edges,
                                scale=1000, variable='V_\mathrm{{y}}', condition_label='Y')

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
