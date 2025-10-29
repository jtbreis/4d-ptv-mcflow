# %%
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from mcflow_plotting.turbulence.pdf import plot_pdf
from IPython import get_ipython
from pathlib import Path

try:
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('load_ext', 'autoreload')
        ip.run_line_magic('autoreload', '2')
except Exception:
    # Not running in an IPython environment; skip autoreload
    pass

# %% LOAD CASE
case = 'TTI_aligned_with_gravity'
filename = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/{case}_tracks_with_voronoi.parquet'
# Adjust as needed
output_path = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/'

df = pd.read_parquet(filename)

voronoi_volume = df[['voronoi_volume_normalized0', 'voronoi_volume_normalized1',
                     'voronoi_volume_normalized2', 'voronoi_volume_normalized3']].dropna().values.flatten()
voronoi_volume = voronoi_volume[~np.isnan(voronoi_volume)]

kde = gaussian_kde(voronoi_volume)
x_vals = np.linspace(voronoi_volume.min(),
                     voronoi_volume.max(), 200)
pdf_vals = kde(x_vals)

plt.figure(figsize=(6, 4))
plt.loglog(x_vals, pdf_vals, linestyle='-', label='aligned with g')

plt.xlabel('Voronoi volume')
plt.ylabel('PDF')
plt.title('Voronoi volume PDF')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.ylim(5e-5, 1e0)

case = 'TTI_no_gravity'
filename = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/{case}_tracks_with_voronoi.parquet'
# Adjust as needed
output_path = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/'

df = pd.read_parquet(filename)
voronoi_volume = df[['voronoi_volume_normalized0', 'voronoi_volume_normalized1',
                     'voronoi_volume_normalized2', 'voronoi_volume_normalized3']].dropna().values.flatten()
voronoi_volume = voronoi_volume[~np.isnan(voronoi_volume)]

kde = gaussian_kde(voronoi_volume)
x_vals = np.linspace(voronoi_volume.min(),
                     voronoi_volume.max(), 200)
pdf_vals = kde(x_vals)

plt.loglog(x_vals, pdf_vals, linestyle='-', label='no g')

case = 'TTI_opposing_gravity'
filename = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/{case}_tracks_with_voronoi.parquet'
# Adjust as needed
output_path = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case}/'

df = pd.read_parquet(filename)
voronoi_volume = df[['voronoi_volume_normalized0', 'voronoi_volume_normalized1',
                     'voronoi_volume_normalized2', 'voronoi_volume_normalized3']].dropna().values.flatten()
voronoi_volume = voronoi_volume[~np.isnan(voronoi_volume)]

kde = gaussian_kde(voronoi_volume)
x_vals = np.linspace(voronoi_volume.min(),
                     voronoi_volume.max(), 200)
pdf_vals = kde(x_vals)

plt.loglog(x_vals, pdf_vals, linestyle='-', label='opposing g')


rpp_approx = 3125/24 * x_vals ** 4 * np.exp(-5*x_vals)
# rpp_approx = 343/15 * np.sqrt(7/(2*np.pi)) * \
# x_vals ** 5/2 * np.exp(-7/2 * x_vals)
plt.loglog(x_vals, rpp_approx, linestyle='--', label='RPP')

# Ensure output directory exists and save
Path(output_path).mkdir(parents=True, exist_ok=True)
out_file = Path(output_path) / 'voronoi_volume_pdf.png'
plt.savefig(out_file, dpi=200, bbox_inches='tight')
plt.xlabel('Voronoi volume')
plt.ylabel('PDF')
plt.title('Voronoi volume PDF')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.ylim(5e-5, 1e0)
plt.legend()
plt.show()

# %%
