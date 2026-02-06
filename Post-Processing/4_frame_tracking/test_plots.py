# %%
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from mcflow_plotting.turbulence.pdf import plot_pdf
from IPython import get_ipython
from pathlib import Path
import re

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

# we know the vertical velocity column is VZ_0
vz_col = 'vy_0'

# select paired data
mask = df[vz_col].notna() & df['voronoi_volume_normalized0'].notna()
vol = df.loc[mask, 'voronoi_volume_normalized0'].values
vz = df.loc[mask, vz_col].values

# plot
plt.figure(figsize=(8, 5))
plt.hist2d(vol, vz, bins=100, cmap='viridis', norm='log')
plt.colorbar(label='Count')
plt.xlabel('Voronoi volume (normalized)')
plt.ylabel(f'Vertical velocity ({vz_col})')
plt.title('Vertical velocity vs Voronoi volume')
# plt.yscale('log')
plt.xscale('log')
plt.grid(alpha=0.3)

# save figure
out_file = Path(output_path) / 'vertical_velocity_vs_volume.png'
plt.tight_layout()
plt.savefig(out_file, dpi=150)
print(f"Saved plot to {out_file}")
plt.show()

# %%
# SVD / PCA on (log10(volume), vertical velocity)

# require positive volumes for log transform
pos_mask = (vol > 0) & np.isfinite(vol) & np.isfinite(vz)
vol_p = vol[pos_mask]
vz_p = vz[pos_mask]

if vol_p.size < 2:
    raise ValueError("Not enough positive finite samples for SVD.")

log_vol = np.log10(vol_p)
X = np.vstack([log_vol, vz_p]).T
n_samples = X.shape[0]

# center
mean_X = X.mean(axis=0)
Xc = X - mean_X

# SVD
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
# eigenvalues of covariance matrix
eigvals = (s ** 2) / (n_samples - 1)
explained_ratio = eigvals / eigvals.sum()

print("Singular values:", s)
print("Eigenvalues (cov):", eigvals)
print("Explained variance ratio:", explained_ratio)
print("Principal directions (rows = principal components):\n", Vt)

# save numeric results
out_npz = Path(output_path) / "svd_results.npz"
np.savez(out_npz, singular_values=s, components=Vt, mean=mean_X,
         eigvals=eigvals, explained_ratio=explained_ratio)
print(f"Saved SVD results to {out_npz}")

# quick visualization: scatter in (log10(volume), vz) with principal component lines
plt.figure(figsize=(7, 6))
plt.scatter(log_vol, vz_p, s=6, alpha=0.3)
origin = mean_X
t = np.linspace(-3, 3, 100)  # parameter along PC direction (in standard units)
colors = ["C1", "C2"]
for i in range(min(2, Vt.shape[0])):
    vec = Vt[i]
    # scale line length by sqrt(eigenvalue) for visibility
    length = 3 * np.sqrt(eigvals[i])
    line = origin + np.outer(t * length, vec)
    plt.plot(line[:, 0], line[:, 1], color=colors[i], lw=2,
             label=f"PC{i+1} ({explained_ratio[i]*100:.1f}%)")

plt.xlabel("log10(Voronoi volume)")
plt.ylabel(f"Vertical velocity ({vz_col})")
plt.title("SVD / PCA on (log10(volume), vertical velocity)")
plt.legend()
plt.grid(alpha=0.3)
out_img = Path(output_path) / "svd_pcs.png"
plt.tight_layout()
plt.savefig(out_img, dpi=150)
print(f"Saved SVD plot to {out_img}")
plt.show()
# %%

# prepare regression dataset: X_train = all numeric columns except the vertical velocity (vz_col)
num_df = df.select_dtypes(include=[np.number])

if vz_col not in num_df.columns:
    raise KeyError(
        f"Target column {vz_col!r} not found among numeric columns.")

all_features = [c for c in num_df.columns if c !=
                vz_col and c != 'vy_1' and c != 'vy_2']
if len(all_features) == 0:
    raise ValueError("No features available after dropping the target column.")

# build dataframe with no missing values in the selected features + target
reg_df = num_df[all_features + [vz_col]
                ].dropna(axis=0, how='any').reset_index(drop=True)

X_train = reg_df[all_features].values
y_train = reg_df[vz_col].values

# Compute regression coefficients using SVD
U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
X_inv = np.dot(Vt.T, np.dot(np.diag(1/S), U.T))
w = np.dot(X_inv, y_train)
# Print results
figure = plt.figure(figsize=(12, 6))
print('Regression coefficients:', w)
plt.bar(all_features, w)
plt.xlabel('Feature index')
plt.ylabel('Weight value')
plt.tight_layout()

# %%
