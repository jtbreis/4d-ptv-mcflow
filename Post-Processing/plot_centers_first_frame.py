# %%
"""
Interactive / Jupyter-style script: load first frame of centers from Centers/Camera*.h5 and plot.

Set ROTATION to 'cw' / 'ccw' (one 90° turn) or an int (repeated 90°, + = counterclockwise).

Run cells in VS Code / Cursor, or: python plot_centers_first_frame.py
"""
from python_4dptv.io.read_h5 import iter_read_h5_centers
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# %%

# %%
H5_PATH = os.path.join(
    PROJECT_ROOT,
    'raw_data/low_threshold/PTV_above/TTI_aligned_with_gravity/Run1/Centers/Camera4.h5',
)
# Set to a .cine path to overlay frame 0; keep None for scatter-only.
CINE_PATH = None
# Set to a path to save PNG; keep None to call plt.show() instead.
OUTPUT_PATH = None
# View rotation: 0 for none; 'cw' / 'ccw' for one 90° turn; or an int (90° steps,
# positive = counterclockwise, same sign as numpy.rot90).
ROTATION = 'ccw'

# %%
frame0 = next(iter_read_h5_centers(H5_PATH))
x = np.asarray(frame0['x'], dtype=float)
y = np.asarray(frame0['y'], dtype=float)
print(f'Particles in first frame: {len(x)}')

# %%


def load_cine_frame0(cine_path):
    import pims

    vid = pims.open(cine_path)
    return np.asarray(vid[0])


def rotation_to_rot90_k(rotation):
    """Return k for numpy.rot90 (positive = counterclockwise)."""
    if rotation in (None, 0, 'none', 'None'):
        return 0
    if isinstance(rotation, (int, np.integer)):
        return int(rotation)
    s = str(rotation).lower().strip()
    if s == 'cw':
        return -1
    if s == 'ccw':
        return 1
    raise ValueError(
        "ROTATION must be 0, 'cw', 'ccw', or an int (90° steps, + = CCW)"
    )


def rot90_xy_with_canvas(x, y, width, height, k):
    """
    Apply k CCW 90° rotations to image pixel coords (x = column, y = row, origin top-left).
    width, height are the canvas size before this rotation (matches image shape[1], shape[0]).
    """
    k = int(k) % 4
    if k == 0:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    W, H = float(width), float(height)
    for _ in range(k):
        xn = y
        yn = (W - 1.0) - x
        x, y = xn, yn
        W, H = H, W
    return x, y


fig, ax = plt.subplots(figsize=(10, 8))
k_rot = rotation_to_rot90_k(ROTATION)

if CINE_PATH:
    img = load_cine_frame0(CINE_PATH)
    h0, w0 = img.shape[0], img.shape[1]
    img = np.rot90(img, k=k_rot)
    xr, yr = rot90_xy_with_canvas(x, y, w0, h0, k_rot)
    ax.imshow(img, cmap='gray', vmin=0, vmax=20)
    ax.scatter(xr, yr, s=8, c='cyan', edgecolors='darkblue',
               linewidths=0.2, alpha=0.85)
else:
    if len(x):
        w0 = float(np.nanmax(x)) + 1.0
        h0 = float(np.nanmax(y)) + 1.0
    else:
        w0, h0 = 1.0, 1.0
    xr, yr = rot90_xy_with_canvas(x, y, w0, h0, k_rot)
    ax.scatter(xr, yr, s=6, c='tab:blue', alpha=0.6)
    ax.set_aspect('equal')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.invert_yaxis()

ax.set_title(
    f'First frame centers: {os.path.basename(H5_PATH)}  (N = {len(x)})')
plt.tight_layout()

# %%
if OUTPUT_PATH:
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f'Saved {OUTPUT_PATH}')
else:
    plt.show()
