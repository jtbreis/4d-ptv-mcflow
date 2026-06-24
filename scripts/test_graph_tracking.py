# %% [markdown]
# # Graph-based 4-frame tracking (interactive test)
#
# Run cell-by-cell in VS Code / Cursor (Python Interactive) or Jupyter.
# Requires editable install from repo root: ``pip install -e graph_tracking[dev]``

# %%
import os
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from check_tracking import check_tracking_file, print_report
from graph_tracking.assign import greedy_assign_paths
from graph_tracking.link_block import LinkParams, link_block
from graph_tracking.load_data import get_nframes, iter_block_starts, load_block
from graph_tracking.track_data import block_to_tracking_data
from python_4be_eti.utils.basic_utils import write_data_h5

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %%
# --- Paths ---
run_dir = PROJECT_ROOT / (
    "data/Windtunnel_PTV_TNTI/X300cm_PTV_above_TNTI/"
    "TNTI_opposing_gravity/Run1"
)
rays_h5 = run_dir / "rays_out_cpp.h5"
output_h5part = run_dir / "tracks_graph_test.h5part"

# %%
# --- Tracking parameters (same meaning as 05_4frame_tracking.py) ---
params = LinkParams(
    box_size_track=0.2,
    initial_x_lo=0.0,
    initial_x_hi=4.5,
    initial_y_lo=-1.5,
    initial_y_hi=1.5,
    initial_z_lo=-1.5,
    initial_z_hi=1.5,
    max_branch=30,
    max_seeds=0,  # 0 = all frame-0 particles; use a limit for quick tests
    seed_rng=0,
    allow_3_frame=False,
    # merge forward + backward 4-frame graphs before assignment
    use_bidirectional_graph=True,
)

max_blocks = 1
block_start = 0
dt = 1e-3
rep_rate = 10.0

nframes = get_nframes(str(rays_h5))
print(f"Input: {rays_h5}")
print(f"Frames in file: {nframes}")
print(
    f"Will process block(s) starting at: {iter_block_starts(nframes)[:max_blocks]}"
)

# %%
# --- Link one block and assign tracks ---
start = block_start
block = load_block(str(rays_h5), start)
print(f"Block frames {block.frame_indices}")
print(f"Particles per frame: {[f.n for f in block.frames]}")

t0 = time.time()
result = link_block(block, params)
assigned = greedy_assign_paths(result.candidates)
elapsed = time.time() - t0

print(f"Seeds tried:     {result.stats['n_seeds_tried']}")
print(f"Path candidates: {result.stats['n_candidates']}")
print(f"Tracks assigned: {len(assigned)}")
print(
    f"Mean branches/seed (nonzero): {result.stats['mean_branches_per_seed']:.1f}"
)
print(f"Link + assign:   {elapsed:.2f} s")

# %%
# --- Write tracks_graph_test.h5part ---
if output_h5part.is_file():
    output_h5part.unlink()

block_data = block_to_tracking_data(block, assigned)
write_data_h5(
    block_data,
    str(run_dir),
    block.frame_indices,
    output_h5part=str(output_h5part),
    dt=dt,
    rep_rate_hz=rep_rate,
    track_length=4,
    position_scale_to_m=1e-3,
    position_units_str="mm",
    use_bspline=True,
    min_track_length=3,
    verbose=True,
)
print(f"Wrote: {output_h5part}")

# %%
# --- Quality check (same thresholds as check_tracking.py, relaxed for small tests) ---
report = check_tracking_file(
    str(output_h5part),
    min_particles_per_frame=50,
    min_finite_vz=100,
)
print_report(
    {
        "filepath": str(output_h5part),
        "n_frames": report["n_frames"],
        "mean_particles": report["mean_particles"],
        "median_particles": report["median_particles"],
        "min_particles": report["min_particles"],
        "max_particles": report["max_particles"],
        "n_finite_vz": report["n_finite_vz"],
        "vz": np.array([]),
    },
    report,
    min_particles_per_frame=50,
)

# %%
# --- Quick v_z histogram ---
vz_all = []
with h5py.File(output_h5part, "r") as f:
    for key in sorted(k for k in f.keys() if k.startswith("Step#")):
        vz_all.append(f[key]["vz"][:])
vz = np.concatenate(vz_all)
vz = vz[np.isfinite(vz)]

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(vz, bins=50, density=True, alpha=0.7, color="steelblue")
ax.set_xlabel(r"$v_z$ [m/s]")
ax.set_ylabel("PDF")
ax.set_title(f"Graph tracks: {len(assigned)} paths, {vz.size} v_z samples")
fig.tight_layout()
plt.show()

# %%
# --- Optional: inspect a single assigned path ---
if assigned:
    path = assigned[0]
    f0, f1, f2, f3 = block.frames
    i0, i1, i2, i3 = path.indices
    pts = np.array([
        [f0.x[i0], f0.y[i0], f0.z[i0]],
        [f1.x[i1], f1.y[i1], f1.z[i1]],
        [f2.x[i2], f2.y[i2], f2.z[i2]],
        [f3.x[i3], f3.y[i3], f3.z[i3]],
    ])
    print(f"Example track cost={path.cost:.4f}")
    print("Frame  x        y        z")
    for fi, p in zip(block.frame_indices, pts):
        print(f"{fi:5d}  {p[0]:8.3f} {p[1]:8.3f} {p[2]:8.3f}")

# %%
