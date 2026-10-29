# %%
"""
Enhanced center finding (bandpass → p97 threshold → multi-scale DoG → refine).

Mirrors Processing Steps/02_center_finding.py, then benchmarks against trackpy
on a short clip. Full frames are processed; comparisons are shown in a
128×128 center window.

Run:
  cd /workspaces/4d-ptv-mcflow
  MPLBACKEND=Agg python "Processing Steps/02_center_finding_enhanced.py"
"""
from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use(os.environ.get('MPLBACKEND', 'Agg'))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

os.chdir('/workspaces/4d-ptv-mcflow')

from python_4dptv.center_finding.detect_centers import CenterFinding  # noqa: E402
from python_4dptv.center_finding.detect_centers_enhanced import (  # noqa: E402
    EnhancedCenterFinding,
    plot_center_window_comparison,
    run_trackpy_baseline,
)

# %% CONFIG

RUN_DIR = (
    '/workspaces/4d-ptv-mcflow/raw_data/2025-09-11-ParticleTracking/'
    'TTI_aligned_with_gravity/Run4'
)
CAMERA = 'Camera1'
FILE_PATH = os.path.join(RUN_DIR, f'{CAMERA}.cine')
OUTPUT_PATH = (
    '/workspaces/4d-ptv-mcflow/raw_data/2025-09-11-ParticleTracking/'
    'TTI_aligned_with_gravity/Run4'
)
OUT_DIR = '/workspaces/4d-ptv-mcflow/scripts/out'
os.makedirs(OUT_DIR, exist_ok=True)

PARTICLE_DIAMETER = 7
TRACKPY_THRESHOLD = 2  # same default style as 02_center_finding.py
TRACKPY_MINMASS = 0
TRACKPY_SEPARATION = 3

N_FRAMES = 12
# Skip cine frame 0 (trigger artifact). Process frames 1 .. N_FRAMES inclusive
# → N_FRAMES usable frames on the raw cine index.
FIRST_FRAME = 1
LAST_FRAME = FIRST_FRAME + N_FRAMES  # slice end exclusive
CENTER_WINDOW = 128
DISPLAY_VMAX = 80
COMPARE_FRAME = FIRST_FRAME  # which frame to visualize in the center window

# %% enhanced center finding (full frames)

print(f'Cine: {FILE_PATH}')
print(f'Frames [{FIRST_FRAME}, {LAST_FRAME})  (n={N_FRAMES})')

enhanced = EnhancedCenterFinding(
    FILE_PATH,
    OUTPUT_PATH,
    PARTICLE_DIAMETER,
    params={
        'bp_threshold_percentile': 97.0,
        'blob_method': 'dog',
        'blob_sigmas': (1.0, 1.5, 2.0, 3.0, 4.0),
        'peak_min_distance': TRACKPY_SEPARATION,
        'peak_threshold_rel': 0.12,
        'refine_on': 'raw',
        'refine_radius': 5,
    },
)
# Do not call remove_frames() here — we explicitly start at frame 1 and take 12.

t0 = time.perf_counter()
print('Enhanced detection…')
enhanced.find_centers(FIRST_FRAME, LAST_FRAME)
t_enhanced = time.perf_counter() - t0
print(
    f'Enhanced total: {t_enhanced:.2f}s  '
    f'({np.mean(enhanced.timings):.2f}s/frame)  '
    f'particles={len(enhanced.f)}'
)

# %% trackpy baseline (same frames, full frame)

print('Trackpy baseline…')
t0 = time.perf_counter()
baseline_f, baseline_timings = run_trackpy_baseline(
    enhanced.frames,
    first_frame=FIRST_FRAME,
    last_frame=LAST_FRAME,
    diameter=PARTICLE_DIAMETER,
    threshold=TRACKPY_THRESHOLD,
    minmass=TRACKPY_MINMASS,
    separation=TRACKPY_SEPARATION,
)
t_baseline = time.perf_counter() - t0
print(
    f'Trackpy total: {t_baseline:.2f}s  '
    f'({np.mean(baseline_timings):.2f}s/frame)  '
    f'particles={len(baseline_f)}'
)

# Also run classic CenterFinding path for write compatibility check (optional short)
classic = CenterFinding(FILE_PATH, OUTPUT_PATH, PARTICLE_DIAMETER)
classic.threshold = TRACKPY_THRESHOLD
classic.separation = TRACKPY_SEPARATION
classic.minmass = TRACKPY_MINMASS
# keep classic.f unset unless user wants full write of baseline

# %% write enhanced matches

enhanced.write_matches()

# %% per-frame counts + timing summary

frames_idx = list(range(FIRST_FRAME, LAST_FRAME))
enh_counts = [
    int((enhanced.f['frame'] == fr).sum()) if enhanced.f is not None else 0
    for fr in frames_idx
]
base_counts = [
    int((baseline_f['frame'] == fr).sum()) if baseline_f is not None else 0
    for fr in frames_idx
]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(frames_idx, base_counts, 'o-', color='cyan', label='trackpy')
axes[0].plot(frames_idx, enh_counts, 's-', color='lime', label='enhanced')
axes[0].set_xlabel('frame')
axes[0].set_ylabel('# particles (full frame)')
axes[0].set_title('Detections per frame')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(
    ['trackpy', 'enhanced'],
    [np.mean(baseline_timings), np.mean(enhanced.timings)],
    color=['cyan', 'lime'],
)
axes[1].set_ylabel('seconds / frame')
axes[1].set_title('Mean runtime (full frame)')
for i, v in enumerate([np.mean(baseline_timings), np.mean(enhanced.timings)]):
    axes[1].text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)
fig.suptitle(f'{CAMERA} Run4 — {N_FRAMES} frames benchmark')
fig.tight_layout()
summary_path = os.path.join(OUT_DIR, 'center_finding_enhanced_benchmark_summary.png')
fig.savefig(summary_path, dpi=140, bbox_inches='tight')
print(f'Saved {summary_path}')

# %% center-window comparison (full-frame detections, 128×128 crop view)

frame_img = np.asarray(enhanced.frames[COMPARE_FRAME])
compare_path = os.path.join(OUT_DIR, 'center_finding_enhanced_center_window.png')
fig, stats = plot_center_window_comparison(
    frame_img,
    baseline_f,
    enhanced.f,
    frame_index=COMPARE_FRAME,
    window=CENTER_WINDOW,
    display_vmax=DISPLAY_VMAX,
    out_path=compare_path,
)

# Extra frames for a small strip
strip_frames = frames_idx[:: max(1, len(frames_idx) // 4)][:4]
fig, axes = plt.subplots(2, len(strip_frames), figsize=(3.2 * len(strip_frames), 6.2))
if len(strip_frames) == 1:
    axes = np.array([[axes[0]], [axes[1]]])
for col, fr in enumerate(strip_frames):
    img = np.asarray(enhanced.frames[fr])
    from python_4dptv.center_finding.detect_centers_enhanced import (
        center_roi,
        features_in_roi,
    )
    roi = center_roi(img, CENTER_WINDOW)
    x0, y0, ww, hh = roi
    view = img[y0:y0 + hh, x0:x0 + ww]
    b = features_in_roi(baseline_f, roi, frame=fr)
    e = features_in_roi(enhanced.f, roi, frame=fr)

    ax = axes[0, col]
    ax.imshow(view, cmap='gray', vmin=0, vmax=DISPLAY_VMAX, interpolation='nearest')
    if b is not None and len(b):
        ax.plot(b['x'], b['y'], 'o', mfc='none', mec='cyan', ms=8, mew=1.0)
    ax.set_title(f'trackpy f{fr} n={0 if b is None else len(b)}', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, col]
    ax.imshow(view, cmap='gray', vmin=0, vmax=DISPLAY_VMAX, interpolation='nearest')
    if e is not None and len(e):
        ax.plot(e['x'], e['y'], '+', color='red', ms=7, mew=1.1)
    ax.set_title(f'enhanced f{fr} n={0 if e is None else len(e)}', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(f'{CENTER_WINDOW}×{CENTER_WINDOW} center — trackpy (top) vs enhanced (bottom)')
fig.tight_layout()
strip_path = os.path.join(OUT_DIR, 'center_finding_enhanced_center_strip.png')
fig.savefig(strip_path, dpi=140, bbox_inches='tight')
print(f'Saved {strip_path}')

print('\n=== Benchmark summary ===')
print(f'Frames: {N_FRAMES}  ({FIRST_FRAME}..{LAST_FRAME - 1})')
print(f'Trackpy:  {t_baseline:.2f}s total  {np.mean(baseline_timings):.2f}s/frame  '
      f'n_total={len(baseline_f)}')
print(f'Enhanced: {t_enhanced:.2f}s total  {np.mean(enhanced.timings):.2f}s/frame  '
      f'n_total={len(enhanced.f)}')
print(f'Center window frame {COMPARE_FRAME}: {stats}')
print(f'Enhanced HDF5: {enhanced.output_path}')
