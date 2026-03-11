# %% Stereo match property accuracy — cleaned
# Mass-based filtering (mass combines diameter and intensity). Three rules: CV, PDF percentile range, gap.
# Plots: one-frame matches; gap PDF (Group A, Group B); percentile range PDF (Group A, Group B);
#        PDF overview by rule; matches discarded per rule.

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

os.chdir('/workspaces/4d-ptv-mcflow')

DATA_PATH = 'data/tracking_test'
STM_FILE = os.path.join(DATA_PATH, 'rays_out_cpp.h5')
N_CAMS = 4
CAMERA_GROUP_A = [0, 1]
CAMERA_GROUP_B = [2, 3]

# Thresholds (mass-based criterion)
CV_THRESHOLD_PCT = 10.0
PDF_PERCENTILE_RANGE_MAX = 10.0
GAP_PERCENTILE_KEEP = 90.0

# %% Load dataset
with h5py.File(STM_FILE, 'r') as f:
    frame_keys = sorted(
        [k for k in f.keys() if k.startswith('frame')],
        key=lambda k: int(k.replace('frame', '') or 0),
    )
    all_mass = [f[fk]['mass'][()] for fk in frame_keys]
    all_diameter = [f[fk]['diameter'][()] for fk in frame_keys]
    all_intensity = [f[fk]['intensity'][()] for fk in frame_keys]

mass = np.concatenate(all_mass, axis=0)
diameter = np.concatenate(all_diameter, axis=0)
intensity = np.concatenate(all_intensity, axis=0)
frame_len = [m.shape[0] for m in all_mass]
frame_idx = np.repeat(np.arange(len(frame_keys)), frame_len)
n_matches = mass.shape[0]
print(f'Loaded {len(frame_keys)} frames, {n_matches} total matches')

# %% Helpers and pass masks
n_valid_a = np.sum(np.isfinite(mass[:, CAMERA_GROUP_A]), axis=1) >= 2
n_valid_b = np.sum(np.isfinite(mass[:, CAMERA_GROUP_B]), axis=1) >= 2


def cv_pct_subset(arr, cols):
    sub = arr[:, cols]
    mean = np.nanmean(sub, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        std = np.nanstd(sub, axis=1, ddof=1)
        cv = np.where(mean != 0, std / np.abs(mean) * 100, np.nan)
    return cv


def percentile_rank_per_camera(arr):
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for c in range(arr.shape[1]):
        col = arr[:, c]
        valid = np.isfinite(col)
        ref = np.sort(col[valid])
        n_ref = len(ref)
        if n_ref == 0:
            continue
        idx = np.searchsorted(ref, col, side='left')
        out[:, c] = np.where(valid, np.clip(idx / n_ref * 100, 0, 100), np.nan)
    return out


def percentile_range_in_groups(percentile_arr, group_a, group_b):
    n = percentile_arr.shape[0]
    range_a = np.full(n, np.nan)
    range_b = np.full(n, np.nan)
    for i in range(n):
        pa = percentile_arr[i, group_a]
        pa = pa[np.isfinite(pa)]
        if len(pa) >= 2:
            range_a[i] = np.ptp(pa)
        pb = percentile_arr[i, group_b]
        pb = pb[np.isfinite(pb)]
        if len(pb) >= 2:
            range_b[i] = np.ptp(pb)
    return range_a, range_b


def within_group_gap(arr, group):
    sub = arr[:, group]
    n_valid = np.sum(np.isfinite(sub), axis=1)
    gap = np.nanmax(sub, axis=1) - np.nanmin(sub, axis=1)
    return np.where(n_valid >= 2, gap, np.nan)


def gap_threshold_and_pass(gap_arr, valid_mask, percentile_keep):
    vals = gap_arr[valid_mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.zeros_like(gap_arr, dtype=bool)
    thresh = np.percentile(vals, percentile_keep)
    pass_mask = valid_mask & np.isfinite(gap_arr) & (gap_arr <= thresh)
    return thresh, pass_mask


# Rule 1: mass CV ≤ threshold
cv_mass_a = cv_pct_subset(mass, CAMERA_GROUP_A)
cv_mass_b = cv_pct_subset(mass, CAMERA_GROUP_B)
pass_cv_mass = (
    (~n_valid_a | (np.isfinite(cv_mass_a) & (cv_mass_a <= CV_THRESHOLD_PCT)))
    & (~n_valid_b | (np.isfinite(cv_mass_b) & (cv_mass_b <= CV_THRESHOLD_PCT)))
)

# Rule 2: mass percentile range ≤ threshold
pct_mass = percentile_rank_per_camera(mass)
range_mass_a, range_mass_b = percentile_range_in_groups(
    pct_mass, CAMERA_GROUP_A, CAMERA_GROUP_B)
pass_mass_a = ~n_valid_a | (np.isfinite(range_mass_a) & (
    range_mass_a <= PDF_PERCENTILE_RANGE_MAX))
pass_mass_b = ~n_valid_b | (np.isfinite(range_mass_b) & (
    range_mass_b <= PDF_PERCENTILE_RANGE_MAX))
pass_mass = pass_mass_a & pass_mass_b

# Rule 3: mass within-group gap ≤ percentile threshold
gap_mass_A = within_group_gap(mass, CAMERA_GROUP_A)
gap_mass_B = within_group_gap(mass, CAMERA_GROUP_B)
valid_gap_A = np.isfinite(gap_mass_A)
valid_gap_B = np.isfinite(gap_mass_B)
thresh_mass_A, pass_gap_mass_A = gap_threshold_and_pass(
    gap_mass_A, valid_gap_A, GAP_PERCENTILE_KEEP)
thresh_mass_B, pass_gap_mass_B = gap_threshold_and_pass(
    gap_mass_B, valid_gap_B, GAP_PERCENTILE_KEEP)
pass_gap_all = pass_gap_mass_A & pass_gap_mass_B

n_pass_cv = np.sum(pass_cv_mass)
n_pass_pdf = np.sum(pass_mass)
n_pass_gap = np.sum(pass_gap_all)

# %% Plot: different matches for one frame (mass, diameter, intensity per camera)
N_MATCHES_TO_SHOW = 6
FRAME_TO_SHOW = len(frame_keys) // 2  # e.g. middle frame
colors_cam = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']

mask_frame = (frame_idx == FRAME_TO_SHOW)
match_indices = np.where(mask_frame)[0]
if len(match_indices) > 0:
    n_valid_f = np.sum(np.isfinite(mass[mask_frame]), axis=1)
    order = np.argsort(-n_valid_f)
    take = min(N_MATCHES_TO_SHOW, len(order))
    selected = match_indices[order[:take]]
    n_show = len(selected)

    fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6), squeeze=False)
    for col, idx in enumerate(selected):
        m_m = mass[idx]
        d_m = diameter[idx]
        i_m = intensity[idx]
        valid = np.isfinite(m_m)
        x = np.arange(N_CAMS)
        for row, (vals, ylabel) in enumerate([
            (m_m, 'Mass'),
            (d_m, 'Diameter'),
            (i_m, 'Intensity'),
        ]):
            ax = axes[row, col]
            ax.bar(x[valid], vals[valid], color=[colors_cam[c] for c in np.where(valid)[0]],
                   edgecolor='black', linewidth=0.5)
            ax.axvline(1.5, color='gray', linestyle='-',
                       linewidth=0.5, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f'C{c}' for c in range(N_CAMS)])
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f'Match {idx}')
            ax.grid(axis='y', alpha=0.3)
    plt.suptitle(
        f'Frame {FRAME_TO_SHOW} ({frame_keys[FRAME_TO_SHOW]}): mass, diameter, intensity per camera ({n_show} matches)')
    plt.tight_layout()
    plt.show()

# %% Gap PDF: two separate plots (Group A, Group B) with cutoff
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, gap_vals, valid_mask, thresh, label, color in [
    (axes[0], gap_mass_A, valid_gap_A,
     thresh_mass_A, 'Group A (cams 0–1)', '#27ae60'),
    (axes[1], gap_mass_B, valid_gap_B,
     thresh_mass_B, 'Group B (cams 2–3)', '#c0392b'),
]:
    vals = gap_vals[valid_mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) > 0:
        ax.hist(vals, bins=50, density=True, color=color,
                alpha=0.7, edgecolor='white', label=label)
    ax.axvline(thresh, color='black', linestyle='--',
               linewidth=2, label=f'cutoff = {thresh:.3g}')
    ax.axvspan(0, thresh, alpha=0.15, color='green')
    ax.set_xlabel('Within-group mass gap')
    ax.set_ylabel('Probability density')
    ax.set_title(f'Mass gap — {label}')
    ax.legend()
    ax.grid(alpha=0.3)
plt.suptitle(
    f'Gap rule: keep if gap ≤ {GAP_PERCENTILE_KEEP}th percentile (per group)')
plt.tight_layout()
plt.show()

# %% Percentile range PDF: two separate plots (Group A, Group B) with cutoff
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, range_vals, label, color in [
    (axes[0], range_mass_a[n_valid_a], 'Group A (cams 0–1)', '#27ae60'),
    (axes[1], range_mass_b[n_valid_b], 'Group B (cams 2–3)', '#c0392b'),
]:
    vals = range_vals[np.isfinite(range_vals)]
    if len(vals) > 0:
        ax.hist(vals, bins=50, density=True, color=color,
                alpha=0.7, edgecolor='white', label=label)
    ax.axvline(PDF_PERCENTILE_RANGE_MAX, color='black', linestyle='--', linewidth=2,
               label=f'cutoff = {PDF_PERCENTILE_RANGE_MAX}')
    ax.axvspan(0, PDF_PERCENTILE_RANGE_MAX, alpha=0.15, color='green')
    ax.set_xlabel('Mass percentile range within group (0–100)')
    ax.set_ylabel('Probability density')
    ax.set_title(f'PDF rule — {label}')
    ax.legend()
    ax.grid(alpha=0.3)
plt.suptitle(
    f'PDF rule: keep if percentile range ≤ {PDF_PERCENTILE_RANGE_MAX}')
plt.tight_layout()
plt.show()

# %% Overview: PDF of mass, diameter, intensity — unfiltered vs each rule


def mean_per_match(arr):
    return np.nanmean(arr, axis=1)


mean_m = mean_per_match(mass)
mean_d = mean_per_match(diameter)
mean_i = mean_per_match(intensity)

rules = [
    (np.ones(n_matches, dtype=bool), 'Unfiltered', 'gray'),
    (pass_cv_mass, f'CV ≤ {CV_THRESHOLD_PCT}%', 'steelblue'),
    (pass_mass, f'PDF range ≤ {PDF_PERCENTILE_RANGE_MAX}', 'coral'),
    (pass_gap_all, f'Gap ≤ {GAP_PERCENTILE_KEEP}pct', 'seagreen'),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, mean_vals, xlabel, title in [
    (axes[0], mean_m, 'Mass (mean over cams)', 'Mass'),
    (axes[1], mean_d, 'Diameter (mean over cams)', 'Diameter'),
    (axes[2], mean_i, 'Intensity (mean over cams)', 'Intensity'),
]:
    for mask, rule_label, color in rules:
        vals = mean_vals[mask & np.isfinite(mean_vals)]
        if len(vals) > 0:
            ax.hist(vals, bins=40, density=True, alpha=0.5, color=color,
                    label=f'{rule_label} (n={len(vals)})', edgecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
plt.suptitle(
    'PDF of mass, diameter, intensity: effect of each filter (mass-based rules)')
plt.tight_layout()
plt.show()

# %% Overview: how many matches discarded by each rule (no combinations)
n_discard_cv = n_matches - n_pass_cv
n_discard_pdf = n_matches - n_pass_pdf
n_discard_gap = n_matches - n_pass_gap

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, [n_pass_cv, n_pass_pdf, n_pass_gap],
       width, label='Pass', color='steelblue', edgecolor='black')
ax.bar(x + width/2, [n_discard_cv, n_discard_pdf, n_discard_gap],
       width, label='Discard', color='coral', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels([f'CV ≤ {CV_THRESHOLD_PCT}%',
                   f'PDF range ≤ {PDF_PERCENTILE_RANGE_MAX}', f'Gap ≤ {GAP_PERCENTILE_KEEP}pct'])
ax.set_ylabel('Number of matches')
ax.set_title('Matches passing vs discarded by each rule (mass-based)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print('--- Matches per rule ---')
print(f'  Total: {n_matches}')
print(
    f'  Rule 1 (CV ≤ {CV_THRESHOLD_PCT}%):     pass {n_pass_cv}, discard {n_discard_cv}')
print(
    f'  Rule 2 (PDF range ≤ {PDF_PERCENTILE_RANGE_MAX}): pass {n_pass_pdf}, discard {n_discard_pdf}')
print(
    f'  Rule 3 (Gap ≤ {GAP_PERCENTILE_KEEP}pct):       pass {n_pass_gap}, discard {n_discard_gap}')
