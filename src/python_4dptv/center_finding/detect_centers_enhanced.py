"""Enhanced center finding: bandpass → percentile threshold → multi-scale DoG/LoG → refine."""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import trackpy as tp
from matplotlib.patches import Circle

from ..io.write_h5 import write_h5
from ..utils.structure import Folders

# Shared preprocessing / blob pipeline (developed in scripts/)
_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts'))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from cine_preprocessing_core import (  # noqa: E402
    DEFAULT_PARAMS,
    detect_blobs_bandpass_threshold,
)


ENHANCED_DEFAULTS: dict[str, Any] = {
    **DEFAULT_PARAMS,
    'bp_threshold_percentile': 97.0,
    'bp_threshold_abs': 0.0,
    'blob_method': 'dog',
    'blob_sigmas': (1.0, 1.5, 2.0, 3.0, 4.0),
    'peak_min_distance': 3,
    'peak_threshold_rel': 0.12,
    'refine_on': 'raw',
    'refine_radius': 5,
}


class EnhancedCenterFinding:
    """Drop-in style alternative to CenterFinding using multi-scale blob detection."""

    def __init__(
        self,
        path: str,
        output_path: str,
        particle_diameter: int = 7,
        *,
        params: dict[str, Any] | None = None,
        output_suffix: str = '_enhanced',
    ):
        self.path = path
        self.frames = pims.open(path)
        os.makedirs(output_path + Folders.CENTERS.value, exist_ok=True)
        base = os.path.splitext(
            output_path + Folders.CENTERS.value + '/' + os.path.basename(path)
        )[0]
        self.output_path = f'{base}{output_suffix}.h5'
        self.particle_diameter = int(particle_diameter)
        self.params = dict(ENHANCED_DEFAULTS)
        if params:
            self.params.update(params)
        self.params['peak_min_distance'] = int(
            self.params.get('peak_min_distance', max(2, particle_diameter // 2))
        )
        self.frames_discarded = False
        self.f: pd.DataFrame | None = None
        self.timings: list[float] = []

    def remove_frames(self, discard_frames: int = 1, tracking_frames: int = 4):
        if self.frames_discarded:
            print('Discard frames have already been removed!')
            return
        total_frames = len(self.frames)
        kept_indices = [
            i for i in range(total_frames)
            if i % (discard_frames + tracking_frames) != (discard_frames - 1)
        ]
        self.frames = [self.frames[i] for i in kept_indices]
        self.frames_discarded = True

    def find_centers(self, first_frame: int = 0, last_frame: int = -1) -> pd.DataFrame:
        """Detect particles on full frames with the enhanced pipeline."""
        frames = self.frames[first_frame:last_frame]
        rows: list[pd.DataFrame] = []
        self.timings = []

        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            img = np.asarray(frame)
            feats, _steps, _sigma = detect_blobs_bandpass_threshold(img, self.params)
            elapsed = time.perf_counter() - t0
            self.timings.append(elapsed)

            if feats is None or not len(feats):
                continue
            out = feats.copy()
            out['frame'] = first_frame + i
            # Compatibility columns with trackpy / write_matches
            if 'size' not in out.columns and 'sigma' in out.columns:
                out['size'] = out['sigma']
            if 'signal' not in out.columns and 'mass' in out.columns:
                out['signal'] = out['mass']
            rows.append(out)

            print(
                f'  enhanced frame {first_frame + i}: n={len(out)}  '
                f'({elapsed:.2f}s)',
                flush=True,
            )

        if rows:
            self.f = pd.concat(rows, ignore_index=True)
        else:
            self.f = pd.DataFrame(
                columns=['x', 'y', 'mass', 'size', 'signal', 'sigma', 'frame']
            )
        return self.f

    def write_matches(self):
        if self.f is None:
            raise RuntimeError('Call find_centers() before write_matches()')
        df = self.f.copy()
        if 'size' in df.columns:
            df['diameter'] = 2 * df['size']
        if 'signal' in df.columns:
            df['intensity'] = df['signal']
        grouped = df.groupby('frame')
        data = {frame: group.reset_index(drop=True) for frame, group in grouped}
        metadata = {
            'method': 'bandpass_threshold_multiscale_dog',
            'particle diameter': self.particle_diameter,
            **{k: (list(v) if isinstance(v, tuple) else v)
               for k, v in self.params.items()
               if k in (
                   'bp_threshold_percentile', 'bp_threshold_abs', 'blob_method',
                   'blob_sigmas', 'peak_min_distance', 'peak_threshold_rel',
                   'refine_on', 'refine_radius', 'bandpass_lshort', 'bandpass_llong',
               )},
        }
        write_h5(filename=self.output_path, data=data, metadata=metadata)
        print(f'Wrote {self.output_path}')

    def check_mass_distribution(self):
        if self.f is None or not len(self.f):
            print('No particles to plot')
            return
        fig, ax = plt.subplots()
        ax.hist(self.f['mass'], bins=300)
        ax.set(xlabel='mass / peak', ylabel='count')
        plt.show()


def center_roi(frame: np.ndarray, window: int = 128) -> list[int]:
    h, w = frame.shape[:2]
    half = window // 2
    x0 = max(0, w // 2 - half)
    y0 = max(0, h // 2 - half)
    return [x0, y0, window, window]


def features_in_roi(features: pd.DataFrame, roi: list[int], frame: int | None = None):
    if features is None or not len(features):
        return features
    x0, y0, ww, hh = roi
    mask = (
        (features['x'] >= x0) & (features['x'] < x0 + ww)
        & (features['y'] >= y0) & (features['y'] < y0 + hh)
    )
    if frame is not None and 'frame' in features.columns:
        mask &= features['frame'] == frame
    out = features.loc[mask].copy()
    out['x'] = out['x'] - x0
    out['y'] = out['y'] - y0
    return out


def run_trackpy_baseline(
    frames,
    *,
    first_frame: int,
    last_frame: int,
    diameter: int,
    threshold: int,
    minmass: int,
    separation: int,
) -> tuple[pd.DataFrame, list[float]]:
    """Per-frame trackpy.locate for timing parity with enhanced finder."""
    subset = frames[first_frame:last_frame]
    rows = []
    timings = []
    for i, frame in enumerate(subset):
        t0 = time.perf_counter()
        feats = tp.locate(
            np.asarray(frame),
            diameter=diameter,
            threshold=threshold,
            minmass=minmass,
            separation=separation,
            preprocess=True,
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        if feats is not None and len(feats):
            out = feats.copy()
            out['frame'] = first_frame + i
            rows.append(out)
        print(
            f'  trackpy frame {first_frame + i}: n={0 if feats is None else len(feats)}  '
            f'({elapsed:.2f}s)',
            flush=True,
        )
    if rows:
        return pd.concat(rows, ignore_index=True), timings
    empty = pd.DataFrame(columns=['x', 'y', 'mass', 'size', 'signal', 'frame'])
    return empty, timings


def plot_center_window_comparison(
    frame: np.ndarray,
    baseline: pd.DataFrame,
    enhanced: pd.DataFrame,
    *,
    frame_index: int,
    window: int = 128,
    display_vmax: float = 80,
    out_path: str | None = None,
):
    """Side-by-side raw overlays + match/diff view in the center window."""
    roi = center_roi(frame, window)
    x0, y0, ww, hh = roi
    view = np.asarray(frame)[y0:y0 + hh, x0:x0 + ww]

    base_roi = features_in_roi(baseline, roi, frame=frame_index)
    enh_roi = features_in_roi(enhanced, roi, frame=frame_index)

    # Nearest-neighbor matching for agreement stats (px)
    match_radius = 3.0
    only_base = only_enh = matched = 0
    matched_pairs: list[tuple[float, float, float, float]] = []
    if base_roi is not None and len(base_roi) and enh_roi is not None and len(enh_roi):
        from scipy.spatial import cKDTree
        bxy = base_roi[['x', 'y']].to_numpy()
        exy = enh_roi[['x', 'y']].to_numpy()
        dist, idx = cKDTree(exy).query(bxy, distance_upper_bound=match_radius)
        used = set()
        for i, (d, j) in enumerate(zip(dist, idx)):
            if np.isfinite(d) and j < len(exy) and j not in used:
                matched += 1
                used.add(j)
                matched_pairs.append((bxy[i, 0], bxy[i, 1], exy[j, 0], exy[j, 1]))
            else:
                only_base += 1
        only_enh = len(exy) - len(used)
    else:
        only_base = 0 if base_roi is None else len(base_roi)
        only_enh = 0 if enh_roi is None else len(enh_roi)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    ax = axes[0]
    ax.imshow(view, cmap='gray', vmin=0, vmax=display_vmax, interpolation='nearest')
    if base_roi is not None and len(base_roi):
        ax.plot(base_roi['x'], base_roi['y'], 'o', mfc='none', mec='cyan',
                ms=9, mew=1.2, label=f'trackpy n={len(base_roi)}')
    ax.set_title(f'trackpy (n={0 if base_roi is None else len(base_roi)})')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1]
    ax.imshow(view, cmap='gray', vmin=0, vmax=display_vmax, interpolation='nearest')
    if enh_roi is not None and len(enh_roi):
        for _, r in enh_roi.iterrows():
            sigma = float(r['sigma']) if 'sigma' in enh_roi.columns else 2.0
            rad = max(2.5, 1.6 * sigma)
            ax.add_patch(Circle((r['x'], r['y']), rad, fill=False,
                                edgecolor='lime', linewidth=1.1))
            ax.plot(r['x'], r['y'], '+', color='red', ms=6, mew=1.0)
    ax.set_title(f'enhanced (n={0 if enh_roi is None else len(enh_roi)})')
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    ax.imshow(view, cmap='gray', vmin=0, vmax=display_vmax, interpolation='nearest')
    if base_roi is not None and len(base_roi):
        ax.plot(base_roi['x'], base_roi['y'], 'o', mfc='none', mec='cyan',
                ms=10, mew=1.0, label='trackpy only / both')
    if enh_roi is not None and len(enh_roi):
        ax.plot(enh_roi['x'], enh_roi['y'], '+', color='red', ms=8, mew=1.2,
                label='enhanced only / both')
    for bx, by, ex, ey in matched_pairs:
        ax.plot([bx, ex], [by, ey], '-', color='yellow', lw=0.8, alpha=0.8)
    ax.set_title(
        f'overlay  match={matched}  only_tp={only_base}  only_enh={only_enh}'
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=7)

    fig.suptitle(
        f'Frame {frame_index} — {window}×{window} center window  '
        f'(ROI x0={x0}, y0={y0})',
        fontsize=11,
    )
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved {out_path}')
    return fig, {
        'roi': roi,
        'n_trackpy': 0 if base_roi is None else len(base_roi),
        'n_enhanced': 0 if enh_roi is None else len(enh_roi),
        'matched': matched,
        'only_trackpy': only_base,
        'only_enhanced': only_enh,
    }
