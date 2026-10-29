# %%
"""
Explore image preprocessing on frame 1 from .cine files.

Frame 0 is discarded (camera triggering artifact). All analysis uses frame 1.
Particle detection is inspected in a 128×128 window centered on the image.

Typical workflows:
  Bandpass → hard threshold (default):
    trackpy bandpass, then zero pixels below a percentile/absolute floor
  Also: bandpass → morph-open / rolling-ball, particle_snr, Sobel→binary→expansion, etc.
Set METHOD = 'bandpass_threshold' (default). Tune bp_threshold_percentile (≈90).
Use 'bandpass_rolling_ball' for rolling-ball (slower on full frames).

Run cell-by-cell in VS Code / Cursor (Python: Run Cell), or:
  cd /workspaces/4d-ptv-mcflow && python scripts/explore_cine_preprocessing.py

Web UI (standalone app under ptv_weblauncher, port 5001):
  ./ptv_weblauncher/cine_preprocessing/run.sh
  # or: cd ptv_weblauncher/cine_preprocessing && REPO_ROOT=/workspaces/4d-ptv-mcflow docker compose up --build
  # open http://localhost:5001

Adjust the CONFIG cell, then re-run downstream cells.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pims
import trackpy as tp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

try:
    from ptv_calib.preprocessing.filter_images import fft_filter
except ImportError:
    fft_filter = None

from cine_preprocessing_core import (  # noqa: E402
    DEFAULT_PARAMS as CORE_DEFAULT_PARAMS,
    apply_bandpass_then_bg,
    apply_bandpass_threshold,
    apply_particle_snr_pipeline,
    bandpass_then_bg_steps,
    bandpass_threshold_steps,
    build_bandpass_then_bg_panels,
    build_bandpass_threshold_panels,
    detect_blobs_bandpass_threshold,
    detect_local_maxima,
    particle_snr_pipeline_steps,
)
# %% CONFIG — edit these, then re-run cells below

RUN_DIR = (
    '/workspaces/4d-ptv-mcflow/raw_data/2025-09-11-ParticleTracking/'
    'TTI_aligned_with_gravity/Run4'
)
# first usable frame (frame 0 is discarded — triggering artifact)
FRAME_INDEX = 1
CAMERA = 'Camera1'  # or None to load all Camera*.cine files
DISPLAY_VMAX = 80  # raw frame display scale (particles are very dim)
CENTER_WINDOW = 128  # inspect particles in a square window around image center
# Optional manual override: [x0, y0, width, height]. None = auto center window above.
ROI = None

# trackpy preview (set diameter=None to skip detection overlay)
PARTICLE_DIAMETER = 7
TRACKPY_THRESHOLD = 10
TRACKPY_MINMASS = 0

# preprocessing parameters used in comparison grid
PREPROCESS_PARAMS = dict(CORE_DEFAULT_PARAMS)

# background removal applied after percentile stretch (see stretch → bg grid cell)
# rolling_ball | morph_open | tophat | gaussian | median | bandpass
BG_REMOVAL = 'rolling_ball'
# primary pipeline for the tune / detection cells
METHOD = 'bandpass_threshold'
# particle_snr / bandpass_then_bg knobs (override CORE_DEFAULT_PARAMS):
PREPROCESS_PARAMS.update({
    'bg_method': 'rolling_ball',       # rolling_ball | morph_open | median | gaussian
    'denoise_method': 'nlm',           # nlm | bilateral | wavelet | none
    'blob_method': 'dog',              # dog | log
    'blob_sigmas': (1.0, 1.5, 2.0, 3.0, 4.0),  # in-focus → defocused
    'peak_min_distance': 3,
    'peak_threshold_rel': 0.12,
    'bandpass_bg_method': 'morph_open',  # used by METHOD='bandpass_then_bg'
    # Zero residual mottled floor after bandpass (particles stay bright)
    'bp_threshold_percentile': 97.0,   # try 85–97; set 0 and use abs below
    'bp_threshold_abs': 0.0,           # if >0, absolute intensity cutoff instead
    'refine_on': 'raw',                # thr locates particle; COM on raw centers it
    'refine_radius': 5,
})
USE_LOCAL_MAXIMA = True  # multi-scale DoG/LoG peaks for bandpass_threshold

# %% helpers


def discover_cine_files(run_dir: str) -> list[str]:
    def _camera_sort(path: str) -> tuple[int, str]:
        match = re.search(r'Camera(\d+)', os.path.basename(path), re.I)
        return (int(match.group(1)), path) if match else (0, path)

    paths = sorted(glob.glob(os.path.join(
        run_dir, 'Camera*.cine')), key=_camera_sort)
    if not paths:
        raise FileNotFoundError(f'No Camera*.cine files found in {run_dir}')
    return paths


def load_cine_frame(cine_path: str, frame_index: int) -> np.ndarray:
    vid = pims.open(cine_path)
    n_frames = len(vid)
    if frame_index < 0 or frame_index >= n_frames:
        raise IndexError(
            f'Frame {frame_index} out of range for {cine_path} ({n_frames} frames)')
    return np.asarray(vid[frame_index])


def as_u8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, (0.5, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (img - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def center_roi(img: np.ndarray, size: int) -> list[int]:
    """Return [x0, y0, width, height] for a size×size window centered on the image."""
    h, w = img.shape[:2]
    half = size // 2
    cx, cy = w // 2, h // 2
    x0 = max(0, min(cx - half, w - size))
    y0 = max(0, min(cy - half, h - size))
    return [x0, y0, size, size]


def crop_roi(img: np.ndarray, roi: list[int]) -> np.ndarray:
    x0, y0, w, h = roi
    return img[y0:y0 + h, x0:x0 + w]


def particles_in_roi(features, roi: list[int]):
    """Keep trackpy features whose (x, y) fall inside roi [x0, y0, w, h]."""
    if features is None or not len(features):
        return features
    x0, y0, w, h = roi
    mask = (
        (features['x'] >= x0) & (features['x'] < x0 + w)
        & (features['y'] >= y0) & (features['y'] < y0 + h)
    )
    return features[mask].copy()


def shift_features_to_crop(features, roi: list[int]):
    """Express particle coordinates relative to the crop origin."""
    if features is None or not len(features):
        return features
    out = features.copy()
    out['x'] = out['x'] - roi[0]
    out['y'] = out['y'] - roi[1]
    return out


def show_image(ax, img, title: str, vmax: float | None = None, cmap: str = 'gray'):
    data = np.asarray(img)
    kwargs = {'cmap': cmap}
    if vmax is not None:
        kwargs['vmin'] = 0
        kwargs['vmax'] = vmax
    ax.imshow(data, **kwargs)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def gradient_magnitude(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Local gradient magnitude |∇I| via Sobel filters."""
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    img_f = np.asarray(img, dtype=np.float64)
    gx = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=k)
    return np.hypot(gx, gy)


def sobel_binary_expand_steps(
    img: np.ndarray, params: dict,
) -> list[tuple[str, np.ndarray]]:
    """
    Emphasize microscopic flaws:
      1. Sobel gradient magnitude
      2. Binary threshold
      3. Morphological expansion (dilation)
    """
    k = int(params.get('sobel_ksize', 3))
    mag = gradient_magnitude(img, ksize=k)
    mag_u8 = as_u8(mag)

    abs_thr = float(params.get('binary_absolute', 0) or 0)
    if abs_thr > 0:
        thr = abs_thr
        thr_label = f'Binary (abs ≥ {thr:g})'
    else:
        p = float(params.get('binary_percentile', 92.0))
        thr = float(np.percentile(mag, p))
        thr_label = f'Binary (p ≥ {p:g}%, thr={thr:.1f})'

    binary = ((mag >= thr).astype(np.uint8)) * 255

    ek = max(1, int(params.get('expand_ksize', 3)))
    if ek % 2 == 0:
        ek += 1
    iters = max(1, int(params.get('expand_iterations', 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
    expanded = cv2.dilate(binary, kernel, iterations=iters)

    return [
        (f'Sobel |∇I| k={k}', mag_u8),
        (thr_label, binary),
        (f'Expansion dilate k={ek}×{iters}', expanded),
    ]


def apply_sobel_binary_expand(img: np.ndarray, params: dict) -> np.ndarray:
    return sobel_binary_expand_steps(img, params)[-1][1]


def apply_percentile_sobel_binary_expand(img: np.ndarray, params: dict) -> np.ndarray:
    return apply_sobel_binary_expand(stretch_frame(img, params), params)


def plot_gradient_panels(
    ax_row: np.ndarray,
    img: np.ndarray,
    label: str,
    *,
    display_vmax: float | None = None,
    grad_vmax: float | None = None,
):
    mag = gradient_magnitude(img)
    show_image(ax_row[0], img, label, vmax=display_vmax)
    vmax_g = grad_vmax if grad_vmax is not None else np.percentile(mag, 99)
    show_image(ax_row[1], mag, f'{label} |∇I|', vmax=vmax_g)
    return mag


def percentile_stretch(img: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo, hi = np.percentile(img, (p_low, p_high))
    if hi <= lo:
        hi = lo + 1.0
    out = (img.astype(np.float64) - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_median(img: np.ndarray, ksize: int) -> np.ndarray:
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)


def apply_tophat(img: np.ndarray, ksize: int) -> np.ndarray:
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(as_u8(img), cv2.MORPH_TOPHAT, kernel)


def apply_rolling_ball(img: np.ndarray, radius: float) -> np.ndarray:
    """Subtract rolling-ball background (skimage). Bright particles on dark bg."""
    from skimage.restoration import rolling_ball

    img_f = np.asarray(img, dtype=np.float64)
    background = rolling_ball(img_f, radius=max(1.0, float(radius)))
    return np.clip(img_f - background, 0, 255).astype(np.uint8)


def apply_morph_open_bg_subtract(img: np.ndarray, ksize: int) -> np.ndarray:
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(as_u8(img), cv2.MORPH_OPEN, kernel)
    return subtract_background(img, opened)


def subtract_background(img: np.ndarray, background: np.ndarray) -> np.ndarray:
    diff = as_u8(img).astype(np.float64) - as_u8(background).astype(np.float64)
    return np.clip(diff, 0, 255).astype(np.uint8)


def clip_low_percentile(img: np.ndarray, percentile: float) -> np.ndarray:
    """Zero out the darkest residual background after subtraction."""
    if percentile <= 0:
        return as_u8(img)
    cutoff = np.percentile(img, percentile)
    out = img.astype(np.float64)
    out[out < cutoff] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def stretch_frame(img: np.ndarray, params: dict) -> np.ndarray:
    return percentile_stretch(
        img, params['percentile_low'], params['percentile_high'])


def remove_background_from_stretch(stretched: np.ndarray, params: dict, method: str) -> np.ndarray:
    """Background removal on an already percentile-stretched image."""
    stretched = as_u8(stretched)
    if method == 'tophat':
        out = apply_tophat(stretched, params['tophat_ksize'])
    elif method == 'gaussian':
        k = max(3, int(params['bg_gaussian_ksize']))
        if k % 2 == 0:
            k += 1
        bg = cv2.GaussianBlur(stretched, (k, k), 0)
        out = subtract_background(stretched, bg)
    elif method == 'median':
        k = max(3, int(params['bg_median_ksize']))
        if k % 2 == 0:
            k += 1
        bg = cv2.medianBlur(stretched, k)
        out = subtract_background(stretched, bg)
    elif method == 'bandpass':
        out = as_u8(apply_trackpy_bandpass(
            stretched, params['bandpass_lshort'], params['bandpass_llong']))
    elif method == 'rolling_ball':
        out = apply_rolling_ball(
            stretched, params.get('rolling_ball_radius', 25))
    elif method == 'morph_open':
        out = apply_morph_open_bg_subtract(
            stretched, params.get('morph_open_ksize', 31))
    else:
        raise ValueError(
            f"Unknown bg method {method!r}. "
            "Use: rolling_ball, morph_open, tophat, gaussian, median, bandpass")
    return clip_low_percentile(out, params.get('bg_clip_percentile', 0))


def apply_percentile_bg_pipeline(img: np.ndarray, params: dict, bg_method: str) -> np.ndarray:
    stretched = stretch_frame(img, params)
    return remove_background_from_stretch(stretched, params, bg_method)


def apply_gaussian_percentile(img: np.ndarray, params: dict) -> np.ndarray:
    """Gaussian blur first, then percentile stretch."""
    blurred = apply_gaussian_blur(img, params['gaussian_blur_ksize'])
    return stretch_frame(blurred, params)


def apply_percentile_gaussian_blur(img: np.ndarray, params: dict) -> np.ndarray:
    """Percentile stretch first, then Gaussian blur (reverse order for comparison)."""
    stretched = stretch_frame(img, params)
    return apply_gaussian_blur(stretched, params['gaussian_blur_ksize'])


def apply_gaussian_percentile_bg_pipeline(
    img: np.ndarray, params: dict, bg_method: str,
) -> np.ndarray:
    """Gaussian blur → percentile stretch → background removal."""
    stretched = apply_gaussian_percentile(img, params)
    return remove_background_from_stretch(stretched, params, bg_method)


def apply_clahe(img: np.ndarray, clip: float, tile: int) -> np.ndarray:
    tile = max(2, int(tile))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    return clahe.apply(as_u8(img))


def apply_hist_eq(img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(as_u8(img))


def apply_trackpy_bandpass(img: np.ndarray, lshort: int, llong: int) -> np.ndarray:
    return tp.bandpass(img, lshort=lshort, llong=llong)


def apply_fft_bandpass(img: np.ndarray, diameter: int) -> np.ndarray:
    if fft_filter is None:
        raise ImportError(
            'ptv_calib is not installed; FFT bandpass unavailable.')
    out = fft_filter(
        img,
        diameterDot=diameter,
        contrast='equalizeHist',
        plotting='None',
        denoise_method='none',
    )
    return as_u8(out)


def build_background_removal_panels(
    frame: np.ndarray,
    params: dict,
) -> list[tuple[str, str, np.ndarray]]:
    """Compare background-removal methods only (applied to the raw frame)."""
    panels: list[tuple[str, str, np.ndarray]] = [
        ('raw', 'raw', frame),
    ]
    methods = [
        (
            'rolling_ball',
            f'rolling-ball r={params.get("rolling_ball_radius", 25)}',
            lambda: apply_rolling_ball(frame, params.get('rolling_ball_radius', 25)),
        ),
        (
            'morph_open',
            f'morph opening k={params.get("morph_open_ksize", 31)}',
            lambda: apply_morph_open_bg_subtract(frame, params.get('morph_open_ksize', 31)),
        ),
        (
            'gaussian',
            f'Gaussian bg k={params["bg_gaussian_ksize"]}',
            lambda: remove_background_from_stretch(frame, params, 'gaussian'),
        ),
        (
            'median',
            f'median bg k={params["bg_median_ksize"]}',
            lambda: remove_background_from_stretch(frame, params, 'median'),
        ),
        (
            'tophat',
            f'top-hat k={params["tophat_ksize"]}',
            lambda: apply_tophat(frame, params['tophat_ksize']),
        ),
        (
            'bandpass',
            f'bandpass ({params["bandpass_lshort"]}, {params["bandpass_llong"]})',
            lambda: as_u8(apply_trackpy_bandpass(
                frame, params['bandpass_lshort'], params['bandpass_llong'])),
        ),
    ]
    for key, title, fn in methods:
        panels.append((key, title, fn()))
    return panels


def build_stretch_background_panels(
    frame: np.ndarray,
    params: dict,
) -> list[tuple[str, str, np.ndarray]]:
    """Compare background-removal strategies after percentile stretch."""
    stretched = stretch_frame(frame, params)
    p_lo, p_hi = params['percentile_low'], params['percentile_high']
    panels = [
        ('percentile', f'percentile stretch ({p_lo}–{p_hi}%)', stretched),
    ]
    bg_methods = [
        ('percentile_rolling_ball', 'rolling_ball', params.get(
            'rolling_ball_radius', 25), 'rolling-ball r'),
        ('percentile_morph_open', 'morph_open', params.get(
            'morph_open_ksize', 31), 'morph-open k'),
        ('percentile_tophat', 'tophat', params['tophat_ksize'], 'top-hat k'),
        ('percentile_gaussian', 'gaussian',
         params['bg_gaussian_ksize'], 'Gaussian bg k'),
        ('percentile_median', 'median',
         params['bg_median_ksize'], 'median bg k'),
        (
            'percentile_bandpass', 'bandpass',
            (params['bandpass_lshort'], params['bandpass_llong']),
            'bandpass',
        ),
    ]
    for key, method, setting, label in bg_methods:
        if method == 'bandpass':
            title = f'{label} ({setting[0]}, {setting[1]})'
        else:
            title = f'{label}={setting}'
        panels.append((
            key,
            title,
            remove_background_from_stretch(stretched, params, method),
        ))
    clip_p = params.get('bg_clip_percentile', 0)
    if clip_p > 0:
        panels.append((
            f'percentile_{BG_REMOVAL}_clip',
            f'{BG_REMOVAL} + clip p={clip_p}',
            clip_low_percentile(
                remove_background_from_stretch(stretched, params, BG_REMOVAL),
                clip_p,
            ),
        ))
    return panels


def build_gaussian_percentile_panels(
    frame: np.ndarray,
    params: dict,
) -> list[tuple[str, str, np.ndarray]]:
    """Compare Gaussian blur and percentile stretch orderings."""
    p_lo, p_hi = params['percentile_low'], params['percentile_high']
    k = params['gaussian_blur_ksize']
    blurred = apply_gaussian_blur(frame, k)
    gauss_pct = apply_gaussian_percentile(frame, params)
    pct_gauss = apply_percentile_gaussian_blur(frame, params)
    panels = [
        ('raw', 'raw frame', frame),
        ('gaussian', f'1. Gaussian blur k={k}', blurred),
        (
            'gaussian_percentile',
            f'2. Gaussian k={k} → percentile ({p_lo}–{p_hi}%)',
            gauss_pct,
        ),
        (
            'percentile',
            f'percentile only ({p_lo}–{p_hi}%)',
            stretch_frame(frame, params),
        ),
        (
            'percentile_gaussian_blur',
            f'percentile → Gaussian k={k} (reverse order)',
            pct_gauss,
        ),
    ]
    for bg_method in ('rolling_ball', 'bandpass', 'tophat', 'gaussian', 'median'):
        if bg_method == 'bandpass':
            setting = (params['bandpass_lshort'], params['bandpass_llong'])
            title = f'Gaussian → percentile → bandpass {setting}'
        elif bg_method == 'tophat':
            title = f'Gaussian → percentile → top-hat k={params["tophat_ksize"]}'
        elif bg_method == 'gaussian':
            title = f'Gaussian → percentile → bg subtract k={params["bg_gaussian_ksize"]}'
        elif bg_method == 'rolling_ball':
            title = f'Gaussian → percentile → rolling-ball r={params.get("rolling_ball_radius", 25)}'
        else:
            title = f'Gaussian → percentile → median bg k={params["bg_median_ksize"]}'
        panels.append((
            f'gaussian_percentile_{bg_method}',
            title,
            apply_gaussian_percentile_bg_pipeline(frame, params, bg_method),
        ))
    return panels


def build_preprocessing_panels(
    frame: np.ndarray,
    params: dict,
) -> list[tuple[str, str, np.ndarray]]:
    """Return (key, title, image) tuples for each preprocessing method."""
    panels: list[tuple[str, str, np.ndarray]] = [
        ('raw', 'raw frame', frame),
    ]

    panels.extend([
        (
            'particle_snr',
            'Particle SNR (bg→denoise→DoG/LoG→contrast)',
            apply_particle_snr_pipeline(frame, params),
        ),
        (
            'bandpass_threshold',
            'Bandpass → hard threshold',
            apply_bandpass_threshold(frame, params),
        ),
        (
            'bandpass_then_bg',
            (
                f'Bandpass → {params.get("bandpass_bg_method", "rolling_ball")}'
            ),
            apply_bandpass_then_bg(frame, params),
        ),
        (
            'bandpass_rolling_ball',
            'Bandpass → rolling-ball',
            apply_bandpass_then_bg(frame, params, bg_method='rolling_ball'),
        ),
        (
            'bandpass_morph_open',
            'Bandpass → morph opening',
            apply_bandpass_then_bg(frame, params, bg_method='morph_open'),
        ),
        (
            'sobel_binary_expand',
            'Sobel → binary → expansion',
            apply_sobel_binary_expand(frame, params),
        ),
        (
            'percentile_sobel_binary_expand',
            'Percentile → Sobel → binary → expansion',
            apply_percentile_sobel_binary_expand(frame, params),
        ),
        (
            'rolling_ball',
            f'rolling-ball r={params.get("rolling_ball_radius", 25)}',
            apply_rolling_ball(frame, params.get('rolling_ball_radius', 25)),
        ),
        (
            'percentile_rolling_ball',
            (
                f'percentile → rolling-ball r={params.get("rolling_ball_radius", 25)}'
            ),
            apply_percentile_bg_pipeline(frame, params, 'rolling_ball'),
        ),
        (
            'percentile',
            f'percentile stretch ({params["percentile_low"]}–{params["percentile_high"]}%)',
            percentile_stretch(
                frame, params['percentile_low'], params['percentile_high']),
        ),
        (
            'gaussian',
            f'Gaussian blur k={params["gaussian_blur_ksize"]}',
            apply_gaussian_blur(frame, params['gaussian_blur_ksize']),
        ),
        (
            'gaussian_percentile',
            (
                f'Gaussian k={params["gaussian_blur_ksize"]} → '
                f'percentile ({params["percentile_low"]}–{params["percentile_high"]}%)'
            ),
            apply_gaussian_percentile(frame, params),
        ),
        (
            'percentile_gaussian_blur',
            (
                f'percentile → Gaussian k={params["gaussian_blur_ksize"]}'
            ),
            apply_percentile_gaussian_blur(frame, params),
        ),
        (
            'gaussian_percentile_bandpass',
            (
                f'Gaussian → percentile → bandpass '
                f'({params["bandpass_lshort"]}, {params["bandpass_llong"]})'
            ),
            apply_gaussian_percentile_bg_pipeline(frame, params, 'bandpass'),
        ),
        (
            'median',
            f'median k={params["median_ksize"]}',
            apply_median(frame, params['median_ksize']),
        ),
        (
            'tophat',
            f'top-hat k={params["tophat_ksize"]}',
            apply_tophat(frame, params['tophat_ksize']),
        ),
        (
            'percentile_tophat',
            (
                f'percentile ({params["percentile_low"]}–{params["percentile_high"]}%) '
                f'→ top-hat k={params["tophat_ksize"]}'
            ),
            apply_percentile_bg_pipeline(frame, params, 'tophat'),
        ),
        (
            'percentile_gaussian',
            (
                f'percentile → Gaussian bg k={params["bg_gaussian_ksize"]}'
            ),
            apply_percentile_bg_pipeline(frame, params, 'gaussian'),
        ),
        (
            'percentile_median',
            (
                f'percentile → median bg k={params["bg_median_ksize"]}'
            ),
            apply_percentile_bg_pipeline(frame, params, 'median'),
        ),
        (
            'percentile_bandpass',
            (
                f'percentile → bandpass ({params["bandpass_lshort"]}, {params["bandpass_llong"]})'
            ),
            apply_percentile_bg_pipeline(frame, params, 'bandpass'),
        ),
        (
            'clahe',
            f'CLAHE clip={params["clahe_clip"]}',
            apply_clahe(frame, params['clahe_clip'], params['clahe_tile']),
        ),
        ('hist_eq', 'histogram equalize', apply_hist_eq(frame)),
        (
            'trackpy_bandpass',
            f'trackpy bandpass ({params["bandpass_lshort"]}, {params["bandpass_llong"]})',
            apply_trackpy_bandpass(
                frame, params['bandpass_lshort'], params['bandpass_llong']),
        ),
    ])

    if fft_filter is not None:
        panels.append((
            'fft_bandpass',
            f'FFT bandpass (d={params["fft_diameter"]})',
            apply_fft_bandpass(frame, params['fft_diameter']),
        ))

    return panels


def plot_preprocessing_grid(
    panels: list[tuple[str, str, np.ndarray]],
    *,
    display_vmax: float | None = None,
    suptitle: str = '',
    ncols: int = 4,
):
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (key, title, img) in zip(axes_flat, panels):
        vmax = display_vmax if key == 'raw' else None
        show_image(ax, img, title, vmax=vmax)

    for ax in axes_flat[n:]:
        ax.axis('off')

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def locate_particles(
    img: np.ndarray,
    diameter: int,
    threshold: int,
    minmass: int,
    *,
    preprocess: bool = False,
):
    return tp.locate(
        img,
        diameter=diameter,
        threshold=threshold,
        minmass=minmass,
        separation=max(2, diameter // 2),
        preprocess=preprocess,
    )


def overlay_trackpy_centers(
    ax,
    img: np.ndarray,
    features,
    diameter: int,
    *,
    title: str | None = None,
    vmax: float | None = None,
):
    label = title or f'trackpy locate (n={len(features)})'
    show_image(ax, img, label, vmax=vmax)
    if features is not None and len(features):
        ax.plot(
            features['x'],
            features['y'],
            'o',
            fillstyle='none',
            markeredgecolor='lime',
            markersize=diameter * 0.6,
            markeredgewidth=0.8,
        )
    return features


# %% load cine files

run_dir = os.path.abspath(RUN_DIR)
cine_paths = discover_cine_files(run_dir)
if CAMERA is not None:
    selected = os.path.join(run_dir, f'{CAMERA}.cine')
    if not os.path.isfile(selected):
        raise FileNotFoundError(f'{selected} not found')
    cine_paths = [selected]

print(f'Run folder: {run_dir}')
print(f'Loading frame {FRAME_INDEX} from {len(cine_paths)} file(s):')
for path in cine_paths:
    print(f'  {os.path.basename(path)}')

frames = {}
for path in cine_paths:
    name = os.path.splitext(os.path.basename(path))[0]
    frames[name] = load_cine_frame(path, FRAME_INDEX)

camera_name = os.path.splitext(os.path.basename(cine_paths[0]))[0]
frame = frames[camera_name]
roi = ROI if ROI is not None else center_roi(frame, CENTER_WINDOW)
view = crop_roi(frame, roi)
print(
    f'\n{camera_name} frame {FRAME_INDEX}: shape={frame.shape}, '
    f'dtype={frame.dtype}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}')
print(f'Center window ROI [x0, y0, w, h]: {roi}')

# %% full frame with center window marked

fig, ax = plt.subplots(figsize=(10, 6))
show_image(ax, frame, f'{camera_name} frame {FRAME_INDEX}', vmax=DISPLAY_VMAX)
x0, y0, w, h = roi
rect = plt.Rectangle((x0, y0), w, h, fill=False,
                     edgecolor='lime', linewidth=1.5)
ax.add_patch(rect)
ax.set_title(
    f'128×128 inspection window (frame {FRAME_INDEX} only; frame 0 discarded)')
fig.tight_layout()
plt.show()

# %% bandpass then threshold / background subtraction (center window)

# Compare on the center crop — rolling-ball on full frames is very slow
frame_roi = crop_roi(frame, roi)

thr_panels = [
    (key, title, img)
    for key, title, img in build_bandpass_threshold_panels(frame_roi, PREPROCESS_PARAMS)
]
fig = plot_preprocessing_grid(
    thr_panels,
    display_vmax=DISPLAY_VMAX,
    suptitle=(
        f'{camera_name} frame {FRAME_INDEX} — bandpass then hard threshold\n'
        'Zeros residual mottled floor; raise percentile if background remains'
    ),
    ncols=4,
)
plt.show()

bp_bg_panels = [
    (key, title, img)
    for key, title, img in build_bandpass_then_bg_panels(frame_roi, PREPROCESS_PARAMS)
]
fig = plot_preprocessing_grid(
    bp_bg_panels,
    display_vmax=DISPLAY_VMAX,
    suptitle=(
        f'{camera_name} frame {FRAME_INDEX} — bandpass then cleanup options\n'
        f'bandpass ({PREPROCESS_PARAMS["bandpass_lshort"]}, '
        f'{PREPROCESS_PARAMS["bandpass_llong"]}) → threshold / rolling-ball / '
        'morph-open / Gaussian / median / top-hat'
    ),
    ncols=4,
)
plt.show()

# Active METHOD step-through for bandpass pipelines
if METHOD == 'bandpass_threshold':
    steps = bandpass_threshold_steps(frame_roi, PREPROCESS_PARAMS)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), squeeze=False)
    show_image(axes[0][0], view, '0. raw', vmax=DISPLAY_VMAX)
    for ax, (label, img) in zip(axes[0][1:], steps):
        show_image(ax, img, label)
    fig.suptitle(f'{camera_name} frame {FRAME_INDEX} — {METHOD}')
    fig.tight_layout()
    plt.show()
    processed_view = steps[-1][1]
    method_title = 'Bandpass → threshold'
elif METHOD.startswith('bandpass_'):
    bg_for_steps = (
        PREPROCESS_PARAMS.get('bandpass_bg_method', 'morph_open')
        if METHOD == 'bandpass_then_bg'
        else METHOD.removeprefix('bandpass_')
    )
    steps = bandpass_then_bg_steps(frame_roi, PREPROCESS_PARAMS, bg_method=bg_for_steps)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), squeeze=False)
    show_image(axes[0][0], view, '0. raw', vmax=DISPLAY_VMAX)
    for ax, (label, img) in zip(axes[0][1:], steps):
        show_image(ax, img, label)
    fig.suptitle(f'{camera_name} frame {FRAME_INDEX} — {METHOD}')
    fig.tight_layout()
    plt.show()
    processed_view = steps[-1][1]
    method_title = f'Bandpass → {bg_for_steps}'
    # Full-frame result for detection is set in the tune cell below

# %% Particle SNR pipeline (recommended)
# bg → denoise → (±deconv) → multi-scale DoG/LoG → contrast → local-maxima

CUSTOM_PARAMS = dict(PREPROCESS_PARAMS)
snr_steps = particle_snr_pipeline_steps(frame, CUSTOM_PARAMS)
n_show = len(snr_steps) + 1  # include raw
fig, axes = plt.subplots(1, n_show, figsize=(3.2 * n_show, 3.5), squeeze=False)
show_image(axes[0][0], view, '0. raw', vmax=DISPLAY_VMAX)
for ax, (label, img) in zip(axes[0][1:], snr_steps):
    show_image(ax, crop_roi(img, roi), label)
fig.suptitle(
    f'{camera_name} frame {FRAME_INDEX} — particle SNR pipeline '
    f'(bg={CUSTOM_PARAMS["bg_method"]}, denoise={CUSTOM_PARAMS["denoise_method"]}, '
    f'blob={CUSTOM_PARAMS["blob_method"]}, contrast={CUSTOM_PARAMS["contrast_method"]})',
    fontsize=10,
)
fig.tight_layout()
plt.show()

if METHOD == 'particle_snr':
    processed_full = snr_steps[-1][1]
    processed_view = crop_roi(processed_full, roi)
    method_title = 'Particle SNR pipeline'

# %% Emphasize microscopic flaws: Sobel → binary → expansion (optional comparison)

sbe_input = frame
sbe_prefix = 'raw'
sbe_steps = [('input', sbe_input), *
             sobel_binary_expand_steps(sbe_input, CUSTOM_PARAMS)]
fig, axes = plt.subplots(1, len(sbe_steps), figsize=(
    3.4 * len(sbe_steps), 3.6), squeeze=False)
for ax, (label, img) in zip(axes[0], sbe_steps):
    crop = crop_roi(img, roi)
    vmax = DISPLAY_VMAX if label == 'input' else None
    show_image(ax, crop, label, vmax=vmax)
fig.suptitle(
    f'{camera_name} frame {FRAME_INDEX} — Sobel → binary → expansion (comparison)',
    fontsize=11,
)
fig.tight_layout()
plt.show()

# %% intensity gradients (center window)

stretched_view = crop_roi(stretch_frame(frame, PREPROCESS_PARAMS), roi)
processed_grad_view = crop_roi(
    apply_percentile_bg_pipeline(frame, PREPROCESS_PARAMS, BG_REMOVAL), roi)

gradient_rows = [
    (view, 'raw'),
    (stretched_view, 'percentile stretch'),
    (processed_grad_view, f'percentile → {BG_REMOVAL}'),
]
all_mags = [gradient_magnitude(img) for img, _ in gradient_rows]
shared_grad_vmax = float(np.percentile(
    np.concatenate([m.ravel() for m in all_mags]), 99))

fig, axes = plt.subplots(3, 2, figsize=(8, 10), squeeze=False)
for ax_row, (img, label) in zip(axes, gradient_rows):
    plot_gradient_panels(
        ax_row,
        img,
        label,
        display_vmax=DISPLAY_VMAX if label == 'raw' else None,
        grad_vmax=shared_grad_vmax,
    )

fig.suptitle(
    f'{camera_name} frame {FRAME_INDEX} — intensity gradients in center {CENTER_WINDOW}×{CENTER_WINDOW} window',
    fontsize=11,
)
fig.tight_layout()
plt.show()

# 1D profiles: mean |∇I| along rows and columns (background illumination trends)
raw_mag = gradient_magnitude(view)
proc_mag = gradient_magnitude(processed_grad_view)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(raw_mag.mean(axis=1), label='raw', color='0.45')
axes[0].plot(proc_mag.mean(axis=1),
             label=f'percentile → {BG_REMOVAL}', color='C1')
axes[0].set_xlabel('row index')
axes[0].set_ylabel('mean |∇I|')
axes[0].set_title('Vertical gradient profile (mean over columns)')
axes[0].legend(fontsize=8)

axes[1].plot(raw_mag.mean(axis=0), label='raw', color='0.45')
axes[1].plot(proc_mag.mean(axis=0),
             label=f'percentile → {BG_REMOVAL}', color='C1')
axes[1].set_xlabel('column index')
axes[1].set_ylabel('mean |∇I|')
axes[1].set_title('Horizontal gradient profile (mean over rows)')
axes[1].legend(fontsize=8)
fig.suptitle(
    f'{camera_name} frame {FRAME_INDEX} — gradient magnitude profiles')
fig.tight_layout()
plt.show()

# %% Gaussian blur then percentile stretch (center window)

gauss_pct_panels = [
    (key, title, crop_roi(img, roi))
    for key, title, img in build_gaussian_percentile_panels(frame, PREPROCESS_PARAMS)
]
fig = plot_preprocessing_grid(
    gauss_pct_panels,
    display_vmax=DISPLAY_VMAX,
    suptitle=(
        f'{camera_name} frame {FRAME_INDEX} — Gaussian blur vs percentile orderings\n'
        'Blur first suppresses noise before contrast stretch'
    ),
    ncols=3,
)
plt.show()

# %% background removal comparison only (center window)

bg_panels = [
    (key, title, crop_roi(img, roi))
    for key, title, img in build_background_removal_panels(frame, PREPROCESS_PARAMS)
]
fig = plot_preprocessing_grid(
    bg_panels,
    display_vmax=DISPLAY_VMAX,
    suptitle=(
        f'{camera_name} frame {FRAME_INDEX} — background removal only\n'
        'raw vs rolling-ball / morph-open / Gaussian / median / top-hat / bandpass'
    ),
    ncols=4,
)
plt.show()

# Same methods after percentile stretch (illumination already boosted)
stretch_bg_panels = [
    (key, title, crop_roi(img, roi))
    for key, title, img in build_stretch_background_panels(frame, PREPROCESS_PARAMS)
]
fig = plot_preprocessing_grid(
    stretch_bg_panels,
    display_vmax=DISPLAY_VMAX,
    suptitle=(
        f'{camera_name} frame {FRAME_INDEX} — percentile stretch, then background removal'
    ),
    ncols=4,
)
plt.show()

# %% tune one method — edit METHOD in CONFIG and re-run this cell

# METHOD default: bandpass_threshold
# also: bandpass_morph_open | bandpass_rolling_ball | particle_snr | ...
CUSTOM_PARAMS = dict(PREPROCESS_PARAMS)

panels_by_key = {
    key: (title, img)
    for key, title, img in build_preprocessing_panels(frame, CUSTOM_PARAMS)
}
if METHOD not in panels_by_key:
    raise ValueError(
        f'Unknown METHOD={METHOD!r}. Choose from: {list(panels_by_key)}')

method_title, processed_full = panels_by_key[METHOD]
processed_view = crop_roi(processed_full, roi)

if METHOD == 'bandpass_threshold':
    steps = bandpass_threshold_steps(frame, CUSTOM_PARAMS)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), squeeze=False)
    show_image(axes[0][0], view, '0. raw', vmax=DISPLAY_VMAX)
    for ax, (label, img) in zip(axes[0][1:], steps):
        show_image(ax, crop_roi(img, roi), label)
    fig.suptitle(method_title)
elif METHOD.startswith('bandpass_'):
    bg_for_steps = (
        CUSTOM_PARAMS.get('bandpass_bg_method', 'morph_open')
        if METHOD == 'bandpass_then_bg'
        else METHOD.removeprefix('bandpass_')
    )
    steps = bandpass_then_bg_steps(frame, CUSTOM_PARAMS, bg_method=bg_for_steps)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), squeeze=False)
    show_image(axes[0][0], view, '0. raw', vmax=DISPLAY_VMAX)
    for ax, (label, img) in zip(axes[0][1:], steps):
        show_image(ax, crop_roi(img, roi), label)
    fig.suptitle(method_title)
elif METHOD == 'particle_snr':
    steps = particle_snr_pipeline_steps(frame, CUSTOM_PARAMS)
    fig, axes = plt.subplots(
        1, len(steps) + 1, figsize=(3.2 * (len(steps) + 1), 3.5), squeeze=False)
    show_image(axes[0][0], view, 'raw', vmax=DISPLAY_VMAX)
    for ax, (label, img) in zip(axes[0][1:], steps):
        show_image(ax, crop_roi(img, roi), label)
    fig.suptitle(method_title)
elif METHOD in ('sobel_binary_expand', 'percentile_sobel_binary_expand'):
    base = stretch_frame(frame, CUSTOM_PARAMS) if METHOD.startswith(
        'percentile_') else frame
    steps = [('input', base), *sobel_binary_expand_steps(base, CUSTOM_PARAMS)]
    fig, axes = plt.subplots(1, len(steps), figsize=(
        3.4 * len(steps), 3.6), squeeze=False)
    for ax, (label, img) in zip(axes[0], steps):
        show_image(
            ax, crop_roi(img, roi), label,
            vmax=DISPLAY_VMAX if label == 'input' and METHOD == 'sobel_binary_expand' else None,
        )
    fig.suptitle(method_title)
elif METHOD.startswith('percentile_') and METHOD != 'percentile':
    stretched_full = stretch_frame(frame, CUSTOM_PARAMS)
    stretched_view = crop_roi(stretched_full, roi)
    bg_method = METHOD.removeprefix('percentile_')
    bg_view = crop_roi(
        remove_background_from_stretch(stretched_full, CUSTOM_PARAMS, bg_method), roi)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    show_image(axes[0], view, 'raw (center window)', vmax=DISPLAY_VMAX)
    show_image(axes[1], stretched_view, '1. percentile stretch')
    show_image(axes[2], bg_view, f'2. remove background ({bg_method})')
    fig.suptitle(method_title)
elif METHOD.startswith('gaussian_percentile'):
    blurred_full = apply_gaussian_blur(
        frame, CUSTOM_PARAMS['gaussian_blur_ksize'])
    blurred_view = crop_roi(blurred_full, roi)
    if METHOD == 'gaussian_percentile':
        result_view = crop_roi(
            apply_gaussian_percentile(frame, CUSTOM_PARAMS), roi)
        step2_title = (
            f'2. percentile ({CUSTOM_PARAMS["percentile_low"]}–'
            f'{CUSTOM_PARAMS["percentile_high"]}%)'
        )
    else:
        bg_method = METHOD.removeprefix('gaussian_percentile_')
        result_view = crop_roi(
            apply_gaussian_percentile_bg_pipeline(frame, CUSTOM_PARAMS, bg_method), roi)
        step2_title = (
            f'2. percentile → remove bg ({bg_method})'
        )
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    show_image(axes[0], view, 'raw (center window)', vmax=DISPLAY_VMAX)
    show_image(
        axes[1], blurred_view,
        f'1. Gaussian blur k={CUSTOM_PARAMS["gaussian_blur_ksize"]}',
    )
    show_image(axes[2], result_view, step2_title)
    fig.suptitle(method_title)
elif METHOD == 'percentile_gaussian_blur':
    stretched_view = crop_roi(stretch_frame(frame, CUSTOM_PARAMS), roi)
    result_view = crop_roi(
        apply_percentile_gaussian_blur(frame, CUSTOM_PARAMS), roi)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    show_image(axes[0], view, 'raw (center window)', vmax=DISPLAY_VMAX)
    show_image(axes[1], stretched_view, '1. percentile stretch')
    show_image(
        axes[2], result_view,
        f'2. Gaussian blur k={CUSTOM_PARAMS["gaussian_blur_ksize"]}',
    )
    fig.suptitle(method_title)
elif METHOD == 'percentile':
    stretched_view = crop_roi(stretch_frame(frame, CUSTOM_PARAMS), roi)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    show_image(axes[0], view, 'raw (center window)', vmax=DISPLAY_VMAX)
    show_image(axes[1], stretched_view, method_title)
elif METHOD == 'rolling_ball':
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    show_image(axes[0], view, 'raw (center window)', vmax=DISPLAY_VMAX)
    show_image(axes[1], processed_view, method_title)
else:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    show_image(axes[0], view, 'raw (center window)', vmax=DISPLAY_VMAX)
    show_image(axes[1], processed_view, method_title)
fig.tight_layout()
plt.show()

# %% particle detection in center window

if PARTICLE_DIAMETER is not None:
    if METHOD == 'bandpass_threshold' and USE_LOCAL_MAXIMA:
        features_full, blob_steps, _sigma_map = detect_blobs_bandpass_threshold(
            frame, CUSTOM_PARAMS)
        processed_full = blob_steps[-1][1]
        processed_view = crop_roi(processed_full, roi)
        detect_label = (
            f'multi-scale {CUSTOM_PARAMS.get("blob_method", "dog").upper()} max '
            f'σ={CUSTOM_PARAMS.get("blob_sigmas")}'
        )
        method_title = 'DoG/LoG max response'
    elif METHOD == 'particle_snr' and USE_LOCAL_MAXIMA:
        detect_on = processed_full
        features_full = detect_local_maxima(detect_on, CUSTOM_PARAMS)
        detect_label = 'local maxima on DoG/LoG'
    else:
        detect_on = processed_full if METHOD != 'raw' else frame
        features_full = locate_particles(
            detect_on,
            diameter=PARTICLE_DIAMETER,
            threshold=TRACKPY_THRESHOLD,
            minmass=TRACKPY_MINMASS,
            preprocess=(METHOD == 'raw'),
        )
        detect_label = 'trackpy locate'
    features_roi = shift_features_to_crop(
        particles_in_roi(features_full, roi), roi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Raw with size-aware circles when sigma column present
    show_image(axes[0], view, f'raw + detections (n={len(features_roi)})',
               vmax=DISPLAY_VMAX)
    show_image(axes[1], processed_view,
               f'{method_title} + detections (n={len(features_roi)})')
    if features_roi is not None and len(features_roi):
        from matplotlib.patches import Circle
        for ax in axes:
            for _, row in features_roi.iterrows():
                sigma = float(row['sigma']) if 'sigma' in features_roi.columns else 2.0
                rad = max(2.5, 1.6 * sigma)
                ax.add_patch(Circle(
                    (row['x'], row['y']), rad,
                    fill=False, edgecolor='lime', linewidth=1.1,
                ))
                ax.plot(row['x'], row['y'], '+', color='red', ms=6, mew=0.9)
    fig.suptitle(
        f'{camera_name} frame {FRAME_INDEX} — {CENTER_WINDOW}×{CENTER_WINDOW} center window\n'
        f'{detect_label}',
        fontsize=10,
    )
    fig.tight_layout()
    plt.show()

    if len(features_roi):
        print(
            f'Particles in center window: {len(features_roi)} ({detect_label})')
        cols = [c for c in ['x', 'y', 'sigma', 'size', 'mass', 'signal']
                if c in features_roi.columns]
        print(features_roi[cols].to_string(index=False))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(features_roi['mass'], bins=40,
                color='steelblue', edgecolor='white')
        ax.set_xlabel('mass / peak response')
        ax.set_ylabel('count')
        ax.set_title(
            f'Mass distribution — center window ({len(features_roi)} particles)')
        fig.tight_layout()
        plt.show()
    else:
        print('No particles detected in center window — try lowering peak_threshold_rel / threshold')
else:
    print('PARTICLE_DIAMETER is None — skipping detection preview')

# %% all cameras — center window side-by-side

if len(frames) > 1:
    names = sorted(frames)
    fig, axes = plt.subplots(1, len(names), figsize=(
        3.0 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        cam_roi = ROI if ROI is not None else center_roi(
            frames[name], CENTER_WINDOW)
        show_image(ax, crop_roi(
            frames[name], cam_roi), name, vmax=DISPLAY_VMAX)
    fig.suptitle(
        f'frame {FRAME_INDEX} — center {CENTER_WINDOW}×{CENTER_WINDOW} — all cameras')
    fig.tight_layout()
    plt.show()

# %% intensity histogram (center window: raw vs processed)

fig, ax = plt.subplots(figsize=(7, 3.5))
bins = np.arange(0, 257)
ax.hist(view.ravel(), bins=bins, alpha=0.5,
        label='raw', color='0.4', density=True)
ax.hist(processed_view.ravel(), bins=bins, alpha=0.5,
        label=method_title, color='C1', density=True)
ax.set_xlim(0, 255)
ax.set_xlabel('intensity')
ax.set_ylabel('density')
ax.legend()
ax.set_title(f'{camera_name} frame {FRAME_INDEX} intensity histogram')
fig.tight_layout()
plt.show()
