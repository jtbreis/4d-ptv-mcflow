"""Shared cine frame loading and preprocessing pipelines for PTV exploration."""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pims
import trackpy as tp

try:
    from ptv_calib.preprocessing.filter_images import fft_filter
except ImportError:
    fft_filter = None

DEFAULT_PARAMS: dict[str, Any] = {
    'gaussian_blur_ksize': 3,
    'median_ksize': 3,
    'tophat_ksize': 15,
    'bg_gaussian_ksize': 31,
    'bg_median_ksize': 31,
    'bg_clip_percentile': 0,
    'clahe_clip': 2.0,
    'clahe_tile': 8,
    'bandpass_lshort': 1,
    'bandpass_llong': 9,
    'fft_diameter': 7,
    'percentile_low': 1.0,
    'percentile_high': 99.5,
    # Emphasize microscopic flaws: Sobel → binary → expansion
    'sobel_ksize': 3,
    'binary_percentile': 92.0,   # keep top (100-p)% of |∇I| as particle pixels
    'binary_absolute': 0.0,      # if >0, use absolute threshold instead of percentile
    'expand_ksize': 3,           # morphological dilation kernel (odd)
    'expand_iterations': 1,
    'rolling_ball_radius': 25,   # skimage rolling-ball background radius (px)
    # Particle-SNR pipeline (sparse bright blobs on dark background)
    'bg_method': 'rolling_ball',       # rolling_ball | morph_open | median | gaussian
    'morph_open_ksize': 31,            # structuring element > largest particle
    'denoise_method': 'nlm',           # nlm | bilateral | wavelet | none
    'nlm_h': 0.05,                     # NLM filter strength (float image 0–1)
    'nlm_patch': 5,
    'nlm_patch_distance': 6,
    'bilateral_d': 5,
    'bilateral_sigma_color': 40,
    'bilateral_sigma_space': 40,
    'wavelet': 'db2',
    'blob_method': 'dog',              # dog | log
    'blob_sigmas': (1.0, 1.5, 2.0, 3.0, 4.0),  # multi-scale: in-focus → defocused
    'contrast_method': 'percentile',   # percentile | clahe | gamma | log
    'gamma': 0.7,
    'deconv_method': 'none',           # none | richardson_lucy | wiener
    'deconv_psf_sigma': 1.2,
    'deconv_iterations': 5,
    'peak_min_distance': 3,
    'peak_threshold_rel': 0.15,
    # Bandpass first, then background subtraction
    'bandpass_bg_method': 'morph_open',  # rolling_ball | morph_open | gaussian | median | tophat
    # Hard threshold after bandpass (zeros residual mottled floor; particles stay)
    # If bp_threshold_abs > 0, use absolute intensity; else percentile of bandpass image.
    'bp_threshold_percentile': 97.0,
    'bp_threshold_abs': 0.0,
    # After DoG/LoG peak finding, snap/refine to intensity centroid (fixes ~0.5 px bias)
    'refine_radius': 5,            # half-window (px) for centroid refinement
    # guide on threshold, COM on raw (default) — aligns markers with particles in the image
    'refine_on': 'raw',            # raw | threshold | bandpass
}

BG_METHODS = ('bandpass', 'tophat', 'gaussian', 'median', 'rolling_ball', 'morph_open')
BANDPASS_THEN_BG_METHODS = ('rolling_ball', 'morph_open', 'gaussian', 'median', 'tophat')

PIPELINE_CHOICES: dict[str, str] = {
    'raw': 'Raw frame',
    'particle_snr': (
        'Particle SNR: bg → denoise → multi-scale DoG/LoG → contrast '
        '(± optional deconv)'
    ),
    'bandpass_then_bg': 'Bandpass → background subtraction',
    'bandpass_threshold': 'Bandpass → hard threshold → multi-scale DoG/LoG max',
    'bandpass_rolling_ball': 'Bandpass → rolling-ball',
    'bandpass_morph_open': 'Bandpass → morph opening',
    'bandpass_gaussian': 'Bandpass → Gaussian bg subtract',
    'bandpass_median': 'Bandpass → median bg subtract',
    'bandpass_tophat': 'Bandpass → top-hat',
    'sobel_binary_expand': 'Sobel → binary → expansion',
    'percentile_sobel_binary_expand': 'Percentile → Sobel → binary → expansion',
    'rolling_ball': 'Rolling-ball background subtract',
    'percentile_rolling_ball': 'Percentile → rolling-ball',
    'percentile': 'Percentile stretch',
    'percentile_bandpass': 'Percentile → bandpass',
    'percentile_tophat': 'Percentile → top-hat',
    'percentile_gaussian': 'Percentile → Gaussian bg subtract',
    'percentile_median': 'Percentile → median bg subtract',
    'gaussian_percentile': 'Gaussian blur → percentile',
    'gaussian_percentile_bandpass': 'Gaussian → percentile → bandpass',
    'gaussian_percentile_tophat': 'Gaussian → percentile → top-hat',
    'gaussian_percentile_gaussian': 'Gaussian → percentile → Gaussian bg subtract',
    'gaussian_percentile_median': 'Gaussian → percentile → median bg subtract',
    'gaussian_percentile_rolling_ball': 'Gaussian → percentile → rolling-ball',
    'percentile_gaussian_blur': 'Percentile → Gaussian blur (reverse order)',
    'tophat': 'Top-hat only',
    'trackpy_bandpass': 'Trackpy bandpass only',
    'clahe': 'CLAHE',
    'hist_eq': 'Histogram equalize',
}


@dataclass
class PipelineResult:
    key: str
    title: str
    steps: list[tuple[str, np.ndarray]]
    final: np.ndarray


def discover_run_dirs(raw_data_root: str) -> list[str]:
    """Find Run* folders containing Camera1.cine under raw_data_root."""
    if not os.path.isdir(raw_data_root):
        return []

    runs: list[str] = []
    for dirpath, dirnames, _ in os.walk(raw_data_root, topdown=True):
        for name in dirnames:
            if not re.match(r'^run\d+$', name.lower()):
                continue
            run_path = os.path.join(dirpath, name)
            if os.path.isfile(os.path.join(run_path, 'Camera1.cine')):
                runs.append(run_path)
        dirnames[:] = [d for d in dirnames if not re.match(r'^run\d+$', d.lower())]
    return sorted(runs)


def discover_cine_files(run_dir: str) -> list[str]:
    def _camera_sort(path: str) -> tuple[int, str]:
        match = re.search(r'Camera(\d+)', os.path.basename(path), re.I)
        return (int(match.group(1)), path) if match else (0, path)

    paths = sorted(glob.glob(os.path.join(run_dir, 'Camera*.cine')), key=_camera_sort)
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


def cine_frame_count(cine_path: str) -> int:
    return len(pims.open(cine_path))


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
    h, w = img.shape[:2]
    half = size // 2
    cx, cy = w // 2, h // 2
    x0 = max(0, min(cx - half, w - size))
    y0 = max(0, min(cy - half, h - size))
    return [x0, y0, size, size]


def crop_roi(img: np.ndarray, roi: list[int]) -> np.ndarray:
    x0, y0, w, h = roi
    return img[y0:y0 + h, x0:x0 + w]


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


def subtract_background(img: np.ndarray, background: np.ndarray) -> np.ndarray:
    diff = as_u8(img).astype(np.float64) - as_u8(background).astype(np.float64)
    return np.clip(diff, 0, 255).astype(np.uint8)


def clip_low_percentile(img: np.ndarray, percentile: float) -> np.ndarray:
    if percentile <= 0:
        return as_u8(img)
    cutoff = np.percentile(img, percentile)
    out = img.astype(np.float64)
    out[out < cutoff] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def hard_threshold(
    img: np.ndarray,
    *,
    percentile: float = 0.0,
    absolute: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Zero pixels below a floor; return (thresholded image, cutoff used)."""
    img_u8 = as_u8(img)
    if absolute > 0:
        cutoff = float(absolute)
    elif percentile > 0:
        cutoff = float(np.percentile(img_u8, percentile))
    else:
        return img_u8, 0.0
    out = img_u8.astype(np.float64)
    out[out < cutoff] = 0
    return np.clip(out, 0, 255).astype(np.uint8), cutoff


def stretch_frame(img: np.ndarray, params: dict) -> np.ndarray:
    return percentile_stretch(
        img, params['percentile_low'], params['percentile_high'])


def apply_trackpy_bandpass(img: np.ndarray, lshort: int, llong: int) -> np.ndarray:
    return tp.bandpass(img, lshort=lshort, llong=llong)


def apply_clahe(img: np.ndarray, clip: float, tile: int) -> np.ndarray:
    tile = max(2, int(tile))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    return clahe.apply(as_u8(img))


def apply_hist_eq(img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(as_u8(img))


def apply_rolling_ball(img: np.ndarray, radius: float) -> np.ndarray:
    """Subtract rolling-ball background (skimage). Bright particles on dark bg."""
    from skimage.restoration import rolling_ball

    img_f = np.asarray(img, dtype=np.float64)
    background = rolling_ball(img_f, radius=max(1.0, float(radius)))
    return np.clip(img_f - background, 0, 255).astype(np.uint8)


def remove_background_from_stretch(
    stretched: np.ndarray, params: dict, method: str,
) -> np.ndarray:
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
        out = apply_rolling_ball(stretched, params.get('rolling_ball_radius', 25))
    elif method == 'morph_open':
        out = apply_morph_open_bg_subtract(stretched, params.get('morph_open_ksize', 31))
    else:
        raise ValueError(f'Unknown bg method {method!r}')
    return clip_low_percentile(out, params.get('bg_clip_percentile', 0))


def apply_morph_open_bg_subtract(img: np.ndarray, ksize: int) -> np.ndarray:
    """Background via morphological opening (structuring element > particles)."""
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(as_u8(img), cv2.MORPH_OPEN, kernel)
    return subtract_background(img, opened)


def _to_float01(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    mx = img.max()
    if mx <= 0:
        return np.zeros_like(img, dtype=np.float64)
    if mx > 1.5:
        return np.clip(img / 255.0, 0, 1)
    return np.clip(img, 0, 1)


def apply_denoise(img: np.ndarray, params: dict) -> np.ndarray:
    """Edge-preserving denoise: nlm | bilateral | wavelet | none."""
    method = str(params.get('denoise_method', 'nlm')).lower()
    if method in ('none', 'off', ''):
        return as_u8(img)

    if method == 'bilateral':
        return cv2.bilateralFilter(
            as_u8(img),
            d=int(params.get('bilateral_d', 5)),
            sigmaColor=float(params.get('bilateral_sigma_color', 40)),
            sigmaSpace=float(params.get('bilateral_sigma_space', 40)),
        )

    img01 = _to_float01(img)
    if method == 'nlm':
        from skimage.restoration import denoise_nl_means

        out = denoise_nl_means(
            img01,
            h=float(params.get('nlm_h', 0.05)),
            patch_size=int(params.get('nlm_patch', 5)),
            patch_distance=int(params.get('nlm_patch_distance', 6)),
            fast_mode=True,
            channel_axis=None,
        )
        return as_u8(out * 255.0)

    if method == 'wavelet':
        from skimage.restoration import denoise_wavelet

        out = denoise_wavelet(
            img01,
            method='BayesShrink',
            mode='soft',
            wavelet=str(params.get('wavelet', 'db2')),
            rescale_sigma=True,
            channel_axis=None,
        )
        return as_u8(np.clip(out, 0, 1) * 255.0)

    raise ValueError(f'Unknown denoise_method {method!r}')


def apply_multi_scale_blob(
    img: np.ndarray, params: dict, *, return_sigma_map: bool = False,
):
    """Multi-scale DoG or LoG — max response across scales (in-focus → defocused).

    Small sigma catches sharp particles; larger sigma catches defocused ones.
    """
    from skimage.filters import difference_of_gaussians, gaussian
    from scipy.ndimage import laplace

    img01 = _to_float01(img)
    sigmas = tuple(params.get('blob_sigmas', (1.0, 1.5, 2.0, 3.0, 4.0)))
    method = str(params.get('blob_method', 'dog')).lower()
    responses = []
    for sigma in sigmas:
        s = float(sigma)
        if method == 'log':
            blurred = gaussian(img01, sigma=s, channel_axis=None)
            # scale-normalized LoG response (bright blobs → positive after abs)
            resp = -laplace(blurred) * (s ** 2)
        else:
            resp = difference_of_gaussians(img01, low_sigma=s, high_sigma=s * 1.6)
        responses.append(np.clip(resp, 0, None))
    stacked = np.stack(responses, axis=0)
    enhanced = np.max(stacked, axis=0)
    out = as_u8(enhanced)
    if not return_sigma_map:
        return out
    argmax = np.argmax(stacked, axis=0)
    sigma_arr = np.asarray(sigmas, dtype=np.float64)
    sigma_map = sigma_arr[argmax]
    return out, enhanced, sigma_map


def apply_contrast(img: np.ndarray, params: dict) -> np.ndarray:
    method = str(params.get('contrast_method', 'percentile')).lower()
    if method == 'clahe':
        return apply_clahe(img, params.get('clahe_clip', 2.0), params.get('clahe_tile', 8))
    if method == 'gamma':
        img01 = _to_float01(img)
        g = float(params.get('gamma', 0.7))
        return as_u8((img01 ** g) * 255.0)
    if method == 'log':
        img01 = _to_float01(img)
        out = np.log1p(img01 * 9.0) / np.log1p(9.0)
        return as_u8(out * 255.0)
    # percentile stretch (default)
    return percentile_stretch(
        img,
        params.get('percentile_low', 1.0),
        params.get('percentile_high', 99.5),
    )


def _gaussian_psf(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    psf /= psf.sum()
    return psf


def apply_deconvolution(img: np.ndarray, params: dict) -> np.ndarray:
    method = str(params.get('deconv_method', 'none')).lower()
    if method in ('none', 'off', ''):
        return as_u8(img)

    img01 = _to_float01(img)
    sigma = float(params.get('deconv_psf_sigma', 1.2))
    k = max(5, int(round(sigma * 6)) | 1)

    if method == 'richardson_lucy':
        from skimage.restoration import richardson_lucy

        psf = _gaussian_psf(k, sigma)
        out = richardson_lucy(
            img01, psf, num_iter=int(params.get('deconv_iterations', 5)), clip=True)
        return as_u8(np.clip(out, 0, 1) * 255.0)

    if method == 'wiener':
        from scipy.signal import wiener

        out = wiener(img01, mysize=max(3, k // 2 | 1))
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return as_u8(np.clip(out, 0, 1) * 255.0)

    raise ValueError(f'Unknown deconv_method {method!r}')


def apply_background_correction(img: np.ndarray, params: dict) -> np.ndarray:
    method = str(params.get('bg_method', 'rolling_ball')).lower()
    if method == 'rolling_ball':
        return apply_rolling_ball(img, params.get('rolling_ball_radius', 25))
    if method == 'morph_open':
        return apply_morph_open_bg_subtract(img, params.get('morph_open_ksize', 31))
    if method in ('median', 'gaussian', 'tophat', 'bandpass'):
        return remove_background_from_stretch(as_u8(img), params, method)
    raise ValueError(f'Unknown bg_method {method!r}')


def particle_snr_pipeline_steps(
    img: np.ndarray, params: dict,
) -> list[tuple[str, np.ndarray]]:
    """
    Recommended order for sparse bright particles on dark background:
      1. background / illumination correction
      2. edge-preserving denoise
      3. optional mild deconvolution
      4. multi-scale DoG/LoG blob enhancement
      5. contrast / intensity normalization
    """
    steps: list[tuple[str, np.ndarray]] = []
    bg_method = params.get('bg_method', 'rolling_ball')
    current = apply_background_correction(img, params)
    steps.append((f'1. Background ({bg_method})', current))

    denoise = params.get('denoise_method', 'nlm')
    current = apply_denoise(current, params)
    steps.append((f'2. Denoise ({denoise})', current))

    deconv = str(params.get('deconv_method', 'none')).lower()
    if deconv not in ('none', 'off', ''):
        current = apply_deconvolution(current, params)
        steps.append((f'3. Deconvolution ({deconv})', current))

    blob = params.get('blob_method', 'dog')
    current = apply_multi_scale_blob(current, params)
    steps.append((f'{"4" if deconv not in ("none", "off", "") else "3"}. Multi-scale {blob.upper()}', current))

    contrast = params.get('contrast_method', 'percentile')
    current = apply_contrast(current, params)
    step_n = 5 if deconv not in ('none', 'off', '') else 4
    steps.append((f'{step_n}. Contrast ({contrast})', current))
    return steps


def apply_particle_snr_pipeline(img: np.ndarray, params: dict) -> np.ndarray:
    return particle_snr_pipeline_steps(img, params)[-1][1]


def detect_local_maxima(
    img: np.ndarray, params: dict, sigma_map: np.ndarray | None = None,
):
    """Local maxima on enhanced image (preferred after DoG/LoG)."""
    from skimage.feature import peak_local_max
    import pandas as pd

    img_f = np.asarray(img, dtype=np.float64)
    thr_rel = float(params.get('peak_threshold_rel', 0.15))
    abs_thr = thr_rel * float(img_f.max()) if img_f.max() > 0 else 0.0
    coords = peak_local_max(
        img_f,
        min_distance=int(params.get('peak_min_distance', 3)),
        threshold_abs=abs_thr,
        exclude_border=False,
    )
    if len(coords) == 0:
        return pd.DataFrame(columns=['y', 'x', 'mass', 'size', 'signal', 'sigma'])
    # peak_local_max returns (row, col) = (y, x)
    y = coords[:, 0].astype(float)
    x = coords[:, 1].astype(float)
    signal = img_f[coords[:, 0], coords[:, 1]]
    if sigma_map is not None:
        size = sigma_map[coords[:, 0], coords[:, 1]].astype(float)
    else:
        sigmas = tuple(params.get('blob_sigmas', (2.0,)))
        size = np.full(len(x), float(sigmas[len(sigmas) // 2]))
    return pd.DataFrame({
        'x': x,
        'y': y,
        'mass': signal,
        'size': size,
        'signal': signal,
        'sigma': size,
    })

def apply_percentile_bg_pipeline(img: np.ndarray, params: dict, bg_method: str) -> np.ndarray:
    return remove_background_from_stretch(stretch_frame(img, params), params, bg_method)


def apply_gaussian_percentile(img: np.ndarray, params: dict) -> np.ndarray:
    blurred = apply_gaussian_blur(img, params['gaussian_blur_ksize'])
    return stretch_frame(blurred, params)


def apply_percentile_gaussian_blur(img: np.ndarray, params: dict) -> np.ndarray:
    return apply_gaussian_blur(stretch_frame(img, params), params['gaussian_blur_ksize'])


def apply_gaussian_percentile_bg_pipeline(
    img: np.ndarray, params: dict, bg_method: str,
) -> np.ndarray:
    return remove_background_from_stretch(
        apply_gaussian_percentile(img, params), params, bg_method)


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
      2. Binary threshold (percentile or absolute)
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


def apply_bandpass_only(img: np.ndarray, params: dict) -> np.ndarray:
    return as_u8(apply_trackpy_bandpass(
        img, params['bandpass_lshort'], params['bandpass_llong']))


def bandpass_threshold_steps(
    img: np.ndarray, params: dict, *, with_blob: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Bandpass → hard threshold → optional multi-scale DoG/LoG max."""
    bp = apply_bandpass_only(img, params)
    thr_img, cutoff = hard_threshold(
        bp,
        percentile=float(params.get('bp_threshold_percentile', 90.0)),
        absolute=float(params.get('bp_threshold_abs', 0.0)),
    )
    abs_mode = float(params.get('bp_threshold_abs', 0.0)) > 0
    if abs_mode:
        thr_label = f'2. Threshold ≥ {cutoff:.0f} (absolute)'
    else:
        p = float(params.get('bp_threshold_percentile', 90.0))
        thr_label = f'2. Threshold ≥ {cutoff:.0f} (p{p:g})'
    steps = [
        (
            f'1. Bandpass ({params["bandpass_lshort"]}, {params["bandpass_llong"]})',
            bp,
        ),
        (thr_label, thr_img),
    ]
    if with_blob:
        blob = str(params.get('blob_method', 'dog')).upper()
        sigmas = tuple(params.get('blob_sigmas', (1.0, 1.5, 2.0, 3.0, 4.0)))
        blob_img = apply_multi_scale_blob(thr_img, params)
        steps.append((
            f'3. Multi-scale {blob} max σ={sigmas}',
            blob_img,
        ))
    return steps


def apply_bandpass_threshold(img: np.ndarray, params: dict) -> np.ndarray:
    """Final image of bandpass → threshold → multi-scale blob max."""
    return bandpass_threshold_steps(img, params, with_blob=True)[-1][1]


def refine_peak_centroids(
    features,
    refine_img: np.ndarray,
    *,
    radius: int = 3,
    sigma_col: str = 'sigma',
    guide_img: np.ndarray | None = None,
):
    """Snap DoG/LoG peaks onto particle centers in refine_img (usually raw).

    1) Optional guide (threshold) finds the particle island.
    2) Snap to brightest pixel in refine_img.
    3) Parabolic sub-pixel fit (same idea as trackpy) on a 3×3 neighborhood.
    """
    if features is None or not len(features):
        return features

    img = np.asarray(refine_img, dtype=np.float64)
    guide = np.asarray(guide_img, dtype=np.float64) if guide_img is not None else None
    h, w = img.shape
    xs, ys, masses = [], [], []

    def _quad_offset(vm, v0, vp):
        """Sub-pixel offset in [-0.5, 0.5] from 3 samples along one axis."""
        denom = (vm - 2.0 * v0 + vp)
        if abs(denom) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (vm - vp) / denom, -0.5, 0.5))

    for _, row in features.iterrows():
        x0 = float(row['x'])
        y0 = float(row['y'])
        rad = int(radius)
        if sigma_col in features.columns:
            rad = max(rad, int(np.ceil(2.5 * float(row[sigma_col]))))
        rad = max(rad, 4)

        if guide is not None:
            xi, yi = int(round(x0)), int(round(y0))
            x1, x2 = max(0, xi - rad), min(w, xi + rad + 1)
            y1, y2 = max(0, yi - rad), min(h, yi + rad + 1)
            gpatch = guide[y1:y2, x1:x2]
            if gpatch.size and float(gpatch.max()) > 0:
                gmax = float(gpatch.max())
                mask = gpatch >= max(1.0, 0.5 * gmax)
                yy, xx = np.mgrid[y1:y2, x1:x2]
                if np.any(mask):
                    x0 = float(xx[mask].mean())
                    y0 = float(yy[mask].mean())

        # Brightest pixel in refine_img near seed
        xi, yi = int(round(x0)), int(round(y0))
        x1, x2 = max(0, xi - rad), min(w, xi + rad + 1)
        y1, y2 = max(0, yi - rad), min(h, yi + rad + 1)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0 or float(patch.max()) <= 0:
            xs.append(x0)
            ys.append(y0)
            masses.append(float(row.get('mass', 0.0)))
            continue

        py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
        cx_i, cy_i = x1 + int(px), y1 + int(py)

        # Parabolic sub-pixel refinement on raw 3×3 (needs 1-pixel border)
        if 1 <= cx_i < w - 1 and 1 <= cy_i < h - 1:
            c = img[cy_i, cx_i]
            dx = _quad_offset(img[cy_i, cx_i - 1], c, img[cy_i, cx_i + 1])
            dy = _quad_offset(img[cy_i - 1, cx_i], c, img[cy_i + 1, cx_i])
            cx = float(cx_i) + dx
            cy = float(cy_i) + dy
            mass = float(c)
        else:
            cx, cy = float(cx_i), float(cy_i)
            mass = float(img[cy_i, cx_i])

        xs.append(cx)
        ys.append(cy)
        masses.append(mass)

    out = features.copy()
    out['x'] = xs
    out['y'] = ys
    out['mass'] = masses
    out['signal'] = masses
    return out


def deduplicate_features(features, min_distance: float = 3.0):
    """Keep highest-mass peak when refined positions collapse together."""
    if features is None or not len(features):
        return features
    out = features.sort_values('mass', ascending=False).copy()
    keep = []
    taken = np.zeros(len(out), dtype=bool)
    xy = out[['x', 'y']].to_numpy(dtype=np.float64)
    masses = out['mass'].to_numpy(dtype=np.float64)
    for i in range(len(out)):
        if taken[i]:
            continue
        keep.append(i)
        d2 = (xy[:, 0] - xy[i, 0]) ** 2 + (xy[:, 1] - xy[i, 1]) ** 2
        taken |= d2 < (min_distance ** 2)
    return out.iloc[keep].sort_index().reset_index(drop=True)


def detect_blobs_bandpass_threshold(img: np.ndarray, params: dict):
    """Bandpass → threshold → multi-scale DoG/LoG max → local maxima → centroid refine.

    Returns (features DataFrame, steps list, sigma_map for the processed image).
    """
    steps = bandpass_threshold_steps(img, params, with_blob=False)
    bp_img = steps[0][1]
    thr_img = steps[-1][1]
    blob_u8, blob_f, sigma_map = apply_multi_scale_blob(
        thr_img, params, return_sigma_map=True)
    blob = str(params.get('blob_method', 'dog')).upper()
    sigmas = tuple(params.get('blob_sigmas', (1.0, 1.5, 2.0, 3.0, 4.0)))
    steps.append((f'3. Multi-scale {blob} max σ={sigmas}', blob_u8))
    features = detect_local_maxima(blob_f, params, sigma_map=sigma_map)

    refine_on = str(params.get('refine_on', 'raw')).lower()
    if refine_on == 'bandpass':
        refine_img, guide = bp_img, thr_img
    elif refine_on == 'threshold':
        refine_img, guide = thr_img, thr_img
    else:
        # Default: guide with threshold, COM on raw (aligns with visible particles)
        refine_img, guide = img, thr_img
    features = refine_peak_centroids(
        features,
        refine_img,
        radius=int(params.get('refine_radius', 5)),
        guide_img=guide,
    )
    features = deduplicate_features(
        features,
        min_distance=float(params.get('peak_min_distance', 3)),
    )
    return features, steps, sigma_map


def bandpass_then_bg_steps(
    img: np.ndarray, params: dict, bg_method: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Bandpass first, then a background-subtraction method."""
    bg_method = bg_method or str(params.get('bandpass_bg_method', 'morph_open'))
    if bg_method not in BANDPASS_THEN_BG_METHODS:
        raise ValueError(
            f'Unknown bandpass_bg_method {bg_method!r}. '
            f'Use one of {BANDPASS_THEN_BG_METHODS}')

    bp = apply_bandpass_only(img, params)
    steps = [(
        f'1. Bandpass ({params["bandpass_lshort"]}, {params["bandpass_llong"]})',
        bp,
    )]
    after = remove_background_from_stretch(bp, params, bg_method)
    if bg_method == 'rolling_ball':
        label = f'2. Rolling-ball r={params.get("rolling_ball_radius", 25)}'
    elif bg_method == 'morph_open':
        label = f'2. Morph opening k={params.get("morph_open_ksize", 31)}'
    elif bg_method == 'gaussian':
        label = f'2. Gaussian bg k={params["bg_gaussian_ksize"]}'
    elif bg_method == 'median':
        label = f'2. Median bg k={params["bg_median_ksize"]}'
    else:
        label = f'2. Top-hat k={params["tophat_ksize"]}'
    steps.append((label, after))
    return steps


def apply_bandpass_then_bg(
    img: np.ndarray, params: dict, bg_method: str | None = None,
) -> np.ndarray:
    return bandpass_then_bg_steps(img, params, bg_method=bg_method)[-1][1]


def run_pipeline(frame: np.ndarray, pipeline_key: str, params: dict) -> PipelineResult:
    title = PIPELINE_CHOICES.get(pipeline_key, pipeline_key)
    steps: list[tuple[str, np.ndarray]] = [('Raw', frame)]

    if pipeline_key == 'raw':
        return PipelineResult(pipeline_key, title, steps, frame)

    if pipeline_key == 'particle_snr':
        for label, img in particle_snr_pipeline_steps(frame, params):
            steps.append((label, img))
        return PipelineResult(pipeline_key, title, steps, steps[-1][1])

    if pipeline_key == 'bandpass_threshold':
        for label, img in bandpass_threshold_steps(frame, params):
            steps.append((label, img))
        return PipelineResult(pipeline_key, title, steps, steps[-1][1])

    if pipeline_key == 'bandpass_then_bg':
        for label, img in bandpass_then_bg_steps(frame, params):
            steps.append((label, img))
        return PipelineResult(pipeline_key, title, steps, steps[-1][1])

    if pipeline_key.startswith('bandpass_') and pipeline_key not in (
        'bandpass_then_bg', 'bandpass_threshold',
    ):
        bg_method = pipeline_key.removeprefix('bandpass_')
        if bg_method not in BANDPASS_THEN_BG_METHODS:
            raise ValueError(f'Unknown pipeline {pipeline_key!r}')
        for label, img in bandpass_then_bg_steps(frame, params, bg_method=bg_method):
            steps.append((label, img))
        return PipelineResult(pipeline_key, title, steps, steps[-1][1])

    if pipeline_key == 'sobel_binary_expand':
        for label, img in sobel_binary_expand_steps(frame, params):
            steps.append((label, img))
        return PipelineResult(pipeline_key, title, steps, steps[-1][1])

    if pipeline_key == 'percentile_sobel_binary_expand':
        stretched = stretch_frame(frame, params)
        steps.append((
            f'Percentile ({params["percentile_low"]}–{params["percentile_high"]}%)',
            stretched,
        ))
        for label, img in sobel_binary_expand_steps(stretched, params):
            steps.append((label, img))
        return PipelineResult(pipeline_key, title, steps, steps[-1][1])

    if pipeline_key == 'rolling_ball':
        final = apply_rolling_ball(frame, params.get('rolling_ball_radius', 25))
        steps.append((
            f'Rolling-ball r={params.get("rolling_ball_radius", 25)}',
            final,
        ))
        return PipelineResult(pipeline_key, title, steps, final)

    if pipeline_key == 'percentile':
        stretched = stretch_frame(frame, params)
        steps.append((
            f'Percentile ({params["percentile_low"]}–{params["percentile_high"]}%)',
            stretched,
        ))
        return PipelineResult(pipeline_key, title, steps, stretched)

    if pipeline_key.startswith('percentile_') and pipeline_key != 'percentile_gaussian_blur':
        bg_method = pipeline_key.removeprefix('percentile_')
        if bg_method not in BG_METHODS:
            raise ValueError(f'Unknown pipeline {pipeline_key!r}')
        stretched = stretch_frame(frame, params)
        steps.append((
            f'Percentile ({params["percentile_low"]}–{params["percentile_high"]}%)',
            stretched,
        ))
        final = remove_background_from_stretch(stretched, params, bg_method)
        steps.append((f'Background removal ({bg_method})', final))
        return PipelineResult(pipeline_key, title, steps, final)

    if pipeline_key == 'gaussian_percentile':
        blurred = apply_gaussian_blur(frame, params['gaussian_blur_ksize'])
        steps.append((f'Gaussian blur k={params["gaussian_blur_ksize"]}', blurred))
        final = stretch_frame(blurred, params)
        steps.append((
            f'Percentile ({params["percentile_low"]}–{params["percentile_high"]}%)',
            final,
        ))
        return PipelineResult(pipeline_key, title, steps, final)

    if pipeline_key.startswith('gaussian_percentile_'):
        bg_method = pipeline_key.removeprefix('gaussian_percentile_')
        if bg_method not in BG_METHODS:
            raise ValueError(f'Unknown pipeline {pipeline_key!r}')
        blurred = apply_gaussian_blur(frame, params['gaussian_blur_ksize'])
        steps.append((f'Gaussian blur k={params["gaussian_blur_ksize"]}', blurred))
        stretched = stretch_frame(blurred, params)
        steps.append((
            f'Percentile ({params["percentile_low"]}–{params["percentile_high"]}%)',
            stretched,
        ))
        final = remove_background_from_stretch(stretched, params, bg_method)
        steps.append((f'Background removal ({bg_method})', final))
        return PipelineResult(pipeline_key, title, steps, final)

    if pipeline_key == 'percentile_gaussian_blur':
        stretched = stretch_frame(frame, params)
        steps.append((
            f'Percentile ({params["percentile_low"]}–{params["percentile_high"]}%)',
            stretched,
        ))
        final = apply_gaussian_blur(stretched, params['gaussian_blur_ksize'])
        steps.append((f'Gaussian blur k={params["gaussian_blur_ksize"]}', final))
        return PipelineResult(pipeline_key, title, steps, final)

    singles = {
        'tophat': lambda: apply_tophat(frame, params['tophat_ksize']),
        'trackpy_bandpass': lambda: as_u8(apply_trackpy_bandpass(
            frame, params['bandpass_lshort'], params['bandpass_llong'])),
        'clahe': lambda: apply_clahe(frame, params['clahe_clip'], params['clahe_tile']),
        'hist_eq': lambda: apply_hist_eq(frame),
    }
    if pipeline_key in singles:
        final = singles[pipeline_key]()
        steps.append((title, final))
        return PipelineResult(pipeline_key, title, steps, final)

    raise ValueError(f'Unknown pipeline {pipeline_key!r}')


def build_bandpass_then_bg_panels(
    frame: np.ndarray, params: dict,
) -> list[tuple[str, str, np.ndarray]]:
    """Bandpass first, then each background-subtraction method."""
    bp = apply_bandpass_only(frame, params)
    panels: list[tuple[str, str, np.ndarray]] = [
        ('raw', 'raw', frame),
        (
            'bandpass',
            f'1. bandpass ({params["bandpass_lshort"]}, {params["bandpass_llong"]})',
            bp,
        ),
    ]
    # Hard threshold (kills residual mottled floor)
    thr_img, cutoff = hard_threshold(
        bp,
        percentile=float(params.get('bp_threshold_percentile', 90.0)),
        absolute=float(params.get('bp_threshold_abs', 0.0)),
    )
    p = float(params.get('bp_threshold_percentile', 90.0))
    abs_mode = float(params.get('bp_threshold_abs', 0.0)) > 0
    thr_title = (
        f'2. threshold ≥ {cutoff:.0f} (abs)'
        if abs_mode else f'2. threshold ≥ {cutoff:.0f} (p{p:g})'
    )
    panels.append(('bandpass_threshold', thr_title, thr_img))

    after_methods = [
        ('rolling_ball', f'2. → rolling-ball r={params.get("rolling_ball_radius", 25)}'),
        ('morph_open', f'2. → morph-open k={params.get("morph_open_ksize", 31)}'),
        ('gaussian', f'2. → Gaussian bg k={params["bg_gaussian_ksize"]}'),
        ('median', f'2. → median bg k={params["bg_median_ksize"]}'),
        ('tophat', f'2. → top-hat k={params["tophat_ksize"]}'),
    ]
    for method, title in after_methods:
        panels.append((
            f'bandpass_{method}',
            title,
            remove_background_from_stretch(bp, params, method),
        ))
    return panels


def build_bandpass_threshold_panels(
    frame: np.ndarray, params: dict,
) -> list[tuple[str, str, np.ndarray]]:
    """Bandpass, then several hard-threshold levels for tuning."""
    bp = apply_bandpass_only(frame, params)
    panels: list[tuple[str, str, np.ndarray]] = [
        ('raw', 'raw', frame),
        (
            'bandpass',
            f'1. bandpass ({params["bandpass_lshort"]}, {params["bandpass_llong"]})',
            bp,
        ),
    ]
    for p in (80.0, 85.0, 90.0, 92.0, 95.0, 97.0):
        thr_img, cutoff = hard_threshold(bp, percentile=p)
        panels.append((
            f'thr_p{p:g}',
            f'→ thr ≥ {cutoff:.0f} (p{p:g})',
            thr_img,
        ))
    return panels


def build_stretch_background_panels(
    frame: np.ndarray, params: dict,
) -> list[tuple[str, str, np.ndarray]]:
    stretched = stretch_frame(frame, params)
    p_lo, p_hi = params['percentile_low'], params['percentile_high']
    panels = [
        ('percentile', f'Percentile ({p_lo}–{p_hi}%)', stretched),
    ]
    for method in BG_METHODS:
        key = f'percentile_{method}'
        if method == 'bandpass':
            label = f'→ bandpass ({params["bandpass_lshort"]}, {params["bandpass_llong"]})'
        elif method == 'tophat':
            label = f'→ top-hat k={params["tophat_ksize"]}'
        elif method == 'gaussian':
            label = f'→ Gaussian bg k={params["bg_gaussian_ksize"]}'
        else:
            label = f'→ median bg k={params["bg_median_ksize"]}'
        panels.append((
            key,
            label,
            remove_background_from_stretch(stretched, params, method),
        ))
    return panels


def build_gaussian_percentile_panels(
    frame: np.ndarray, params: dict,
) -> list[tuple[str, str, np.ndarray]]:
    p_lo, p_hi = params['percentile_low'], params['percentile_high']
    k = params['gaussian_blur_ksize']
    panels = [
        ('gaussian', f'Gaussian k={k}', apply_gaussian_blur(frame, k)),
        ('gaussian_percentile', f'→ percentile ({p_lo}–{p_hi}%)', apply_gaussian_percentile(frame, params)),
        ('percentile', f'Percentile only ({p_lo}–{p_hi}%)', stretch_frame(frame, params)),
        (
            'percentile_gaussian_blur',
            f'Percentile → Gaussian k={k}',
            apply_percentile_gaussian_blur(frame, params),
        ),
    ]
    for method in BG_METHODS:
        key = f'gaussian_percentile_{method}'
        panels.append((
            key,
            f'Gaussian → percentile → {method}',
            apply_gaussian_percentile_bg_pipeline(frame, params, method),
        ))
    return panels


def particles_in_roi(features, roi: list[int]):
    if features is None or not len(features):
        return features
    x0, y0, w, h = roi
    mask = (
        (features['x'] >= x0) & (features['x'] < x0 + w)
        & (features['y'] >= y0) & (features['y'] < y0 + h)
    )
    return features[mask].copy()


def shift_features_to_crop(features, roi: list[int]):
    if features is None or not len(features):
        return features
    out = features.copy()
    out['x'] = out['x'] - roi[0]
    out['y'] = out['y'] - roi[1]
    return out


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
