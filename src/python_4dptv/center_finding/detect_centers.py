import numpy as np
import trackpy as tp
import pims
import random
import matplotlib.pyplot as plt


def find_centers(path, particle_radius, first_frame, last_frame):
    frames = pims.open(f'{path}')[first_frame:last_frame]
    f = tp.batch(frames, particle_radius, threshold=10)
    return f, frames


def check_center_finding(f, frames, nframe, roi: list[int] = [100, 200, 100, 200], markersize=10):
    x_min, x_max = roi[0], roi[1]
    y_min, y_max = roi[2], roi[3]

    roi_points = f[(f['frame'] == nframe) &
                   (f['x'] >= x_min) & (f['x'] <= x_max) &
                   (f['y'] >= y_min) & (f['y'] <= y_max)]

    roi_points = roi_points.copy()
    roi_points['x'] = roi_points['x'] - x_min
    roi_points['y'] = roi_points['y'] - y_min

    tp.annotate(roi_points, frames[nframe][y_min:y_max,
                x_min:x_max], plot_style={'markersize': markersize})


def test_parameters(path, particle_diameter, threshold=1023/255, minmass=0, crop_size=100):
    frames = pims.open(f'{path}')
    num_frames = len(frames)

    sample_indices = random.sample(range(num_frames), 4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, idx in enumerate(sample_indices):
        frame = frames[idx]
        # Crop the image (e.g., center 100x100 region)
        h, w = frame.shape[:2]
        y0 = max(0, h // 2 - crop_size // 2)
        x0 = max(0, w // 2 - crop_size // 2)
        cropped = frame[y0:y0+crop_size, x0:x0+crop_size]
        features = tp.locate(cropped, particle_diameter,
                             minmass=minmass, threshold=threshold, separation=int(particle_diameter/2))
        tp.annotate(features, cropped, ax=axes[i])
        axes[i].set_title(f'Frame {idx}')

    plt.tight_layout()
    plt.show()
