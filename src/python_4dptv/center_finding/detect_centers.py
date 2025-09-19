import numpy as np
import trackpy as tp
import pims
import random
import matplotlib.pyplot as plt

from ..io.write_h5 import write_h5
import os


class CenterFinding():
    def __init__(self, path: str, output_path: str, particle_diameter: int, threshold: int = 10, minmass: int = 0):
        self.frames = pims.open(path)
        self.output_path = os.path.splitext(
            output_path + os.path.basename(path))[0] + '.h5'
        self.particle_diameter = particle_diameter
        self.threshold = threshold
        self.minmass = minmass
        self.separation = int(self.particle_diameter/2)
        self.frames_discarded = False

    def remove_frames(self, discard_frames=1, tracking_frames=4):
        if self.frames_discarded is True:
            print("Discard frames have already been removed!")
            return
        total_frames = len(self.frames)
        kept_indices = [i for i in range(
            total_frames) if i % (discard_frames+tracking_frames) != (discard_frames-1)]
        self.frames = [self.frames[i] for i in kept_indices]
        self.frames_discarded = True

    def find_centers(self, first_frame: int = 0, last_frame: int = -1):
        frames = self.frames[first_frame:last_frame]
        self.f = tp.batch(frames, self.particle_diameter,
                          threshold=self.threshold, minmass=self.minmass)

    def check_center_finding(self, nframe: int = 0, roi: list[int] = [100, 200, 100, 200], markersize=10):
        x_min, x_max = roi[0], roi[1]
        y_min, y_max = roi[2], roi[3]

        roi_points = self.f[(self.f['frame'] == nframe) &
                            (self.f['x'] >= x_min) & (self.f['x'] <= x_max) &
                            (self.f['y'] >= y_min) & (self.f['y'] <= y_max)]

        roi_points = roi_points.copy()
        roi_points['x'] = roi_points['x'] - x_min
        roi_points['y'] = roi_points['y'] - y_min

        tp.annotate(roi_points, self.frames[nframe][y_min:y_max,
                    x_min:x_max], plot_style={'markersize': markersize})

    def test_parameters(self, crop_size=100):
        sample_indices = random.sample(range(len(self.frames)), 4)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for i, idx in enumerate(sample_indices):
            frame = self.frames[idx]
            # Crop the image (e.g., center 100x100 region)
            h, w = frame.shape[:2]
            y0 = max(0, h // 2 - crop_size // 2)
            x0 = max(0, w // 2 - crop_size // 2)
            cropped = frame[y0:y0+crop_size, x0:x0+crop_size]
            features = tp.locate(cropped, self.particle_diameter,
                                 minmass=self.minmass, threshold=self.threshold, separation=self.separation)
            tp.annotate(features, cropped, ax=axes[i])
            axes[i].set_title(f'Frame {idx}')

        plt.tight_layout()
        plt.show()

    def write_matches(self):
        grouped = self.f.groupby('frame')
        data = {frame: group.reset_index(drop=True)
                for frame, group in grouped}
        metadata = {'particle diameter': self.particle_diameter, 'threshold': self.threshold,
                    'minmass': self.minmass, 'separation': self.separation}
        write_h5(filename=self.output_path, data=data, metadata=metadata)

    def check_mass_distribution(self):
        fig, ax = plt.subplots()
        ax.hist(self.f['mass'], bins=50)

        # Optionally, label the axes.
        ax.set(xlabel='mass', ylabel='count')
        plt.show()
