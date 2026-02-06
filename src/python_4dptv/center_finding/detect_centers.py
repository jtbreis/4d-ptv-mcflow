import numpy as np
import trackpy as tp
import pims
import random
import matplotlib.pyplot as plt
import os

from ..io.write_h5 import write_h5
from ..utils.structure import Folders


class CenterFinding():
    def __init__(self, path: str, output_path: str, particle_diameter: int, threshold: int = 10, minmass: int = 0):
        self.frames = pims.open(path)
        os.makedirs(output_path + Folders.CENTERS.value, exist_ok=True)
        self.output_path = os.path.splitext(
            output_path + Folders.CENTERS.value + '/' + os.path.basename(path))[0] + '.h5'
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
                          threshold=self.threshold, minmass=self.minmass, separation=self.separation)

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

    def test_parameters(self, crop_size=100, start_frame=0, nframes=200, output=None):
        sample_indices = range(start_frame, start_frame+4)

        layout = [
            ["a", "b", "c", "d", "e", "e"],
            ["f", "g", "h", "i", "e", "e"],
        ]

        fig, axes = plt.subplot_mosaic(layout, figsize=(18, 8))
        fig.suptitle(
            f'Particle diameter: {self.particle_diameter}, Threshold: {self.threshold}, Seperation: {self.separation}, Minmass: {self.minmass}')

        for i, idx in enumerate(sample_indices):
            frame = self.frames[idx]
            # Crop the image (e.g., center 100x100 region)
            h, w = frame.shape[:2]
            y0 = max(0, h // 2 - crop_size // 2)
            x0 = max(0, w // 2 - crop_size // 2)
            cropped = frame[y0:y0+crop_size, x0:x0+crop_size]
            features = tp.locate(cropped, self.particle_diameter,
                                 minmass=self.minmass, threshold=self.threshold, separation=self.separation)

            tp.annotate(features, cropped,
                        ax=axes[layout[0][i]], imshow_style={'vmax': 50})
            mass = tp.locate(frame, self.particle_diameter,
                             minmass=self.minmass, threshold=self.threshold, separation=self.separation)
            axes[layout[0][i]].set_title(f'Frame {idx}')
            axes[layout[1][i]].hist(mass['mass'], bins=50)
            axes[layout[1][i]].set_title(f'#particles: {len(mass)}')

        self.find_centers(start_frame, start_frame+nframes)
        axes["e"].hist(self.f['mass'], bins=200)

        plt.tight_layout()
        if output is not None:
            plt.savefig(output, format='pdf')
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
        ax.hist(self.f['mass'], bins=300)

        # Optionally, label the axes.
        ax.set(xlabel='mass', ylabel='count')
        plt.show()
