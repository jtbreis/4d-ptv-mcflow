import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from python_4dptv.utils.structure import Filenames
import os
from scipy.spatial import cKDTree


def evaluate_stereomatching_error(path, boundingbox):
    with h5py.File(path + Filenames.STM.value, "r") as f:
        XYZe = np.empty(len(f.keys()), dtype=object)
        for frame_idx, frame in enumerate(f.values()):
            XYZe[frame_idx] = np.array(frame['xyze']).transpose()
        XYZstm = np.vstack(XYZe)

    centers_folder = os.path.join(path, "Centers")
    h5_files = [os.path.join(centers_folder, fname) for fname in os.listdir(
        centers_folder) if fname.endswith(".h5")]
    h5_files.sort()

    camera_data = np.empty(len(h5_files), dtype=object)
    for camidx, h5_file in enumerate(h5_files):
        with h5py.File(h5_file, "r") as f:
            XYZk = np.empty(len(f.keys()), dtype=object)
            for frame_idx, frame in enumerate(f.values()):
                XYZk[frame_idx] = np.array(frame['XYZ'])
        camera_data[camidx] = np.vstack(XYZk)

    XYZknown = np.vstack(camera_data)
    XYZknown = np.unique(XYZknown, axis=0)

    tree = cKDTree(XYZknown)
    distances, indices = tree.query(XYZstm[:, :-1])
    closest_matches = XYZknown[indices]

    error_histogram(distances, path)

    unique_z = np.unique(XYZknown[:, 2])

    labels = ['X Error', 'Y Error', 'Z Error', 'Distance Error']
    errors = [XYZstm[:, 0] - closest_matches[:, 0],
              XYZstm[:, 1] - closest_matches[:, 1],
              XYZstm[:, 2] - closest_matches[:, 2],
              distances]

    max_errors = [np.max(np.abs(err)) for err in errors]

    for z_val in unique_z:
        mask = np.abs(XYZstm[:, 2] - z_val) <= 1.0
        if np.sum(mask) == 0:
            continue
        fig, axes = plt.subplots(
            2, 2, subplot_kw={'projection': '3d'}, figsize=(16, 12))
        fig.suptitle(f'Error Plots for Z = {z_val}')
        for ax, err, label in zip(axes.flat, [e[mask] for e in errors], labels):
            sc = ax.scatter(XYZstm[mask, 0], XYZstm[mask, 1], XYZstm[mask, 2],
                            c=err, cmap='viridis', s=30)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(boundingbox[0], boundingbox[1])
            ax.set_ylim(boundingbox[2], boundingbox[3])
            ax.set_zlim(boundingbox[4], boundingbox[5])
            ax.set_title(label)
            plt.colorbar(sc, ax=ax)
            sc.set_clim(-max_errors[labels.index(label)],
                        max_errors[labels.index(label)])
        plt.savefig(os.path.join(path, f"error_plots/z_plane_{z_val}.pdf"))
        plt.tight_layout()
        plt.show()


def error_histogram(distances, folder):
    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Distance to Closest Match')
    plt.ylabel('Frequency')
    plt.title('Histogram of Stereomatching Error Distances')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "error_plots/distance_histogram.pdf"))
    plt.show()
