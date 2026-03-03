import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from python_4dptv.utils.structure import Filenames
import os
from scipy.spatial import cKDTree


def _compute_distances_to_known(XYZstm, XYZknown, remove_offset=False):
    """
    Compute distances from stereo-matched points to nearest known grid point.
    XYZstm, XYZknown: (N, 3) and (M, 3). Returns distances (1d) and err_x, err_y, err_z.
    """
    tree = cKDTree(XYZknown)
    dists, indices = tree.query(XYZstm, k=1)
    dists = np.atleast_1d(np.squeeze(dists))
    indices = np.atleast_1d(np.squeeze(indices))
    closest = XYZknown[indices]
    err_x = XYZstm[:, 0] - closest[:, 0]
    err_y = XYZstm[:, 1] - closest[:, 1]
    err_z = XYZstm[:, 2] - closest[:, 2]
    if remove_offset:
        offset_x, offset_y, offset_z = np.mean(err_x), np.mean(err_y), np.mean(err_z)
        dists = np.sqrt((err_x - offset_x)**2 + (err_y - offset_y)**2 + (err_z - offset_z)**2)
    return dists, err_x, err_y, err_z


def _load_xyz_known_from_centers_folder(centers_folder):
    """Load and merge XYZ from all camera H5 files in centers_folder. Returns (N, 3) unique points."""
    h5_files = sorted(
        os.path.join(centers_folder, f) for f in os.listdir(centers_folder) if f.endswith(".h5")
    )
    camera_data = []
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            XYZk = np.vstack([np.array(f[k]["XYZ"]) for k in f.keys()])
        camera_data.append(XYZk)
    XYZknown = np.unique(np.vstack(camera_data), axis=0)
    return XYZknown[:, :3]


def compute_stereomatching_error_by_layers(path, calibration_layer_indices, remove_offset=False):
    """
    Compute mean stereomatching error (mm) separately on calibration layers and
    on in-between (hold-out) layers. Returns (mean_error_cal_layers, mean_error_in_between_layers).
    path: folder containing STM output and Centers (e.g. .../Tests).
    calibration_layer_indices: array or list of frame/layer indices used for calibration.
    """
    path = path.rstrip(os.sep)
    calibration_layer_indices = np.asarray(calibration_layer_indices, dtype=int)

    centers_folder = os.path.join(path, "Centers")
    if not os.path.isdir(centers_folder):
        raise FileNotFoundError(f"Centers folder not found: {centers_folder}")
    h5_files = sorted(os.path.join(centers_folder, f) for f in os.listdir(centers_folder) if f.endswith(".h5"))
    with h5py.File(h5_files[0], "r") as fc:
        frame_keys = sorted(fc.keys(), key=lambda x: int(x.replace("frame", "")) if x.startswith("frame") else 0)
        XYZknown_per_frame = [np.atleast_2d(np.array(fc[k]["XYZ"]))[:, :3] for k in frame_keys]

    mean_error_per_frame = []
    with h5py.File(path + Filenames.STM.value, "r") as f:
        keys = sorted(f.keys(), key=lambda x: int(x.replace("frame", "")) if x.startswith("frame") else 0)
        for frame_idx, key in enumerate(keys):
            XYZstm_f = np.atleast_2d(np.array(f[key]["xyze"]).transpose())[:, :3]
            if XYZstm_f.size == 0 or frame_idx >= len(XYZknown_per_frame):
                mean_error_per_frame.append(np.nan)
                continue
            XYZknown_f = XYZknown_per_frame[frame_idx]
            if XYZknown_f.size == 0:
                mean_error_per_frame.append(np.nan)
                continue
            distances, _, _, _ = _compute_distances_to_known(XYZstm_f, XYZknown_f, remove_offset)
            mean_error_per_frame.append(float(np.mean(distances)))

    mean_error_per_frame = np.array(mean_error_per_frame)
    n_frames = len(mean_error_per_frame)
    cal_idx_valid = calibration_layer_indices[calibration_layer_indices < n_frames]
    holdout_mask = np.ones(n_frames, dtype=bool)
    holdout_mask[cal_idx_valid] = False
    holdout_indices = np.where(holdout_mask)[0]

    cal_means = mean_error_per_frame[cal_idx_valid]
    cal_means = cal_means[~np.isnan(cal_means)]
    in_between_means = mean_error_per_frame[holdout_indices]
    in_between_means = in_between_means[~np.isnan(in_between_means)]

    mean_cal = float(np.mean(cal_means)) if len(cal_means) > 0 else np.nan
    mean_in_between = float(np.mean(in_between_means)) if len(in_between_means) > 0 else np.nan
    return mean_cal, mean_in_between


def evaluate_stereomatching_error(path, boundingbox, remove_offset=False, plot_suffix=''):
    """
    Evaluate stereo matching error (stereo XYZ vs known grid from Centers).
    path: folder that contains a "Tests" subfolder with STM output and Centers.

    If remove_offset is True, subtract the mean error in x, y, z before computing
    distances and plots (e.g. for Calibration_After where a constant offset is expected).
    plot_suffix is appended to output filenames (e.g. '_after') when not empty.
    """
    path = path.rstrip(os.sep)
    stm_path = os.path.join(path, "Tests") + Filenames.STM.value
    centers_folder = os.path.join(path, "Tests", "Centers")
    if not os.path.isdir(centers_folder):
        raise FileNotFoundError(f"Centers folder not found: {centers_folder}")

    with h5py.File(stm_path, "r") as f:
        XYZstm = np.vstack([np.array(frame["xyze"]).transpose() for frame in f.values()])[:, :3]

    XYZknown = _load_xyz_known_from_centers_folder(centers_folder)
    distances, err_x, err_y, err_z = _compute_distances_to_known(XYZstm, XYZknown, remove_offset)

    title_suffix = " (offset removed)" if remove_offset else ""
    error_histogram(distances, path, plot_suffix=plot_suffix, title_suffix=title_suffix)

    unique_z = np.unique(XYZknown[:, 2])

    labels = ['X Error', 'Y Error', 'Z Error', 'Distance Error']
    errors = [err_x, err_y, err_z, distances]

    plot_mean_error_vs_z(unique_z, XYZstm, errors, path, plot_suffix=plot_suffix, title_suffix=title_suffix)

    max_errors = [np.max(np.abs(err)) for err in errors]

    for z_val in unique_z:
        mask = np.abs(XYZstm[:, 2] - z_val) <= 0.3
        if np.sum(mask) == 0:
            continue
        fig, axes = plt.subplots(
            2, 2, subplot_kw={'projection': '3d'}, figsize=(16, 12))
        fig.suptitle(f'Error Plots for Z = {z_val}')
        for ax, err, label in zip(axes.flat, [e[mask] for e in errors], labels):
            # Orientation: X right, Y up, Z depth (plot as scatter(x, z, y))
            sc = ax.scatter(XYZstm[mask, 0], XYZstm[mask, 2], XYZstm[mask, 1],
                            c=err, cmap='viridis', s=30)
            ax.set_xlabel('X')
            ax.set_ylabel('Z (depth)')
            ax.set_zlabel('Y')
            ax.set_xlim(boundingbox[0], boundingbox[1])
            ax.set_ylim(boundingbox[4], boundingbox[5])
            ax.set_zlim(boundingbox[2], boundingbox[3])
            ax.view_init(elev=20, azim=-60)
            ax.set_title(label)
            plt.colorbar(sc, ax=ax)
            sc.set_clim(-max_errors[labels.index(label)],
                        max_errors[labels.index(label)])
        base = f"Error/planes/z_plane_{z_val}{plot_suffix}.pdf"
        plt.savefig(os.path.join(path, base))
        plt.tight_layout()
        plt.show()


def plot_mean_error_vs_z(unique_z, XYZstm, errors, folder, plot_suffix='', title_suffix=''):
    """Plot mean absolute error in X, Y, Z and mean distance vs Z location."""
    z_locations = []
    mean_x, mean_y, mean_z, mean_dist = [], [], [], []
    for z_val in unique_z:
        mask = np.abs(XYZstm[:, 2] - z_val) <= 0.3
        if np.sum(mask) == 0:
            continue
        z_locations.append(z_val)
        mean_x.append(np.mean(np.abs(errors[0][mask])))
        mean_y.append(np.mean(np.abs(errors[1][mask])))
        mean_z.append(np.mean(np.abs(errors[2][mask])))
        mean_dist.append(np.mean(errors[3][mask]))
    z_locations = np.array(z_locations)
    mean_x, mean_y, mean_z = np.array(mean_x), np.array(mean_y), np.array(mean_z)
    mean_dist = np.array(mean_dist)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_locations, mean_x, 'o-', label='Mean |X error|')
    ax.plot(z_locations, mean_y, 's-', label='Mean |Y error|')
    ax.plot(z_locations, mean_z, '^-', label='Mean |Z error|')
    ax.plot(z_locations, mean_dist, 'd-', label='Mean distance error')
    ax.set_xlabel('Z location')
    ax.set_ylabel('Mean error')
    ax.set_title('Mean stereomatching error vs Z location' + title_suffix)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"Error/mean_error_vs_z{plot_suffix}.pdf"))
    plt.show()


def error_histogram(distances, folder, plot_suffix='', title_suffix=''):
    mean_error = np.mean(distances)
    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2,
                label=f'Mean error = {mean_error:.4f}')
    plt.xlabel('Distance to Closest Match')
    plt.ylabel('Frequency')
    plt.title('Histogram of Stereomatching Error Distances' + title_suffix)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"Error/distance_histogram{plot_suffix}.pdf"))
    plt.show()
