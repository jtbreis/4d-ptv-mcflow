import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from python_4dptv.utils.structure import Filenames
import os
from scipy.spatial import cKDTree


def compute_mean_stereomatching_error(path, remove_offset=False):
    """
    Compute mean distance error between stereo-matched 3D positions and known
    grid positions from Centers. Returns mean error in mm (no plotting).
    """
    path = path.rstrip(os.sep)
    with h5py.File(path + Filenames.STM.value, "r") as f:
        XYZe = np.empty(len(f.keys()), dtype=object)
        for frame_idx, frame in enumerate(f.values()):
            XYZe[frame_idx] = np.array(frame['xyze']).transpose()
        XYZstm = np.vstack(XYZe)

    centers_folder = None
    for subfolder in ("/Centers", "/Tests/Centers", "/Calibration/Tests/Centers/Camera"):
        cand = path + subfolder
        if os.path.isdir(cand) and any(f.endswith(".h5") for f in os.listdir(cand)):
            centers_folder = cand
            break
    if centers_folder is None:
        raise FileNotFoundError(
            f"Centers folder not found under {path}. Looked for .../Centers, .../Tests/Centers, "
            f".../Calibration/Tests/Centers/Camera"
        )
    h5_files = [os.path.join(centers_folder, fname) for fname in os.listdir(centers_folder) if fname.endswith(".h5")]
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
    distances, indices = tree.query(XYZstm[:, :3], k=1)
    distances = np.atleast_1d(np.squeeze(distances))
    indices = np.atleast_1d(np.squeeze(indices))
    if remove_offset:
        closest = XYZknown[indices]
        err_x = XYZstm[:, 0] - closest[:, 0]
        err_y = XYZstm[:, 1] - closest[:, 1]
        err_z = XYZstm[:, 2] - closest[:, 2]
        offset_x, offset_y, offset_z = np.mean(err_x), np.mean(err_y), np.mean(err_z)
        distances = np.sqrt((err_x - offset_x)**2 + (err_y - offset_y)**2 + (err_z - offset_z)**2)
    return float(np.mean(distances))


def compute_stereomatching_error_by_layers(path, calibration_layer_indices, remove_offset=False):
    """
    Compute mean stereomatching error (mm) separately on calibration layers and
    on in-between (hold-out) layers. Returns (mean_error_cal_layers, mean_error_in_between_layers).
    calibration_layer_indices: array or list of frame/layer indices used for calibration.
    """
    path = path.rstrip(os.sep)
    calibration_layer_indices = np.asarray(calibration_layer_indices, dtype=int)

    centers_folder = path.rstrip(os.sep) + "/Tests/Centers"
    if not os.path.isdir(centers_folder):
        raise FileNotFoundError(f"Centers folder not found: {centers_folder}")
    h5_files = [os.path.join(centers_folder, fn) for fn in os.listdir(centers_folder) if fn.endswith(".h5")]
    h5_files.sort()
    # Load known XYZ per frame from first camera
    with h5py.File(h5_files[0], "r") as fc:
        frame_keys = sorted(fc.keys(), key=lambda x: int(x.replace("frame", "")) if x.startswith("frame") else 0)
        XYZknown_per_frame = [np.atleast_2d(np.array(fc[k]['XYZ']))[:, :3] for k in frame_keys]

    mean_error_per_frame = []
    with h5py.File(path + Filenames.STM.value, "r") as f:
        keys = sorted(f.keys(), key=lambda x: int(x.replace("frame", "")) if x.startswith("frame") else 0)
        for frame_idx, key in enumerate(keys):
            frame = f[key]
            XYZstm_f = np.array(frame['xyze']).transpose()
            if XYZstm_f.size == 0:
                mean_error_per_frame.append(np.nan)
                continue
            XYZstm_f = np.atleast_2d(XYZstm_f)[:, :3]

            if frame_idx >= len(XYZknown_per_frame):
                mean_error_per_frame.append(np.nan)
                continue
            XYZknown_f = XYZknown_per_frame[frame_idx]
            if XYZknown_f.size == 0:
                mean_error_per_frame.append(np.nan)
                continue

            tree = cKDTree(XYZknown_f)
            distances, indices = tree.query(XYZstm_f, k=1)
            distances = np.atleast_1d(np.squeeze(distances))
            if remove_offset:
                indices = np.atleast_1d(np.squeeze(indices))
                closest = XYZknown_f[indices]
                err_x = XYZstm_f[:, 0] - closest[:, 0]
                err_y = XYZstm_f[:, 1] - closest[:, 1]
                err_z = XYZstm_f[:, 2] - closest[:, 2]
                offset_x, offset_y, offset_z = np.mean(err_x), np.mean(err_y), np.mean(err_z)
                distances = np.sqrt((err_x - offset_x)**2 + (err_y - offset_y)**2 + (err_z - offset_z)**2)
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

    If remove_offset is True, subtract the mean error in x, y, z before computing
    distances and plots, so only residual error is reported (e.g. for
    Calibration_After where the target was moved and a constant offset is expected).
    plot_suffix is appended to output filenames (e.g. '_after') when not empty.
    """
    
    with h5py.File(path + "/Tests" + Filenames.STM.value, "r") as f:
        XYZe = np.empty(len(f.keys()), dtype=object)
        for frame_idx, frame in enumerate(f.values()):
            XYZe[frame_idx] = np.array(frame['xyze']).transpose()
        XYZstm = np.vstack(XYZe)

    centers_folder = path.rstrip(os.sep) + "/Tests/Centers"
    if not os.path.isdir(centers_folder):
        raise FileNotFoundError(f"Centers folder not found: {centers_folder}")
    h5_files = [os.path.join(centers_folder, fname) for fname in os.listdir(centers_folder) if fname.endswith(".h5")]
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

    err_x = XYZstm[:, 0] - closest_matches[:, 0]
    err_y = XYZstm[:, 1] - closest_matches[:, 1]
    err_z = XYZstm[:, 2] - closest_matches[:, 2]

    if remove_offset:
        offset_x, offset_y, offset_z = np.mean(err_x), np.mean(err_y), np.mean(err_z)
        err_x = err_x - offset_x
        err_y = err_y - offset_y
        err_z = err_z - offset_z
        distances = np.sqrt(err_x**2 + err_y**2 + err_z**2)

    error_histogram(distances, path, plot_suffix=plot_suffix,
                   title_suffix=' (offset removed)' if remove_offset else '')

    unique_z = np.unique(XYZknown[:, 2])

    labels = ['X Error', 'Y Error', 'Z Error', 'Distance Error']
    errors = [err_x, err_y, err_z, distances]

    plot_mean_error_vs_z(unique_z, XYZstm, errors, path, plot_suffix=plot_suffix,
                         title_suffix=' (offset removed)' if remove_offset else '')

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
