import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os


def plot_particles_xyze(xyze, boundingbox, path, frame):
    """
    Plots 3D particle positions with color-coded uncertainty.

    Parameters:
        xyze (np.ndarray): N x 4 array, columns are X, Y, Z, uncertainty.
    """
    x, y, z, e = xyze[0, :], xyze[1, :], xyze[2, :], xyze[3, :]
    nparticles = xyze.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, z, y, c=e, cmap='viridis', marker='o')
    cb = plt.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label('Error')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_xlim(boundingbox[0], boundingbox[1])
    ax.set_ylim(boundingbox[4], boundingbox[5])
    ax.set_zlim(boundingbox[2], boundingbox[3])
    plt.title(
        f'3D Particle Positions with Error / Matched Particles: {nparticles}')
    plt.tight_layout()
    out_dir = os.path.join(path, "matching_plots")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"matches_frame_{frame}.pdf"))
    plt.show()
