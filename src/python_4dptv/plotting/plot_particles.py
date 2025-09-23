import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_particles_xyze(xyze):
    """
    Plots 3D particle positions with color-coded uncertainty.

    Parameters:
        xyze (np.ndarray): N x 4 array, columns are X, Y, Z, uncertainty.
    """
    x, y, z, e = xyze[0, :], xyze[1, :], xyze[2, :], xyze[3, :]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=e, cmap='viridis', marker='o')
    cb = plt.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label('Uncertainty')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Particle Positions with Uncertainty')
    plt.tight_layout()
    plt.show()
