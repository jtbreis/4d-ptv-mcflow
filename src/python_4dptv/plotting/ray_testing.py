import numpy as np

import matplotlib.pyplot as plt


def plot_rays(XYZ, xyz0, dd, cameras, nrays=10, frame=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=0, azim=0)
    colors = plt.cm.viridis(np.linspace(0, 1, XYZ[0][frame].shape[2]))
    cam_colors = plt.cm.tab20b(np.linspace(0, 1, len(cameras)))
    for cam in cameras:
        for layer in range(XYZ[cam][frame].shape[2]):
            ax.scatter(XYZ[cam][frame][:nrays, 0, layer], XYZ[cam][frame][:nrays, 1, layer],
                       XYZ[cam][frame][:nrays, 2, layer], c=cam_colors[cam], marker='o')

            ax.scatter(xyz0[cam][frame][:nrays, 0], xyz0[cam][frame][:nrays, 1],
                       xyz0[cam][frame][:nrays, 2], c='r', marker='x')
            ax.quiver3D(xyz0[cam][frame][:nrays, 0], xyz0[cam][frame][:nrays, 1], xyz0[cam][frame][:nrays, 2],
                        dd[cam][frame][:nrays, 0], dd[cam][frame][:nrays, 1], dd[cam][frame][:nrays, 2], length=100, label=f'Camera {cam}' if layer == 0 else None, color=cam_colors[cam])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()
