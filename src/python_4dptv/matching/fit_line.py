import numpy as np


def fit3dline(XYZ):
    xyz0 = np.mean(XYZ, axis=2)
    dd = np.zeros_like(xyz0)

    for i in range(XYZ.shape[0]):
        centerXYZ = XYZ[i, :, :]
        A = centerXYZ - xyz0[i][:, np.newaxis]
        _, _, Vt = np.linalg.svd(A.transpose())
        dd[i, :] = Vt[0]

    return xyz0, dd
