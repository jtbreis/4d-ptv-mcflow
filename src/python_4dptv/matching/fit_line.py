import os

import numpy as np

# Optional JIT for the per-ray SVD loop (calibration transform stays in skimage/PTVcalib).
_HAS_NUMBA = False
if os.environ.get("RAYS_DISABLE_NUMBA", "").lower() not in ("1", "true", "yes"):
    try:
        from numba import njit, prange

        _HAS_NUMBA = True
    except ImportError:
        pass


def _fit3dline_numpy(XYZ):
    xyz0 = np.mean(XYZ, axis=2)
    dd = np.zeros_like(xyz0)

    for i in range(XYZ.shape[0]):
        centerXYZ = XYZ[i, :, :]
        A = centerXYZ - xyz0[i][:, np.newaxis]
        _, _, Vt = np.linalg.svd(A.transpose())
        dd[i, :] = Vt[0]

    return xyz0, dd


if _HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _fit3dline_numba(XYZ):
        n = XYZ.shape[0]
        n_planes = XYZ.shape[2]
        xyz0 = np.empty((n, 3))
        dd = np.empty((n, 3))
        for i in prange(n):
            for j in range(3):
                s = 0.0
                for k in range(n_planes):
                    s += XYZ[i, j, k]
                xyz0[i, j] = s / n_planes

            A = np.empty((n_planes, 3))
            for p in range(n_planes):
                for j in range(3):
                    A[p, j] = XYZ[i, j, p] - xyz0[i, j]
            _, _, Vt = np.linalg.svd(A)
            dd[i, 0] = Vt[0, 0]
            dd[i, 1] = Vt[0, 1]
            dd[i, 2] = Vt[0, 2]
        return xyz0, dd


def fit3dline(XYZ):
    """
    Fit a 3D line per ray from multi-plane XYZ (n_rays, 3, n_planes).

    When Numba is installed (and RAYS_DISABLE_NUMBA is unset), uses a parallel JIT
    implementation of the SVD loop. Pixel→world calibration is unchanged.
    """
    if XYZ.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    if _HAS_NUMBA:
        return _fit3dline_numba(XYZ.astype(np.float64, copy=False))
    return _fit3dline_numpy(XYZ)
