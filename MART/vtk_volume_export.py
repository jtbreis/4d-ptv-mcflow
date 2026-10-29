"""
VTK volume writers for ParaView (used by ``mart_test.py`` and ``npy_to_paraview.py``).

Delegates to ``npy_to_paraview`` for both legacy ``.vtk`` and appended ``.vti``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from npy_to_paraview import (
    write_vti_image_data_appended,
    write_vtk_legacy_structured_points_binary,
)


def write_image_data_vti(
    path: str | Path,
    volume: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    scalar_name: str = "mart",
) -> None:
    """Save ``volume`` as ParaView-readable ``.vti`` (XML ImageData, appended raw)."""
    write_vti_image_data_appended(
        Path(path), volume, origin, spacing, scalar_name=scalar_name
    )


def write_structured_points_vtk(
    path: str | Path,
    volume: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    scalar_name: str = "mart",
    title: str | None = None,
) -> None:
    """Save ``volume`` as VTK legacy ``STRUCTURED_POINTS`` binary ``.vtk``."""
    p = Path(path)
    write_vtk_legacy_structured_points_binary(
        p,
        volume,
        origin,
        spacing,
        scalar_name=scalar_name,
        title=title if title is not None else p.stem,
    )
