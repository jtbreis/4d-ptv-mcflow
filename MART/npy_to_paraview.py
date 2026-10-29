#!/usr/bin/env python3
"""
Convert a MART (or any) 3D NumPy volume to a ParaView-friendly VTK file.

``mart_test.py`` writes ``*.npy`` plus ``*.npy.meta.npz`` with ``grid_origin``,
``grid_spacing``, and ``dims``. This script loads those by default when the meta
file sits next to the array.

Outputs
-------
* **.vtk** (default): VTK legacy ``STRUCTURED_POINTS``, ``BINARY``, ``double``
  scalars in **big-endian** byte order (classic VTK legacy convention; ParaView
  reads this reliably).
* **.vti**: VTK XML ImageData with ``AppendedData`` raw blocks (length-prefixed
  little-endian ``UInt32`` + payload, matching ``byte_order="LittleEndian"``).

Examples::

  python MART/npy_to_paraview.py raw_data/mart_test/output/mart_volume.npy \\
      -o raw_data/mart_test/output/mart_volume.vtk

  python MART/npy_to_paraview.py volume.npy --meta volume.npy.meta.npz \\
      -o volume.vti --format vti

  python MART/npy_to_paraview.py volume.npy --origin -50,-70,-25 --spacing 6.25,6.25,6.25 \\
      -o volume.vtk
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def _parse_triplet(s: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated numbers, got {s!r}")
    return parts[0], parts[1], parts[2]


def _find_meta_npy(npy_path: Path) -> Path | None:
    """Match mart_test naming: ``stem.npy.meta.npz``."""
    cand = npy_path.parent / (npy_path.name + ".meta.npz")
    if cand.is_file():
        return cand
    cand2 = npy_path.with_suffix(".meta.npz")
    if cand2.is_file():
        return cand2
    return None


def _load_volume_and_geometry(
    npy_path: Path,
    meta_path: Path | None,
    origin_arg: str | None,
    spacing_arg: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    vol = np.load(npy_path)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D array in {npy_path}, got shape {vol.shape}")

    origin: np.ndarray | None = None
    spacing: np.ndarray | None = None

    if meta_path is not None and meta_path.is_file():
        m = np.load(meta_path)
        origin = np.asarray(m["grid_origin"], dtype=np.float64).ravel()[:3]
        spacing = np.asarray(m["grid_spacing"], dtype=np.float64).ravel()[:3]
    if origin_arg is not None:
        ox, oy, oz = _parse_triplet(origin_arg)
        origin = np.array([ox, oy, oz], dtype=np.float64)
    if spacing_arg is not None:
        sx, sy, sz = _parse_triplet(spacing_arg)
        spacing = np.array([sx, sy, sz], dtype=np.float64)

    if origin is None or spacing is None:
        raise SystemExit(
            "Need geometry: pass --meta (or rely on auto-discovered *.meta.npz), "
            "or both --origin and --spacing as x,y,z comma triplets."
        )

    title = npy_path.name
    return vol.astype(np.float64, copy=False), origin, spacing, title


def write_vtk_legacy_structured_points_binary(
    path: Path,
    volume: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    scalar_name: str,
    title: str,
) -> None:
    """
    VTK legacy ``STRUCTURED_POINTS`` binary file.

    Scalar payload uses **big-endian float64**, which matches the usual VTK
    legacy ``BINARY`` convention and loads correctly in ParaView.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vol = np.ascontiguousarray(volume, dtype=np.float64)
    nx, ny, nz = vol.shape
    npts = nx * ny * nz
    if npts != vol.size:
        raise RuntimeError("internal shape error")

    ox, oy, oz = (float(x) for x in origin.ravel()[:3])
    sx, sy, sz = (float(x) for x in spacing.ravel()[:3])

    header = (
        f"# vtk DataFile Version 3.0\n"
        f"{title}\n"
        f"BINARY\n"
        f"DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {nx} {ny} {nz}\n"
        f"ORIGIN {ox} {oy} {oz}\n"
        f"SPACING {sx} {sy} {sz}\n"
        f"POINT_DATA {npts}\n"
        f"SCALARS {scalar_name} double 1\n"
        f"LOOKUP_TABLE default\n"
    ).encode("ascii")

    # X varies fastest: C-order ravel matches VTK structured points layout.
    payload = vol.astype(">f8", copy=False).tobytes(order="C")

    with path.open("wb") as f:
        f.write(header)
        f.write(payload)


def write_vti_image_data_appended(
    path: Path,
    volume: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    scalar_name: str,
) -> None:
    """
    VTK XML ImageData with one ``Float64`` array in ``AppendedData`` (raw).

    Each appended block is ``UInt32`` length (little-endian) + payload bytes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vol = np.ascontiguousarray(volume, dtype="<f8")
    nx, ny, nz = vol.shape
    imax, jmax, kmax = nx - 1, ny - 1, nz - 1
    ox, oy, oz = (float(x) for x in origin.ravel()[:3])
    sx, sy, sz = (float(x) for x in spacing.ravel()[:3])

    raw = vol.tobytes(order="C")
    appended = struct.pack("<I", len(raw)) + raw

    head = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32">
  <ImageData WholeExtent="0 {imax} 0 {jmax} 0 {kmax}"
             Origin="{ox} {oy} {oz}"
             Spacing="{sx} {sy} {sz}">
    <Piece Extent="0 {imax} 0 {jmax} 0 {kmax}">
      <PointData Scalars="{scalar_name}">
        <DataArray type="Float64" Name="{scalar_name}" NumberOfComponents="1"
                   format="appended" offset="0"/>
      </PointData>
    </Piece>
  </ImageData>
  <AppendedData encoding="raw">
_"""

    tail = b"\n  </AppendedData>\n</VTKFile>\n"

    with path.open("wb") as f:
        f.write(head.encode("ascii"))
        f.write(appended)
        f.write(tail)


def _try_pyvista(path: Path, volume, origin, spacing, scalar_name: str) -> bool:
    try:
        import pyvista as pv
    except ImportError:
        return False

    grid = pv.ImageData()
    grid.dimensions = volume.shape
    grid.origin = tuple(float(x) for x in origin.ravel()[:3])
    grid.spacing = tuple(float(x) for x in spacing.ravel()[:3])
    # PyVista / VTK image ordering: Fortran flatten for point association
    grid.point_data[scalar_name] = np.ascontiguousarray(volume, dtype=np.float64).ravel(
        order="F"
    )
    grid.save(str(path))
    return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Convert 3D NumPy volume to VTK legacy (.vtk) or VTI (.vti) for ParaView"
    )
    p.add_argument("npy", type=str, help="Path to .npy volume (shape nx,ny,nz)")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output path; extension selects format if --format is omitted",
    )
    p.add_argument(
        "--format",
        choices=("vtk", "vti", "pyvista"),
        default=None,
        help="vtk=legacy binary (default from .vtk), vti=XML ImageData, "
        "pyvista=delegate to PyVista if installed",
    )
    p.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to .npz with grid_origin, grid_spacing (default: auto-detect)",
    )
    p.add_argument(
        "--origin",
        type=str,
        default=None,
        help="Override origin as x,y,z (comma-separated, same units as calibration)",
    )
    p.add_argument(
        "--spacing",
        type=str,
        default=None,
        help="Override voxel spacing dx,dy,dz",
    )
    p.add_argument(
        "--scalar-name",
        type=str,
        default="mart",
        help="Name of the scalar field in ParaView",
    )
    args = p.parse_args(argv)

    npy_path = Path(args.npy).resolve()
    out_path = Path(args.output).resolve()

    meta_path = Path(args.meta).resolve() if args.meta else None
    if meta_path is None:
        found = _find_meta_npy(npy_path)
        meta_path = found

    vol, origin, spacing, title = _load_volume_and_geometry(
        npy_path, meta_path, args.origin, args.spacing
    )

    fmt = args.format
    if fmt is None:
        suf = out_path.suffix.lower()
        if suf == ".vti":
            fmt = "vti"
        elif suf in (".vtk", ".pvtk"):
            fmt = "vtk"
        else:
            fmt = "vtk"
            if out_path.suffix == "":
                out_path = out_path.with_suffix(".vtk")

    print(f"Volume shape {vol.shape}, origin={origin}, spacing={spacing}", file=sys.stderr)
    print(f"Writing {fmt} → {out_path}", file=sys.stderr)

    if fmt == "pyvista":
        if not _try_pyvista(out_path, vol, origin, spacing, args.scalar_name):
            raise SystemExit("PyVista is not installed; use --format vtk or vti.")
        return

    if fmt == "vtk":
        write_vtk_legacy_structured_points_binary(
            out_path, vol, origin, spacing, args.scalar_name, title
        )
    elif fmt == "vti":
        write_vti_image_data_appended(out_path, vol, origin, spacing, args.scalar_name)
    else:
        raise SystemExit(f"Unknown format {fmt!r}")


if __name__ == "__main__":
    main()
