"""
Post-process tracked particles from an h5part file: keep only 4-frame tracks,
fit a cubic B-spline per track, compute velocity and acceleration, and write
(1) a _bspline.h5part with only tracked particles (4 positions per track) and
    B-spline-derived v/a, and
(2) a B-spline curves file where each snapshot (4-frame block) is stored at a
    different time step.

Example:
  python Post-Processing/bspline_4frame_tracks.py \\
    /workspaces/4d-ptv-mcflow/data/sensitivity_box_size/tracking_test_threshold1/tracks_box_size_0.25.h5part

Outputs (same directory as input):
  - tracks_box_size_0.25_bspline.h5part   (particles + v/a from B-spline)
  - tracks_box_size_0.25_bspline_curves.h5 + .xmf (B-spline polylines per snapshot, ParaView-readable)
"""
import argparse
import os
import re
import sys

import h5py
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional: ensure 4BE-ETI is importable when run from repo (e.g. pip install -e 4BE-ETI)
try:
    from python_4be_eti.track import (
        velocity_acceleration_from_bspline,
        sample_bspline_curve,
    )
except ImportError:
    # Fallback: use scipy directly if 4BE-ETI not installed
    from scipy.interpolate import make_interp_spline

    def velocity_acceleration_from_bspline(t, X, Y, Z):
        t = np.asarray(t, dtype=np.float64)
        X, Y, Z = np.asarray(X, np.float64), np.asarray(
            Y, np.float64), np.asarray(Z, np.float64)
        n = len(t)
        if n < 4:
            raise ValueError(
                "B-spline fit requires at least 4 points, got %d" % n)
        spl_x = make_interp_spline(t, X, k=3)
        spl_y = make_interp_spline(t, Y, k=3)
        spl_z = make_interp_spline(t, Z, k=3)
        vx = spl_x.derivative(1)(t)
        vy = spl_y.derivative(1)(t)
        vz = spl_z.derivative(1)(t)
        ax = spl_x.derivative(2)(t)
        ay = spl_y.derivative(2)(t)
        az = spl_z.derivative(2)(t)
        return vx, vy, vz, ax, ay, az

    def sample_bspline_curve(t, X, Y, Z, num_samples=50):
        t = np.asarray(t, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        n = len(t)
        if n < 4:
            raise ValueError(
                "B-spline curve requires at least 4 points, got %d" % n)
        t_flat = np.linspace(t[0], t[-1], num_samples)
        spl_x = make_interp_spline(t, X, k=3)
        spl_y = make_interp_spline(t, Y, k=3)
        spl_z = make_interp_spline(t, Z, k=3)
        return t_flat, spl_x(t_flat), spl_y(t_flat), spl_z(t_flat)


# XDMF Mixed topology: 2 = POLYLINE (VTK)
_POLYLINE_TYPE = 2


def _polylines_to_geometry(polylines: list):
    """Build global Points (Nx3) and Connectivity (VTK Mixed polyline) from list of (Nx3) arrays."""
    if not polylines:
        return np.zeros((0, 3), dtype=np.float64), np.array([], dtype=np.int32)
    points_list = []
    connectivity_list = []
    pt_offset = 0
    for pts in polylines:
        pts = np.asarray(pts, dtype=np.float64)
        n_pts = len(pts)
        points_list.append(pts)
        connectivity_list.append(
            [_POLYLINE_TYPE, n_pts] + list(range(pt_offset, pt_offset + n_pts)))
        pt_offset += n_pts
    points = np.vstack(points_list)
    connectivity = np.array(
        [x for c in connectivity_list for x in c], dtype=np.int32)
    return points, connectivity


def _write_bspline_curves_h5_xmf(
    out_curves_h5: str,
    out_curves_xmf: str,
    all_curve_data: list,
    num_blocks: int,
) -> None:
    """Write B-spline curves to HDF5 + XDMF so ParaView can open them (one grid per snapshot)."""
    if num_blocks <= 0:
        with h5py.File(out_curves_h5, "w") as f:
            grp = f.create_group("Step0")
            grp.create_dataset("Points", data=np.zeros(
                (0, 3), dtype=np.float64))
            grp.create_dataset(
                "Connectivity", data=np.array([], dtype=np.int32))
        h5_name = os.path.basename(out_curves_h5)
        xdmf_lines = [
            '<?xml version="1.0" ?>',
            '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
            '<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
            '  <Domain>',
            '    <Grid Name="BsplineCurves" GridType="Collection" CollectionType="Temporal">',
            '      <Grid Name="Step0">',
            '        <Topology TopologyType="Mixed" NumberOfElements="0">',
            '          <DataItem Format="HDF" DataType="Int" Dimensions="0">%s:/Step0/Connectivity</DataItem>' % h5_name,
            '        </Topology>',
            '        <Geometry GeometryType="XYZ">',
            '          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="0 3">%s:/Step0/Points</DataItem>' % h5_name,
            '        </Geometry>',
            '      </Grid>',
            '    </Grid>',
            '  </Domain>',
            '</Xdmf>',
        ]
        with open(out_curves_xmf, "w") as f:
            f.write("\n".join(xdmf_lines))
        return
    h5_name = os.path.basename(out_curves_h5)
    n_cells_per_block = []
    with h5py.File(out_curves_h5, "w") as f:
        for block_idx in range(num_blocks):
            curves_in_block = [
                (tid, t, xc, yc, zc)
                for bidx, (tid, t, xc, yc, zc) in all_curve_data
                if bidx == block_idx
            ]
            polylines = [
                np.column_stack([x_curve, y_curve, z_curve]).astype(np.float64)
                for tid, t, x_curve, y_curve, z_curve in curves_in_block
            ]
            n_cells_per_block.append(len(polylines))
            points, connectivity = _polylines_to_geometry(polylines)
            grp = f.create_group("Step%d" % block_idx)
            grp.create_dataset("Points", data=points)
            grp.create_dataset("Connectivity", data=connectivity)

    # XDMF: temporal collection, one Grid per snapshot (ParaView time steps)
    xdmf_lines = [
        '<?xml version="1.0" ?>',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
        '<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">',
        '  <Domain>',
        '    <Grid Name="BsplineCurves" GridType="Collection" CollectionType="Temporal">',
    ]
    with h5py.File(out_curves_h5, "r") as f:
        for block_idx in range(num_blocks):
            grp_name = "Step%d" % block_idx
            n_cells = n_cells_per_block[block_idx]
            n_conn = f[grp_name]["Connectivity"].shape[0]
            n_pts = f[grp_name]["Points"].shape[0]
            xdmf_lines.extend([
                '      <Grid Name="%s">' % grp_name,
                '        <Topology TopologyType="Mixed" NumberOfElements="%d">' % (
                    n_cells,),
                '          <DataItem Format="HDF" DataType="Int" Dimensions="%d">%s:/%s/Connectivity</DataItem>' % (
                    n_conn, h5_name, grp_name),
                '        </Topology>',
                '        <Geometry GeometryType="XYZ">',
                '          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="%d 3">%s:/%s/Points</DataItem>' % (
                    n_pts, h5_name, grp_name),
                '        </Geometry>',
                '      </Grid>',
            ])
    xdmf_lines.extend([
        '    </Grid>',
        '  </Domain>',
        '</Xdmf>',
    ])
    with open(out_curves_xmf, "w") as f:
        f.write("\n".join(xdmf_lines))


def _step_frame_num(key: str) -> int:
    m = re.match(r"Step#(\d+)", key)
    return int(m.group(1)) if m else -1


def load_tracks_raw(filepath: str):
    """
    Load from h5part into: step_keys (sorted), tracks_raw, has_props, props_ncams.

    tracks_raw: dict track_id -> list of (frame, x, y, z [, d, i, m]) where d,i,m
    are scalars (legacy) or 1d arrays of length n_cams (per-ray from stereomatching).
    props_ncams: 0 if not has_props, else number of cameras per particle (1 = scalar).
    """
    with h5py.File(filepath, "r") as f:
        step_keys = [k for k in f.keys() if k.startswith("Step#")]
        if not step_keys:
            raise ValueError("No Step# groups in %s" % filepath)
        step_keys.sort(key=_step_frame_num)
        has_props = "diameter" in f[step_keys[0]]
        props_ncams = 0

        tracks_raw = {}
        for key in step_keys:
            frame = _step_frame_num(key)
            grp = f[key]
            x = np.asarray(grp["x"]).ravel()
            y = np.asarray(grp["y"]).ravel()
            z = np.asarray(grp["z"]).ravel()
            id_ = np.asarray(grp["id"]).ravel().astype(int)
            if has_props:
                d = np.asarray(grp["diameter"])
                intensity = np.asarray(grp["intensity"])
                m = np.asarray(grp["mass"])
                if d.ndim == 2 and d.shape[1] > 0:
                    props_ncams = max(props_ncams, d.shape[1])
                elif d.ndim == 1 and d.size > 0:
                    props_ncams = max(props_ncams, 1)
                props_2d = d.ndim == 2
            for i in range(len(id_)):
                tid = id_[i]
                if tid not in tracks_raw:
                    tracks_raw[tid] = []
                if has_props:
                    if props_2d:
                        tracks_raw[tid].append((
                            frame, x[i], y[i], z[i],
                            np.asarray(d[i], dtype=np.float64).ravel(),
                            np.asarray(intensity[i], dtype=np.float64).ravel(),
                            np.asarray(m[i], dtype=np.float64).ravel(),
                        ))
                    else:
                        tracks_raw[tid].append((
                            frame, x[i], y[i], z[i],
                            float(d.ravel()[i]),
                            float(intensity.ravel()[i]),
                            float(m.ravel()[i]),
                        ))
                else:
                    tracks_raw[tid].append(
                        (frame, x[i], y[i], z[i], np.nan, np.nan, np.nan))

        if has_props and props_ncams == 0:
            props_ncams = 1

    return step_keys, tracks_raw, has_props, props_ncams


def get_four_frame_blocks(step_keys):
    """Group sorted step frame numbers into consecutive blocks of 4. Returns list of (frame0, frame1, frame2, frame3)."""
    frames = sorted([_step_frame_num(k) for k in step_keys])
    if not frames:
        return []
    blocks = []
    i = 0
    while i + 4 <= len(frames):
        a, b, c, d = frames[i], frames[i + 1], frames[i + 2], frames[i + 3]
        if b == a + 1 and c == b + 1 and d == c + 1:
            blocks.append((a, b, c, d))
            i += 4
        else:
            i += 1
    return blocks


def build_block_tracks(tracks_raw, block_frames, dt=1.0):
    """
    For one 4-frame block, get tracks that have exactly these 4 frames.
    Returns list of (track_id, frames_arr, x_arr, y_arr, z_arr, vx, vy, vz, ax, ay, az, d, i, m).
    """
    block_set = set(block_frames)
    out = []
    for tid, points in tracks_raw.items():
        points.sort(key=lambda p: p[0])
        frames_in_block = [p[0] for p in points if p[0] in block_set]
        if sorted(frames_in_block) != list(block_frames):
            continue
        # exactly these 4 points
        points_block = [p for p in points if p[0] in block_set]
        points_block.sort(key=lambda p: p[0])
        frames_arr = np.array([p[0] for p in points_block])
        x_arr = np.array([p[1] for p in points_block], dtype=np.float64)
        y_arr = np.array([p[2] for p in points_block], dtype=np.float64)
        z_arr = np.array([p[3] for p in points_block], dtype=np.float64)
        d = np.stack([np.atleast_1d(p[4]) for p in points_block])
        i = np.stack([np.atleast_1d(p[5]) for p in points_block])
        m = np.stack([np.atleast_1d(p[6]) for p in points_block])

        t = frames_arr.astype(np.float64) * dt
        try:
            vx, vy, vz, ax, ay, az = velocity_acceleration_from_bspline(
                t, x_arr, y_arr, z_arr)
        except Exception:
            continue
        out.append((tid, frames_arr, x_arr, y_arr, z_arr,
                   vx, vy, vz, ax, ay, az, d, i, m))
    return out


def run(
    filepath: str,
    dt: float = 1.0,
    num_curve_samples: int = 50,
    output_dir: str | None = None,
):
    """
    Load filepath (e.g. .../tracks_box_size_0.25.h5part), compute B-spline v/a for
    4-frame tracks, write _bspline.h5part and _bspline_curves_StepN.vtk (one VTK per snapshot).
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(filepath))
    base = os.path.basename(filepath)
    if base.endswith(".h5part"):
        stem = base[: -len(".h5part")]
    else:
        stem = base
    out_bspline_h5 = os.path.join(output_dir, stem + "_bspline.h5part")
    out_curves_h5 = os.path.join(output_dir, stem + "_bspline_curves.h5")
    out_curves_xmf = os.path.join(output_dir, stem + "_bspline_curves.xmf")

    step_keys, tracks_raw, has_props, props_ncams = load_tracks_raw(filepath)
    blocks = get_four_frame_blocks(step_keys)
    if not blocks:
        # Single block: use all frames if they form one contiguous 4-frame set
        frames = sorted([_step_frame_num(k) for k in step_keys])
        if len(frames) >= 4:
            # take first 4 consecutive
            for j in range(len(frames) - 3):
                a, b, c, d = frames[j], frames[j +
                                               1], frames[j + 2], frames[j + 3]
                if b == a + 1 and c == b + 1 and d == c + 1:
                    blocks = [(a, b, c, d)]
                    break
        if not blocks:
            raise ValueError(
                "No consecutive 4-frame block found in %s (frames: %s)" % (
                    filepath, frames)
            )

    # Per-frame: (track_id, x, y, z, vx, vy, vz, ax, ay, az, d, i, m) for 4-frame tracks in the block containing that frame
    # frame -> list of (tid, x, y, z, vx, vy, vz, ax, ay, az, d, i, m)
    frame_to_particles = {}
    for key in step_keys:
        frame_to_particles[_step_frame_num(key)] = []

    # list of (block_index, list of (track_id, t, x_curve, y_curve, z_curve))
    all_curve_data = []

    for block_idx, block_frames in enumerate(blocks):
        block_tracks = build_block_tracks(tracks_raw, block_frames, dt)
        for (
            tid,
            frames_arr,
            x_arr,
            y_arr,
            z_arr,
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
            d,
            i,
            m,
        ) in block_tracks:
            for idx, frame in enumerate(frames_arr):
                frame_to_particles[int(frame)].append(
                    (
                        tid,
                        float(x_arr[idx]),
                        float(y_arr[idx]),
                        float(z_arr[idx]),
                        float(vx[idx]),
                        float(vy[idx]),
                        float(vz[idx]),
                        float(ax[idx]),
                        float(ay[idx]),
                        float(az[idx]),
                        np.asarray(d[idx], dtype=np.float64).ravel(),
                        np.asarray(i[idx], dtype=np.float64).ravel(),
                        np.asarray(m[idx], dtype=np.float64).ravel(),
                    )
                )
            # B-spline curve for this 4-frame track
            t = frames_arr.astype(np.float64) * dt
            try:
                _, x_curve, y_curve, z_curve = sample_bspline_curve(
                    t, x_arr, y_arr, z_arr, num_samples=num_curve_samples
                )
            except Exception:
                continue
            all_curve_data.append(
                (block_idx, (tid, t, x_curve, y_curve, z_curve)))

    def write_outputs():
        with h5py.File(out_bspline_h5, "w") as f:
            for key in step_keys:
                frame = _step_frame_num(key)
                particles = frame_to_particles.get(frame, [])
                if not particles:
                    grp = f.create_group(key)
                    grp.create_dataset("id", data=np.array([], dtype=np.int64))
                    grp.create_dataset(
                        "x", data=np.array([], dtype=np.float64))
                    grp.create_dataset(
                        "y", data=np.array([], dtype=np.float64))
                    grp.create_dataset(
                        "z", data=np.array([], dtype=np.float64))
                    grp.create_dataset(
                        "vx", data=np.array([], dtype=np.float64))
                    grp.create_dataset(
                        "vy", data=np.array([], dtype=np.float64))
                    grp.create_dataset(
                        "vz", data=np.array([], dtype=np.float64))
                    grp.create_dataset(
                        "ax", data=np.array([], dtype=np.float64))
                    grp.create_dataset(
                        "ay", data=np.array([], dtype=np.float64))
                    grp.create_dataset(
                        "az", data=np.array([], dtype=np.float64))
                    if has_props:
                        if props_ncams > 1:
                            z = np.zeros((0, props_ncams), dtype=np.float64)
                            grp.create_dataset("diameter", data=z)
                            grp.create_dataset("intensity", data=z.copy())
                            grp.create_dataset("mass", data=z.copy())
                            for c in range(props_ncams):
                                grp.create_dataset(
                                    f"intensity_{c}", data=np.array([], dtype=np.float64)
                                )
                                grp.create_dataset(
                                    f"mass_{c}", data=np.array([], dtype=np.float64)
                                )
                        else:
                            grp.create_dataset(
                                "diameter", data=np.array([], dtype=np.float64))
                            grp.create_dataset(
                                "intensity", data=np.array([], dtype=np.float64))
                            grp.create_dataset(
                                "mass", data=np.array([], dtype=np.float64))
                            grp.create_dataset(
                                "intensity_0", data=np.array([], dtype=np.float64))
                            grp.create_dataset(
                                "mass_0", data=np.array([], dtype=np.float64))
                    continue
                ids = np.array([p[0] for p in particles])
                x = np.array([p[1] for p in particles])
                y = np.array([p[2] for p in particles])
                z = np.array([p[3] for p in particles])
                vx = np.array([p[4] for p in particles])
                vy = np.array([p[5] for p in particles])
                vz = np.array([p[6] for p in particles])
                ax = np.array([p[7] for p in particles])
                ay = np.array([p[8] for p in particles])
                az = np.array([p[9] for p in particles])
                grp = f.create_group(key)
                grp.create_dataset("id", data=ids)
                grp.create_dataset("x", data=x)
                grp.create_dataset("y", data=y)
                grp.create_dataset("z", data=z)
                grp.create_dataset("vx", data=vx)
                grp.create_dataset("vy", data=vy)
                grp.create_dataset("vz", data=vz)
                grp.create_dataset("ax", data=ax)
                grp.create_dataset("ay", data=ay)
                grp.create_dataset("az", data=az)
                if has_props:
                    ds = np.stack([np.atleast_1d(p[10]) for p in particles])
                    ins = np.stack([np.atleast_1d(p[11]) for p in particles])
                    ms = np.stack([np.atleast_1d(p[12]) for p in particles])
                    grp.create_dataset("diameter", data=ds)
                    grp.create_dataset("intensity", data=ins)
                    grp.create_dataset("mass", data=ms)
                    for c in range(ins.shape[1]):
                        grp.create_dataset(f"intensity_{c}", data=ins[:, c])
                    for c in range(ms.shape[1]):
                        grp.create_dataset(f"mass_{c}", data=ms[:, c])

        max_block = max(
            bidx for bidx, _ in all_curve_data) if all_curve_data else -1
        num_blocks = max_block + 1
        _write_bspline_curves_h5_xmf(
            out_curves_h5, out_curves_xmf, all_curve_data, num_blocks)

    try:
        write_outputs()
    except PermissionError:
        fallback_dir = os.getcwd()
        out_bspline_h5 = os.path.join(fallback_dir, stem + "_bspline.h5part")
        out_curves_h5 = os.path.join(fallback_dir, stem + "_bspline_curves.h5")
        out_curves_xmf = os.path.join(
            fallback_dir, stem + "_bspline_curves.xmf")
        print(
            "Permission denied writing to original directory; writing to current directory:",
            fallback_dir,
            file=sys.stderr,
        )
        write_outputs()

    return out_bspline_h5, out_curves_h5, out_curves_xmf


def main():
    parser = argparse.ArgumentParser(
        description="B-spline fit 4-frame tracks, write _bspline.h5part and _bspline_curves.h5/.xmf (ParaView)"
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Input tracks file (e.g. .../tracks_box_size_0.25.h5part)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step between frames (default: 1.0)",
    )
    parser.add_argument(
        "--num-curve-samples",
        type=int,
        default=50,
        help="Number of points per B-spline curve (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input file)",
    )
    args = parser.parse_args()
    out_h5, out_curves_h5, out_curves_xmf = run(
        args.filepath,
        dt=args.dt,
        num_curve_samples=args.num_curve_samples,
        output_dir=args.output_dir,
    )
    print("Wrote:", out_h5)
    print("Wrote:", out_curves_h5)
    print("Wrote:", out_curves_xmf, "(open this in ParaView)")


if __name__ == "__main__":
    main()
