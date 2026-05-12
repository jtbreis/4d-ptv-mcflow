"""
Compute rays (and write ray HDF5) for all runs in a case.
Expects center finding output: data/PTV_center/{case}/{run}/ with Centers/ and calib.h5.
Equivalent to Processing Steps/03_compute_rays.py, batched over runs.
"""
import os
import sys
import argparse
from python_4dptv.matching.rays import Rays

os.chdir('/workspaces/4d-ptv-mcflow')

_DEFAULT_CASE = 'TTI_opposing_gravity'


def parse_args():
    p = argparse.ArgumentParser(
        description='Compute rays for all runs in a case (center finding must be done first).'
    )
    p.add_argument('--case', type=str, default=_DEFAULT_CASE,
                   help='Case name (e.g. TTI_aligned_with_gravity).')
    p.add_argument('--runs', type=str, default=None,
                   help='Comma-separated run names. If omitted, all runs with Centers/ under output-base are processed.')
    p.add_argument('--output-base', type=str, default=None,
                   help='Parent of the case folder (same as center finding OUTPUT_BASE). '
                        'Runs are read from {output-base}/{case}/Run*/. '
                        'Default: data/PTV_center.')
    p.add_argument('--flush-every', type=int, default=1,
                   help='After this many frames written per camera, flush the HDF5 file '
                        '(default 1 = every frame; larger = fewer flushes, less I/O overhead).')
    p.add_argument('--n-workers', type=int, default=1,
                   help='Parallel processes (one per camera, capped at n cameras). '
                        'Requires stream_to_disk (default).')
    p.add_argument(
        '--center-rotate',
        type=str,
        default=None,
        help="Per-camera center (x,y) remap (90° rotation): comma-separated cw/ccw/none in calib order, "
             "or 'pairwise' (4 cams: 0–1 cw, 2–3 ccw). Env: RAYS_CENTER_ROTATE.",
    )
    p.add_argument(
        '--image-width',
        type=str,
        default=None,
        help='Image width(s) in pixels before rotation: one integer, or comma-separated '
             'per camera (same order as calib). Required with --center-rotate. Env: RAYS_IMAGE_WIDTH.',
    )
    p.add_argument(
        '--image-height',
        type=str,
        default=None,
        help='Image height(s): same rules as --image-width. Env: RAYS_IMAGE_HEIGHT.',
    )
    return p.parse_args()


def _parse_image_dim(value):
    """None, int, or list of int for per-camera image size."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [int(x) for x in value]
    s = str(value).strip()
    if not s:
        return None
    if ',' in s:
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return [int(p) for p in parts]
    return int(s)


def _apply_compute_rays_env_overrides(args):
    """Fill args from environment when CLI omitted (Docker job lines)."""
    if args.center_rotate is None:
        v = os.environ.get('RAYS_CENTER_ROTATE')
        if v is not None and str(v).strip():
            args.center_rotate = v.strip()
    if args.image_width is None:
        v = os.environ.get('RAYS_IMAGE_WIDTH')
        if v is not None and str(v).strip():
            args.image_width = v.strip()
    if args.image_height is None:
        v = os.environ.get('RAYS_IMAGE_HEIGHT')
        if v is not None and str(v).strip():
            args.image_height = v.strip()
    return args


def _finalize_image_dims(args):
    args.image_width = _parse_image_dim(args.image_width)
    args.image_height = _parse_image_dim(args.image_height)
    return args


def discover_runs(output_base):
    if not os.path.isdir(output_base):
        return []
    runs = []
    for name in sorted(os.listdir(output_base)):
        path = os.path.join(output_base, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, 'Centers')):
            runs.append(name)
    return runs


def run_rays_for_run(
    process_data_path,
    flush_every=1,
    n_workers=1,
    center_rotate=None,
    image_width=None,
    image_height=None,
):
    rays = Rays(
        process_data_path,
        center_rotate=center_rotate,
        image_width=image_width,
        image_height=image_height,
    )
    rays.compute_rays(n_workers=n_workers, flush_every=flush_every)
    rays.write_rays()


def main():
    args = _finalize_image_dims(_apply_compute_rays_env_overrides(parse_args()))
    if args.output_base:
        out_base = os.path.join(args.output_base, args.case)
    else:
        out_base = os.path.join('data', 'PTV_center', args.case)

    if args.runs:
        runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    else:
        runs = discover_runs(out_base)

    if not runs:
        print(
            f"No runs with center-finding output found under {os.path.abspath(out_base)}", file=sys.stderr)
        sys.exit(1)
    if args.flush_every < 1 or args.n_workers < 1:
        print("flush_every and n_workers must be >= 1", file=sys.stderr)
        sys.exit(1)

    print(f"Compute rays (all runs): case={args.case}, runs={runs}")
    print(
        f"  output_base={out_base}  flush_every={args.flush_every}  n_workers={args.n_workers}"
        + (
            f"  center_rotate={args.center_rotate!r}  image_wh=({args.image_width!r}, {args.image_height!r})"
            if args.center_rotate
            else ""
        ),
    )

    for run in runs:
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(out_path):
            print(f"Skip {run}: missing {out_path}")
            continue
        print(f"Processing run: {run}")
        run_rays_for_run(
            out_path,
            flush_every=args.flush_every,
            n_workers=args.n_workers,
            center_rotate=args.center_rotate,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        print(f"Done {run}")
    print("All runs completed.")


if __name__ == '__main__':
    main()
