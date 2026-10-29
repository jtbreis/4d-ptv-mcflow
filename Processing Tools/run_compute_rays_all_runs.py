"""
Compute rays (and write ray HDF5) for all runs in a case.
Expects center finding output under {output-base}/{case}/{run}/ with Centers_TH1/,
Centers_TH2/, and/or legacy Centers/. Calibration at {output-base}/Calibration/calib.h5
or per-run calib as configured by Rays.
Equivalent to Processing Steps/03_compute_rays.py, batched over runs.
"""
import os
import sys
import argparse
from datetime import datetime, timezone
from python_4dptv.matching.rays import Rays
from python_4dptv.io.h5_metadata import utc_timestamp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pipeline_paths import default_processed_base, ensure_output_run_dir, resolve_case_paths

os.chdir('/workspaces/4d-ptv-mcflow')

_DEFAULT_CASE = 'TNTI_opposing_gravity'


def parse_args():
    p = argparse.ArgumentParser(
        description='Compute rays for all runs in a case (center finding must be done first).'
    )
    p.add_argument('--case', type=str, default=_DEFAULT_CASE,
                   help='Case name (e.g. TNTI_aligned_with_gravity).')
    p.add_argument('--runs', type=str, default=None,
                   help='Comma-separated run names. If omitted, all runs with Centers_* under output-base are processed.')
    p.add_argument('--output-base', type=str, default=None,
                   help='Parent of the case folder for writing rays.h5 ({output-base}/{case}/Run*/).')
    p.add_argument('--input-base', type=str, default=None,
                   help='Parent of the case folder for reading Centers_* ({input-base}/{case}/Run*/). '
                        'Default: same as --output-base.')
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
    p.add_argument(
        '--centers-threshold',
        type=str,
        default=None,
        help='Per-camera Centers folder selector: comma-separated TH1|TH2 in calib order '
             '(reads Centers_TH1/ or Centers_TH2/ per camera). Env: RAYS_CENTERS_THRESHOLD.',
    )
    p.add_argument(
        '--write-protect',
        action='store_true',
        help='Remove write permission from rays.h5 after writing. Env: RAYS_WRITE_PROTECT=1.',
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
    if args.centers_threshold is None:
        v = os.environ.get('RAYS_CENTERS_THRESHOLD')
        if v is not None and str(v).strip():
            args.centers_threshold = v.strip()
    if not args.write_protect:
        v = os.environ.get('RAYS_WRITE_PROTECT', '').strip().lower()
        args.write_protect = v in ('1', 'true', 'yes', 'on')
    return args


def _finalize_image_dims(args):
    args.image_width = _parse_image_dim(args.image_width)
    args.image_height = _parse_image_dim(args.image_height)
    return args


def _run_has_centers(run_path: str) -> bool:
    if os.path.isdir(os.path.join(run_path, 'Centers')):
        return True
    try:
        entries = os.listdir(run_path)
    except OSError:
        return False
    return any(
        name.startswith('Centers_TH') and os.path.isdir(os.path.join(run_path, name))
        for name in entries
    )


def discover_runs(output_base):
    if not os.path.isdir(output_base):
        return []
    runs = []
    for name in sorted(os.listdir(output_base)):
        path = os.path.join(output_base, name)
        if os.path.isdir(path) and _run_has_centers(path):
            runs.append(name)
    return runs


def _args_to_metadata(args, *, case, run, input_path, output_path) -> dict:
    return {
        "timestamp_utc": utc_timestamp(),
        "job_id": "compute_rays",
        "case": case,
        "run": run,
        "input_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "input_base": getattr(args, "input_base", None),
        "output_base": getattr(args, "output_base", None),
        "parameters": {
            "flush_every": args.flush_every,
            "n_workers": args.n_workers,
            "center_rotate": args.center_rotate,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "centers_threshold": args.centers_threshold,
            "write_protect": args.write_protect,
        },
    }


def run_rays_for_run(
    input_path,
    output_path,
    flush_every=1,
    n_workers=1,
    center_rotate=None,
    image_width=None,
    image_height=None,
    centers_threshold=None,
    write_protect=False,
    run_metadata=None,
):
    ensure_output_run_dir(output_path)
    rays = Rays(
        input_path,
        center_rotate=center_rotate,
        image_width=image_width,
        image_height=image_height,
        centers_threshold=centers_threshold,
        write_protect=write_protect,
        output_path=output_path,
        run_metadata=run_metadata,
    )
    rays.compute_rays(n_workers=n_workers, flush_every=flush_every)
    rays.write_rays()


def main():
    args = _finalize_image_dims(_apply_compute_rays_env_overrides(parse_args()))
    in_base, out_base = resolve_case_paths(
        args.case,
        input_base=args.input_base,
        output_base=args.output_base,
        default_output_base=default_processed_base(),
    )
    args.input_base = in_base
    args.output_base = out_base

    if args.runs:
        runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    else:
        runs = discover_runs(in_base)

    if not runs:
        print(
            f"No runs with center-finding output found under {os.path.abspath(in_base)}", file=sys.stderr)
        sys.exit(1)
    if args.flush_every < 1 or args.n_workers < 1:
        print("flush_every and n_workers must be >= 1", file=sys.stderr)
        sys.exit(1)

    print(f"Compute rays (all runs): case={args.case}, runs={runs}")
    extras = []
    if args.center_rotate:
        extras.append(
            f"center_rotate={args.center_rotate!r}  "
            f"image_wh=({args.image_width!r}, {args.image_height!r})"
        )
    if args.centers_threshold:
        extras.append(f"centers_threshold={args.centers_threshold!r}")
    if args.write_protect:
        extras.append("write_protect=True")
    print(
        f"  input_base={in_base}  output_base={out_base}  "
        f"flush_every={args.flush_every}  n_workers={args.n_workers}"
        + (f"  {'  '.join(extras)}" if extras else "")
    )

    for run in runs:
        in_path = os.path.join(in_base, run)
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(in_path):
            print(f"Skip {run}: missing input {in_path}")
            continue
        print(f"Processing run: {run}")
        run_rays_for_run(
            in_path,
            out_path,
            flush_every=args.flush_every,
            n_workers=args.n_workers,
            center_rotate=args.center_rotate,
            image_width=args.image_width,
            image_height=args.image_height,
            centers_threshold=args.centers_threshold,
            write_protect=args.write_protect,
            run_metadata=_args_to_metadata(
                args, case=args.case, run=run, input_path=in_path, output_path=out_path,
            ),
        )
        print(f"Done {run}")
    print("All runs completed.")


if __name__ == '__main__':
    main()
