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
    return p.parse_args()


def discover_runs(output_base):
    if not os.path.isdir(output_base):
        return []
    runs = []
    for name in sorted(os.listdir(output_base)):
        path = os.path.join(output_base, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, 'Centers')):
            runs.append(name)
    return runs


def run_rays_for_run(process_data_path, flush_every=1, n_workers=1):
    rays = Rays(process_data_path)
    rays.compute_rays(n_workers=n_workers, flush_every=flush_every)
    rays.write_rays()


def main():
    args = parse_args()
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
        f"  output_base={out_base}  flush_every={args.flush_every}  n_workers={args.n_workers}")

    for run in runs:
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(out_path):
            print(f"Skip {run}: missing {out_path}")
            continue
        print(f"Processing run: {run}")
        run_rays_for_run(out_path, flush_every=args.flush_every,
                         n_workers=args.n_workers)
        print(f"Done {run}")
    print("All runs completed.")


if __name__ == '__main__':
    main()
