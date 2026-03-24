"""
Run stereo matching (STM) only for all runs in a case.
Expects precomputed rays on disk (rays.h5 per run), e.g. from compute_rays / 03_compute_rays.
"""
import os
import sys
import argparse
from python_4dptv.matching.stereomatching import StereoMatching
from python_4dptv.utils.structure import Filenames

os.chdir('/workspaces/4d-ptv-mcflow')

_DEFAULT_CASE = 'TTI_opposing_gravity'


def parse_args():
    p = argparse.ArgumentParser(
        description='Run stereo matching for all runs in a case (rays.h5 must already exist).'
    )
    p.add_argument('--case', type=str, default=_DEFAULT_CASE,
                   help='Case name (e.g. TTI_aligned_with_gravity).')
    p.add_argument('--runs', type=str, default=None,
                   help='Comma-separated run names. If omitted, all runs with rays.h5 under output-base are processed.')
    p.add_argument('--output-base', type=str, default=None,
                   help='Parent of the case folder (same as center finding / compute_rays). '
                        'Runs are read from {output_base}/{case}/Run*/. '
                        'Default: data/PTV_center (i.e. data/PTV_center/{case}/...).')
    p.add_argument('--min-cameras', type=int, default=3,
                   help='Minimum number of cameras for a match (default: 3).')
    p.add_argument('--max-distance', type=float, default=0.15,
                   help='Maximum distance for stereo matching (default: 0.15).')
    p.add_argument('--multiple-matches-per-ray-distance', type=float, default=0.5,
                   help='Multiple matches per ray distance (default: 0.5).')
    p.add_argument('--max-matches-per-ray', type=int, default=4,
                   help='Maximum matches per ray (default: 4).')
    p.add_argument('--nvoxels', type=str, default='550,350,200',
                   help='Number of voxels [nx,ny,nz] comma-separated (default: 550,350,200).')
    p.add_argument('--bounding-box', type=str, default='-55,55,-35,35,-20,20',
                   help='Bounding box [minX,maxX,minY,maxY,minZ,maxZ] comma-separated.')
    p.add_argument('--threads', type=int, default=12,
                   help='Thread count for the STM binary (OMP_NUM_THREADS; default: 12).')
    return p.parse_args()


def parse_int_list(s, length, name):
    parts = [x.strip() for x in s.split(',') if x.strip()]
    if len(parts) != length:
        raise ValueError(
            f'{name} must have {length} comma-separated values, got {s}')
    return [int(x) for x in parts]


def parse_float_list(s, length, name):
    parts = [x.strip() for x in s.split(',') if x.strip()]
    if len(parts) != length:
        raise ValueError(
            f'{name} must have {length} comma-separated values, got {s}')
    return [float(x) for x in parts]


def rays_path_for_run(run_dir):
    return run_dir + Filenames.RAYS.value


def discover_runs(output_base):
    if not os.path.isdir(output_base):
        return []
    runs = []
    for name in sorted(os.listdir(output_base)):
        path = os.path.join(output_base, name)
        if os.path.isdir(path) and os.path.isfile(rays_path_for_run(path)):
            runs.append(name)
    return runs


def run_stereo_matching_for_run(process_data_path, args):
    rays_file = rays_path_for_run(process_data_path)
    if not os.path.isfile(rays_file):
        print(
            f"  Skip: missing rays file (compute rays first): {os.path.abspath(rays_file)}",
            file=sys.stderr,
        )
        return False
    nvoxels = parse_int_list(args.nvoxels, 3, 'nvoxels')
    boundingbox = parse_float_list(args.bounding_box, 6, 'bounding-box')
    print("  Stereo matching (using existing rays.h5)")
    sm = StereoMatching(
        process_data_path,
        args.min_cameras, args.max_distance,
        args.multiple_matches_per_ray_distance, args.max_matches_per_ray,
        nvoxels, boundingbox,
    )
    sm.run_stereomatching(nthreads=args.threads)
    return True


def main():
    args = parse_args()
    if args.threads < 1:
        print('Error: --threads must be >= 1', file=sys.stderr)
        sys.exit(1)
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
            f"No runs with rays.h5 found under {os.path.abspath(out_base)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Stereo matching (all runs): case={args.case}, runs={runs}")
    print(
        f"  threads={args.threads}, min_cameras={args.min_cameras}, max_distance={args.max_distance}, "
        f"nvoxels={args.nvoxels}, bounding_box={args.bounding_box}",
    )
    print(f"  output_base={out_base}")

    for run in runs:
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(out_path):
            print(f"Skip {run}: missing {out_path}")
            continue
        print(f"Processing run: {run}")
        if run_stereo_matching_for_run(out_path, args):
            print(f"Done {run}")
    print("All runs completed.")


if __name__ == '__main__':
    main()
