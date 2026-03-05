"""
Run stereo matching only (rays + stereo matching) for all runs in a case.
Expects center finding to have been run first (data/PTV_center/{case}/{run}/ with Centers/).
"""
import os
import sys
import argparse
from python_4dptv.matching.rays import Rays
from python_4dptv.matching.stereomatching import StereoMatching

os.chdir('/workspaces/4d-ptv-mcflow')

_DEFAULT_CASE = 'TTI_opposing_gravity'


def parse_args():
    p = argparse.ArgumentParser(
        description='Run rays + stereo matching for all runs in a case (center finding must be done first).'
    )
    p.add_argument('--case', type=str, default=_DEFAULT_CASE,
                   help='Case name (e.g. TTI_aligned_with_gravity).')
    p.add_argument('--runs', type=str, default=None,
                   help='Comma-separated run names. If omitted, all runs with Centers/ under output-base are processed.')
    p.add_argument('--output-base', type=str, default=None,
                   help='Override output base path. Default: data/PTV_center/{case}.')
    # Stereo matching parameters
    p.add_argument('--min-cameras', type=int, default=3,
                   help='Minimum number of cameras for a match (default: 3).')
    p.add_argument('--max-distance', type=float, default=0.2,
                   help='Maximum distance for stereo matching (default: 0.2).')
    p.add_argument('--multiple-matches-per-ray-distance', type=float, default=2.0,
                   help='Multiple matches per ray distance (default: 2.0).')
    p.add_argument('--max-matches-per-ray', type=int, default=1,
                   help='Maximum matches per ray (default: 1).')
    p.add_argument('--nvoxels', type=str, default='600,600,500',
                   help='Number of voxels [nx,ny,nz] comma-separated (default: 600,600,500).')
    p.add_argument('--bounding-box', type=str, default='-50,50,-35,35,-20,20',
                   help='Bounding box [minX,maxX,minY,maxY,minZ,maxZ] comma-separated (default: -50,50,-35,35,-20,20).')
    return p.parse_args()


def parse_int_list(s, length, name):
    parts = [x.strip() for x in s.split(',') if x.strip()]
    if len(parts) != length:
        raise ValueError(f'{name} must have {length} comma-separated values, got {s}')
    return [int(x) for x in parts]


def parse_float_list(s, length, name):
    parts = [x.strip() for x in s.split(',') if x.strip()]
    if len(parts) != length:
        raise ValueError(f'{name} must have {length} comma-separated values, got {s}')
    return [float(x) for x in parts]


def discover_runs(output_base):
    if not os.path.isdir(output_base):
        return []
    runs = []
    for name in sorted(os.listdir(output_base)):
        path = os.path.join(output_base, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, 'Centers')):
            runs.append(name)
    return runs


def run_stereo_matching_for_run(process_data_path, args):
    nvoxels = parse_int_list(args.nvoxels, 3, 'nvoxels')
    boundingbox = parse_float_list(args.boundingbox, 6, 'bounding-box')
    print("  Ray computation")
    rays = Rays(process_data_path)
    rays.compute_rays()
    rays.write_rays()
    del rays
    print("  Stereo matching")
    sm = StereoMatching(
        process_data_path,
        args.min_cameras, args.max_distance,
        args.multiple_matches_per_ray_distance, args.max_matches_per_ray,
        nvoxels, boundingbox,
    )
    sm.run_stereomatching()


def main():
    args = parse_args()
    out_base = args.output_base or os.path.join('data', 'PTV_center', args.case)

    if args.runs:
        runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    else:
        runs = discover_runs(out_base)

    if not runs:
        print(f"No runs with center-finding output found under {os.path.abspath(out_base)}", file=sys.stderr)
        sys.exit(1)

    print(f"Stereo matching (all runs): case={args.case}, runs={runs}")
    print(f"  min_cameras={args.min_cameras}, max_distance={args.max_distance}, nvoxels={args.nvoxels}, bounding_box={args.bounding_box}")
    print(f"  output_base={out_base}")

    for run in runs:
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(out_path):
            print(f"Skip {run}: missing {out_path}")
            continue
        print(f"Processing run: {run}")
        run_stereo_matching_for_run(out_path, args)
        print(f"Done {run}")
    print("All runs completed.")


if __name__ == '__main__':
    main()
