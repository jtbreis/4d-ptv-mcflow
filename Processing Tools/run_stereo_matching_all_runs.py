"""
Run stereo matching (STM) only for all runs in a case.
Expects precomputed rays on disk (rays.h5 per run), e.g. from compute_rays / 03_compute_rays.
"""
import os
import sys
import argparse
from python_4dptv.matching.stereomatching import StereoMatching
from python_4dptv.utils.structure import Filenames
from python_4dptv.io.h5_metadata import utc_timestamp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pipeline_paths import default_data_base, ensure_output_run_dir, resolve_case_paths

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
                   help='Parent of the case folder for writing rays_out_cpp.h5.')
    p.add_argument('--input-base', type=str, default=None,
                   help='Parent of the case folder for reading rays.h5. Default: same as --output-base.')
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
    p.add_argument('--timing', action='store_true',
                   help='Pass --timing to STM (per-stage ms/frame at end of run).')
    p.add_argument('--spatial-boxes', type=int, default=0, dest='spatial_boxes',
                   help='STM --spatial-boxes: sub-volume count (0 = single full volume, default).')
    p.add_argument('--spatial-overlap', type=int, default=1, dest='spatial_overlap_cells',
                   help='STM --spatial-overlap: halo in global voxels per face (default: 1).')
    p.add_argument('--frame-parallelism', type=int, default=0, dest='frame_parallelism',
                   help='STM --frame-parallelism: max frames processed concurrently (0 = use OMP_NUM_THREADS).')
    p.add_argument('--max-frames', type=int, default=None, dest='max_frames',
                   help='Only process the first N frames (default: all frames in rays.h5).')
    p.add_argument('--resume', action='store_true',
                   help='Keep existing rays_out_cpp.h5 and skip frames that already have matches.')
    p.add_argument('--write-particle-props', action=argparse.BooleanOptionalAction, default=True,
                   help='Write diameter/intensity/mass from rays.h5 into rays_out_cpp.h5 (default: on). '
                        'Use --no-write-particle-props to skip.')
    p.add_argument('--write-diameter', action=argparse.BooleanOptionalAction, default=True,
                   help='Write diameter dataset when particle props are enabled (default: on).')
    p.add_argument('--write-intensity', action=argparse.BooleanOptionalAction, default=True,
                   help='Write intensity dataset when particle props are enabled (default: on).')
    p.add_argument('--write-mass', action=argparse.BooleanOptionalAction, default=True,
                   help='Write mass dataset when particle props are enabled (default: on).')
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


def _args_to_metadata(args, *, case, run, input_path, output_path) -> dict:
    return {
        "timestamp_utc": utc_timestamp(),
        "job_id": "stereo_matching",
        "case": case,
        "run": run,
        "input_path": os.path.abspath(input_path),
        "output_path": os.path.abspath(output_path),
        "input_base": getattr(args, "input_base", None),
        "output_base": getattr(args, "output_base", None),
        "parameters": {
            "threads": args.threads,
            "min_cameras": args.min_cameras,
            "max_distance": args.max_distance,
            "multiple_matches_per_ray_distance": args.multiple_matches_per_ray_distance,
            "max_matches_per_ray": args.max_matches_per_ray,
            "nvoxels": args.nvoxels,
            "bounding_box": args.bounding_box,
            "timing": args.timing,
            "spatial_boxes": args.spatial_boxes,
            "spatial_overlap_cells": args.spatial_overlap_cells,
            "frame_parallelism": args.frame_parallelism,
            "max_frames": args.max_frames,
            "resume": args.resume,
            "write_particle_props": args.write_particle_props,
            "write_diameter": args.write_diameter,
            "write_intensity": args.write_intensity,
            "write_mass": args.write_mass,
        },
    }


def run_stereo_matching_for_run(input_path, output_path, args, *, case, run):
    rays_file = rays_path_for_run(input_path)
    if not os.path.isfile(rays_file):
        print(
            f"  Skip: missing rays file (compute rays first): {os.path.abspath(rays_file)}",
            file=sys.stderr,
        )
        return False
    ensure_output_run_dir(output_path)
    nvoxels = parse_int_list(args.nvoxels, 3, 'nvoxels')
    boundingbox = parse_float_list(args.bounding_box, 6, 'bounding-box')
    print("  Stereo matching (using existing rays.h5)")
    sm = StereoMatching(
        output_path,
        args.min_cameras, args.max_distance,
        args.multiple_matches_per_ray_distance, args.max_matches_per_ray,
        nvoxels, boundingbox,
        rays_input_path=input_path,
        run_metadata=_args_to_metadata(
            args, case=case, run=run, input_path=input_path, output_path=output_path,
        ),
    )
    sm.run_stereomatching(
        nthreads=args.threads,
        nframes=args.max_frames,
        timing=args.timing,
        spatial_boxes=args.spatial_boxes,
        spatial_overlap_cells=args.spatial_overlap_cells,
        frame_parallelism=args.frame_parallelism,
        write_particle_props=args.write_particle_props,
        write_diameter=args.write_diameter,
        write_intensity=args.write_intensity,
        write_mass=args.write_mass,
        resume=args.resume,
    )
    return True


def main():
    args = parse_args()
    if args.threads < 1:
        print('Error: --threads must be >= 1', file=sys.stderr)
        sys.exit(1)
    if args.max_frames is not None and args.max_frames < 1:
        print('Error: --max-frames must be >= 1', file=sys.stderr)
        sys.exit(1)
    in_base, out_base = resolve_case_paths(
        args.case,
        input_base=args.input_base,
        output_base=args.output_base,
        default_output_base=default_data_base(),
    )
    args.input_base = in_base
    args.output_base = out_base

    if args.runs:
        runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    else:
        runs = discover_runs(in_base)

    if not runs:
        print(
            f"No runs with rays.h5 found under {os.path.abspath(in_base)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Stereo matching (all runs): case={args.case}, runs={runs}")
    print(
        f"  threads={args.threads}, min_cameras={args.min_cameras}, max_distance={args.max_distance}, "
        f"nvoxels={args.nvoxels}, bounding_box={args.bounding_box}",
    )
    print(
        f"  spatial_boxes={args.spatial_boxes}, spatial_overlap_cells={args.spatial_overlap_cells}, "
        f"frame_parallelism={args.frame_parallelism}, max_frames={args.max_frames}, resume={args.resume}, timing={args.timing}",
    )
    print(
        f"  write_particle_props={args.write_particle_props}, write_diameter={args.write_diameter}, "
        f"write_intensity={args.write_intensity}, write_mass={args.write_mass}",
    )
    print(f"  input_base={in_base}  output_base={out_base}")

    for run in runs:
        in_path = os.path.join(in_base, run)
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(in_path):
            print(f"Skip {run}: missing input {in_path}")
            continue
        print(f"Processing run: {run}")
        if run_stereo_matching_for_run(in_path, out_path, args, case=args.case, run=run):
            print(f"Done {run}")
    print("All runs completed.")


if __name__ == '__main__':
    main()
