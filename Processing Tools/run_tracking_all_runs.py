"""
Run 4-frame tracking only for all runs in a case.
Expects stereo matching to have been run first (rays_out_cpp.h5 in each run folder).
Requires 4BE-ETI.
"""
import os
import sys
import argparse

os.chdir('/workspaces/4d-ptv-mcflow')

_DEFAULT_CASE = 'TTI_opposing_gravity'

try:
    from python_4be_eti.perform_tracking import FourFrameTracking
    from python_4be_eti.utils.basic_utils import create_h5_file
    HAS_TRACKING = True
except ImportError:
    HAS_TRACKING = False


def parse_args():
    p = argparse.ArgumentParser(
        description='Run 4-frame tracking for all runs in a case (stereo matching must be done first).'
    )
    p.add_argument('--case', type=str, default=_DEFAULT_CASE,
                   help='Case name (e.g. TTI_aligned_with_gravity).')
    p.add_argument('--runs', type=str, default=None,
                   help='Comma-separated run names. If omitted, all runs with rays_out_cpp.h5 under output-base are processed.')
    p.add_argument('--output-base', type=str, default=None,
                   help=('Override output root (without case), e.g. data/low_threshold/PTV_center. '
                         'Runs are read from {output_base}/{case}/Run*/. '
                         'Default root: data/PTV_center'))
    # Tracking parameters (4BE-ETI)
    p.add_argument('--rays-filename', type=str, default='rays_out_cpp.h5',
                   help='Rays output filename (default: rays_out_cpp.h5).')
    p.add_argument('--box-size-x', type=float, default=3.5,
                   help='Box size x for track initialization (default: 3.5).')
    p.add_argument('--box-size-initial-x-lo', type=float, default=None,
                   help=('Signed x offset from seed on frame+1 (added to x0); '
                         'physical min x = x0+min(lo,hi), max x = x0+max(lo,hi). '
                         'Default if omitted: -box-size-x. Use negative lo to include points on the -x side of the seed.'))
    p.add_argument('--box-size-initial-x-hi', type=float, default=None,
                   help='Signed x offset from seed on frame+1; default if omitted: +box-size-x.')
    p.add_argument('--box-size-y', type=float, default=1.0,
                   help='Box size y for track initialization (default: 1.0).')
    p.add_argument('--box-size-z', type=float, default=1.0,
                   help='Box size z for track initialization (default: 1.0).')
    p.add_argument('--box-size-track', type=float, default=1.0,
                   help='Box size after track initialized (default: 1.0).')
    p.add_argument('--dt', type=float, default=1e-3,
                   help='Time step (default: 1e-3).')
    p.add_argument('--rep-rate', type=float, default=10.0,
                   help='Repetition rate (default: 10.0).')
    p.add_argument('--workers', type=int, default=8,
                   help='Number of workers for tracking (default: 8).')
    p.add_argument('--max-frame-ranges', type=int, default=None,
                   help='Only process the first N frame ranges per run (default: all).')
    p.add_argument('--particle-progress-interval', type=int, default=1000,
                   help='Print per-frame ``particle i/n`` every N particles; 0 = off (default: 1000).')
    p.add_argument('--max-candidates-mesh', type=int, default=None,
                   help='3D tracking: max candidates per meshgrid stage (default: 1000).')
    p.add_argument('--max-targets-mesh', type=int, default=None,
                   help='3D tracking: max target particles per meshgrid stage (default: 2000).')
    p.add_argument('--debug-mesh-match-prints', type=int, default=0,
                   help='3D: print mesh-stage array sizes for the first N no_previous_tracks_3d calls with candidates (0=off). Uses NumPy for init when >0.')
    p.add_argument('--position-unit', type=str, default='mm',
                   help='Position unit for x/y/z (e.g. mm, m). Used to scale v/a to SI (default: mm).')
    p.add_argument('--write-paraview', action='store_true', default=True,
                   help='Write Paraview output (default: True).')
    p.add_argument('--no-write-paraview', action='store_false', dest='write_paraview',
                   help='Disable Paraview output.')
    p.add_argument('--write-failed-tracks', action='store_true',
                   help='Also write particles for which tracking failed (id=0, tracked=False).')
    return p.parse_args()


def discover_runs(output_base, rays_filename):
    if not os.path.isdir(output_base):
        return []
    runs = []
    for name in sorted(os.listdir(output_base)):
        path = os.path.join(output_base, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, rays_filename)):
            runs.append(name)
    return runs


def run_tracking_for_run(process_data_path, args):
    create_h5_file(folder=process_data_path)
    tracking = FourFrameTracking(
        process_data_path,
        filename=args.rays_filename,
        box_size_x=args.box_size_x, box_size_y=args.box_size_y, box_size_z=args.box_size_z,
        box_size_initial_x_lo=args.box_size_initial_x_lo,
        box_size_initial_x_hi=args.box_size_initial_x_hi,
        box_size_track=args.box_size_track, dt=args.dt, rep_rate=args.rep_rate,
        use_bspline=True,
        position_unit=args.position_unit,
        write_paraview=args.write_paraview,
        write_failed_tracks=args.write_failed_tracks,
        particle_progress_interval=args.particle_progress_interval,
        max_candidates_mesh=args.max_candidates_mesh,
        max_targets_mesh=args.max_targets_mesh,
        debug_mesh_match_prints=args.debug_mesh_match_prints,
    )
    tracking.run_tracking(
        workers=args.workers,
        max_frame_ranges=args.max_frame_ranges,
    )


def main():
    args = parse_args()
    if not HAS_TRACKING:
        print("4BE-ETI is required for tracking. Install the 4BE-ETI package.", file=sys.stderr)
        sys.exit(1)

    if args.output_base:
        out_base = os.path.join(args.output_base, args.case)
    else:
        out_base = os.path.join('data', 'PTV_center', args.case)

    if args.runs:
        runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    else:
        runs = discover_runs(out_base, args.rays_filename)

    if not runs:
        print(f"No runs with {args.rays_filename} found under {os.path.abspath(out_base)}", file=sys.stderr)
        sys.exit(1)

    print(f"Tracking (all runs): case={args.case}, runs={runs}")
    print(
        f"  box_size_x={args.box_size_x}, "
        f"box_size_initial_x_lo={args.box_size_initial_x_lo}, "
        f"box_size_initial_x_hi={args.box_size_initial_x_hi}, "
        f"box_size_y={args.box_size_y}, box_size_z={args.box_size_z}, box_size_track={args.box_size_track}"
    )
    print(f"  dt={args.dt}, rep_rate={args.rep_rate}, workers={args.workers}, max_frame_ranges={args.max_frame_ranges}")
    print(f"  position_unit={args.position_unit}")
    print(f"  output_base={out_base}")

    for run in runs:
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(out_path):
            print(f"Skip {run}: missing {out_path}")
            continue
        print(f"Processing run: {run}")
        run_tracking_for_run(out_path, args)
        print(f"Done {run}")
    print("All runs completed.")


if __name__ == '__main__':
    main()
