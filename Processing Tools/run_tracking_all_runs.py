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
                   help='Override output base path. Default: data/PTV_center/{case}.')
    # Tracking parameters (4BE-ETI)
    p.add_argument('--rays-filename', type=str, default='rays_out_cpp.h5',
                   help='Rays output filename (default: rays_out_cpp.h5).')
    p.add_argument('--box-size-x', type=float, default=3.5,
                   help='Box size x for track initialization (default: 3.5).')
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
        box_size_track=args.box_size_track, dt=args.dt, rep_rate=args.rep_rate,
        write_paraview=args.write_paraview,
        write_failed_tracks=args.write_failed_tracks,
    )
    tracking.run_tracking(workers=args.workers)


def main():
    args = parse_args()
    if not HAS_TRACKING:
        print("4BE-ETI is required for tracking. Install the 4BE-ETI package.", file=sys.stderr)
        sys.exit(1)

    out_base = args.output_base or os.path.join('data', 'PTV_center', args.case)

    if args.runs:
        runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    else:
        runs = discover_runs(out_base, args.rays_filename)

    if not runs:
        print(f"No runs with {args.rays_filename} found under {os.path.abspath(out_base)}", file=sys.stderr)
        sys.exit(1)

    print(f"Tracking (all runs): case={args.case}, runs={runs}")
    print(f"  box_size_x={args.box_size_x}, box_size_y={args.box_size_y}, box_size_z={args.box_size_z}, box_size_track={args.box_size_track}")
    print(f"  dt={args.dt}, rep_rate={args.rep_rate}, workers={args.workers}")
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
