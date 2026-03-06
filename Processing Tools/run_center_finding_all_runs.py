"""
Run center finding only for all runs in a case/dataset (all cameras per run).
Output: data/PTV_center/{case}/{run}/ for use by stereo-matching and tracking containers.
"""
import os
import sys
import argparse
from python_4dptv.center_finding.detect_centers import CenterFinding

os.chdir('/workspaces/4d-ptv-mcflow')

_DEFAULT_CASE = 'TTI_opposing_gravity'
_DEFAULT_DATASET = '2025-09-11-ParticleTracking'
CAMERAS = [1, 2, 3, 4]


def parse_args():
    p = argparse.ArgumentParser(
        description='Run center finding for all runs in a case/dataset.'
    )
    p.add_argument('--case', type=str, default=_DEFAULT_CASE,
                   help='Case name (e.g. TTI_aligned_with_gravity).')
    p.add_argument('--dataset', type=str, default=_DEFAULT_DATASET,
                   help='Dated dataset folder (e.g. 2025-09-11-ParticleTracking).')
    p.add_argument('--runs', type=str, default=None,
                   help='Comma-separated run names. If omitted, all runs under raw_data/{dataset}/{case}/ are processed.')
    p.add_argument('--raw-data-base', type=str, default=None,
                   help='Override raw base path. Default: raw_data/{dataset}/{case}. Must contain Run1, Run2, ... (each with CameraN.cine).')
    p.add_argument('--output-base', type=str, default=None,
                   help='Override output base path. Default: data/PTV_center/{case}.')
    # Center finding parameters
    p.add_argument('--particle-diameter', type=int, default=7,
                   help='Particle diameter for trackpy (default: 7).')
    p.add_argument('--threshold', type=int, default=10,
                   help='Detection threshold (default: 10).')
    p.add_argument('--minmass', type=int, default=0,
                   help='Minimum mass (default: 0).')
    p.add_argument('--separation', type=int, default=None,
                   help='Minimum separation between particles (default: particle_diameter/2).')
    p.add_argument('--cores', type=str, default='1',
                   help='Number of processes for trackpy batch (default 1). Use "auto" for all CPUs.')
    return p.parse_args()


def discover_runs(raw_data_base):
    """Find run directories (Run1, Run2, ...) that contain Camera1.cine. Camera data is always in subfolder RunX."""
    if not os.path.isdir(raw_data_base):
        return [], f"not a directory or does not exist"
    runs = []
    for name in sorted(os.listdir(raw_data_base)):
        path = os.path.join(raw_data_base, name)
        if not os.path.isdir(path):
            continue
        # Expect Run1, Run2, Run10, etc. (case-insensitive) containing Camera1.cine
        if name.lower().startswith('run') and name[3:].isdigit():
            if os.path.isfile(os.path.join(path, 'Camera1.cine')):
                runs.append(name)
    return runs, None


def run_center_finding_for_run(raw_data_path, process_data_path, args):
    for cam in CAMERAS:
        print(f"  Center finding Camera{cam}")
        filename = os.path.join(raw_data_path, f'Camera{cam}.cine')
        if not os.path.isfile(filename):
            print(f"  Skip Camera{cam}: not found {filename}")
            continue
        cf = CenterFinding(
            filename, process_data_path,
            particle_diameter=args.particle_diameter,
            threshold=args.threshold,
            minmass=args.minmass,
        )
        if args.separation is not None:
            cf.separation = args.separation
        cf.remove_frames()
        cf.find_centers(processes=args.cores)
        cf.write_matches()
        saved_path = os.path.abspath(cf.output_path)
        if os.path.isfile(saved_path):
            size = os.path.getsize(saved_path)
            print(f"  Saved to {saved_path} ({size} bytes)")
        else:
            print(f"  WARNING: expected file not found at {saved_path}", file=sys.stderr)
        del cf


def _parse_cores(cores_str):
    """Return int (>=1) or 'auto' for trackpy processes."""
    if cores_str.strip().lower() == 'auto':
        return 'auto'
    try:
        n = int(cores_str)
        if n < 1:
            raise ValueError('--cores must be >= 1 or "auto"')
        return n
    except ValueError as e:
        if 'invalid literal' in str(e).lower():
            raise ValueError('--cores must be a positive integer or "auto"') from e
        raise


def main():
    args = parse_args()
    try:
        args.cores = _parse_cores(args.cores)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    raw_base = args.raw_data_base or os.path.join('raw_data', args.dataset, args.case)
    out_base = os.path.join(args.output_base, args.case) or os.path.join('data', 'PTV_center', args.case)

    if args.runs:
        runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    else:
        runs, reason = discover_runs(raw_base)
        if reason:
            print(f"Cannot discover runs: {raw_base} {reason}", file=sys.stderr)
            sys.exit(1)

    if not runs:
        abs_base = os.path.abspath(raw_base)
        try:
            contents = os.listdir(raw_base) if os.path.isdir(raw_base) else []
        except OSError:
            contents = []
        print(f"No runs found under {abs_base}", file=sys.stderr)
        print(f"  Expected subfolders Run1, Run2, ... each containing Camera1.cine. Contents: {contents}", file=sys.stderr)
        sys.exit(1)

    out_base_abs = os.path.abspath(out_base)
    print(f"Output will be saved to: {out_base_abs}")
    # Check writability (e.g. Docker mount must be read-write; if you see failures, check the volume mount)
    if not os.path.exists(out_base):
        try:
            os.makedirs(out_base, exist_ok=True)
        except OSError as e:
            print(f"WARNING: cannot create output directory: {e}", file=sys.stderr)
    if os.path.exists(out_base) and not os.access(out_base, os.W_OK):
        print(f"WARNING: output directory is not writable: {out_base_abs}", file=sys.stderr)
    if os.path.exists(out_base) and os.access(out_base, os.W_OK):
        # If running in Docker, data/ is typically a volume mount; files here appear on the host.
        if out_base_abs.startswith('/workspaces/') and 'data' in out_base.split(os.sep):
            print("  (In Docker: this path is under the data volume; files appear on the host at OUTPUT_DIR/PTV_center/...)")
    print(f"Center finding (all runs): case={args.case}, dataset={args.dataset}, runs={runs}")
    print(f"  particle_diameter={args.particle_diameter}, threshold={args.threshold}, minmass={args.minmass}, separation={args.separation}, cores={args.cores}")
    print(f"  raw_base={raw_base}, output_base={out_base}")

    for run in runs:
        raw_path = os.path.join(raw_base, run)
        out_path = os.path.join(out_base, run)
        if not os.path.isdir(raw_path):
            print(f"Skip {run}: missing {raw_path}")
            continue
        print(f"Processing run: {run}")
        os.makedirs(out_path, exist_ok=True)
        run_center_finding_for_run(raw_path, out_path, args)
        print(f"Done {run}")
    print("All runs completed.")


if __name__ == '__main__':
    main()
