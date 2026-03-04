# %%
from python_4dptv.center_finding.detect_centers import CenterFinding
import matplotlib.pyplot as plt
import os
import sys
import argparse

os.chdir('/workspaces/4d-ptv-mcflow')

# Default case name and dated dataset folder (used when --raw-data-base/--output-base not set)
_DEFAULT_CASE = 'TTI_opposing_gravity'
_DEFAULT_DATASET = '2025-09-11-ParticleTracking'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find center-finding parameters: generate ParamTest PDFs for visual inspection.'
    )
    parser.add_argument(
        '--run',
        type=str,
        default='Run1',
        help='Run name (e.g. Run1, Run2). Appended to raw/output base paths.',
    )
    parser.add_argument(
        '--case',
        type=str,
        default=_DEFAULT_CASE,
        help='Case/experiment name (e.g. TTI_aligned_with_gravity). Used in paths: .../{case}/{run}.',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=_DEFAULT_DATASET,
        help='Dated dataset folder for raw path (e.g. 2025-09-11-ParticleTracking). Raw: .../raw_data/{dataset}/{case}/{run}.',
    )
    parser.add_argument(
        '--raw-data-base',
        type=str,
        default=None,
        help='Override: full base path for raw .cine data (run appended). If unset, uses raw_data/{dataset}/{case}.',
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default=None,
        help='Override: full base path for output (run appended). If unset, uses data/PTV_center/{case}.',
    )
    parser.add_argument(
        '--cameras',
        type=str,
        default='Camera1,Camera2,Camera3,Camera4',
        help='Comma-separated camera names.',
    )
    parser.add_argument(
        '--nframes',
        type=int,
        default=8,
        help='Number of frames per parameter set for test_parameters.',
    )
    return parser.parse_args()


args = parse_args()
if args.raw_data_base is None:
    args.raw_data_base = os.path.join('raw_data', args.dataset, args.case)
if args.output_base is None:
    args.output_base = os.path.join('data', 'PTV_center', args.case)
run = args.run
base_path = os.path.join(args.raw_data_base, run)
output_path = os.path.join(args.output_base, run)
cameras = [c.strip() for c in args.cameras.split(',') if c.strip()]

print(
    f"Center-finding parameter search: case={args.case}, dataset={args.dataset}, run={run}")
print(f"  base_path={base_path}, output_path={output_path}")

# %%
parameter_sets = []
for diameter in range(5, 9, 2):
    for threshold in range(1, 3):
        for separation in range(2, 5):
            parameter_sets.append({
                'particle_diameter': diameter,
                'threshold': threshold,
                'separation': separation
            })

print(f"{len(parameter_sets)} parameter sets × {len(cameras)} cameras")

# %%
# Generate test_parameters PDFs for visual inspection (all cameras, all parameter sets)
nframes = args.nframes
for camera in cameras:
    file_path = os.path.join(base_path, f'{camera}.cine')
    if not os.path.isfile(file_path):
        print(f"Skip {camera}: not found {file_path}")
        continue
    center_finding = CenterFinding(file_path, output_path, 7)
    center_finding.remove_frames()  # only necessary for 4 frame tracking
    out_dir = os.path.join(output_path, 'ParamTest', camera)
    os.makedirs(out_dir, exist_ok=True)
    for idx, params in enumerate(parameter_sets):
        center_finding.particle_diameter = params['particle_diameter']
        center_finding.threshold = params['threshold']
        center_finding.separation = params['separation']
        center_finding.test_parameters(
            nframes=nframes,
            output=os.path.join(out_dir, f'set_{idx}.pdf')
        )
    plt.close('all')
    print(f"Done {camera}")
