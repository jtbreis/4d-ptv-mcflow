import os
from python_4dptv.center_finding.detect_centers import CenterFinding
from python_4dptv.matching.rays import Rays
from python_4dptv.matching.stereomatching import StereoMatching
import sys
import argparse

os.chdir('/workspaces/4d-ptv-mcflow')
parser = argparse.ArgumentParser(description='Complete 4D-PTV workflow.')
parser.add_argument('--case_name', type=str,
                    default='TTI_opposing_gravity/Run1', help='Case Name')
args = parser.parse_args()

case_name = args.case_name
print(f'Starting Case {case_name}')

raw_data_path = f'/workspaces/4d-ptv-mcflow/raw_data/julian/2025-09-11-ParticleTracking/{case_name}'
process_data_path = f'/workspaces/4d-ptv-mcflow/data/julian/PTV_center/{case_name}'

cameras = [1, 2, 3, 4]
particle_diameter = 7

# Set StereoMatching Parameters
mincameras = 3
maxdistance = 0.2
multiplematchesperraydistance = 2
maxmatchesperray = 1
# Number of Voxels in direction [nx, ny, nz]
nvoxels = [600, 600, 500]
# Bounding Box [minX, maxX, minY, maxY, minZ, maxZ]
boundingbox = [-50, 50, -35, 35, -20, 20]

for cam in cameras:
    print(f"Starting Center Finding Camera {cam}")
    filename = os.path.join(raw_data_path + f'/Camera{cam}.cine')
    center_finding = CenterFinding(
        filename, process_data_path, particle_diameter=particle_diameter)
    center_finding.remove_frames()
    center_finding.find_centers()
    center_finding.write_matches()
    del center_finding

print("Starting Ray Computation")
rays = Rays(process_data_path)
rays.compute_rays()
rays.write_rays()
del rays

print("Starting Stereomatching")
stereomatching = StereoMatching(process_data_path, mincameras, maxdistance,
                                multiplematchesperraydistance, maxmatchesperray, nvoxels, boundingbox)
stereomatching.run_stereomatching()
