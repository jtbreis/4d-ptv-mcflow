# %%
from python_4dptv.center_finding.detect_centers import CenterFinding
import matplotlib.pyplot as plt
from skimage import exposure
import os
import numpy as np
os.chdir('/workspaces/4d-ptv-mcflow')

camera = 'Camera3'
file_path = f'raw_data/julian/2025-09-11-ParticleTracking/TTI_aligned_with_gravity/Run1/{camera}.cine'
output_path = 'data/julian/PTV_center/TTI_aligned_with_gravity/Run1/'

# %%
center_finding = CenterFinding(file_path, output_path, 7)
center_finding.remove_frames()  # only necessary for 4 frame tracking

# %%
parameter_sets = []
for diameter in range(3, 9, 2):
    for threshold in range(1, 3):
        for separation in range(2, 5):
            parameter_sets.append({
                'particle_diameter': diameter,
                'threshold': threshold,
                'separation': separation
            })

print(len(parameter_sets))
# %%
for idx, params in enumerate(parameter_sets):
    center_finding.particle_diameter = params['particle_diameter']
    center_finding.threshold = params['threshold']
    center_finding.separation = params['separation']
    center_finding.test_parameters(nframes=50,
                                   output=output_path+f'/ParamTest/{camera}/set_{idx}.pdf')

# %%
# center_finding.particle_diameter = 9
# center_finding.threshold = 2
# center_finding.separation = 3
# center_finding.test_parameters(output=output_path+f'/ParamTest/set_0.pdf')

# %%
center_finding.find_centers(0, 200)
# %%
center_finding.check_mass_distribution()

# %%
center_finding.write_matches()
