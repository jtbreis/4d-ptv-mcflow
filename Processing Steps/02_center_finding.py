# %%
from python_4dptv.center_finding.detect_centers import CenterFinding
import matplotlib.pyplot as plt
from skimage import exposure
import os
import numpy as np
os.chdir('/workspaces/4d-ptv-mcflow')

file_path = '/workspaces/4d-ptv-mcflow/raw_data/2025-11-13-ParticleTracking_AboveTTI/TTI_aligned_with_gravity/Run1/Camera4.cine'
output_path = '/workspaces/4d-ptv-mcflow/raw_data/tracking_tests'

# %%
center_finding = CenterFinding(file_path, output_path, 7)
center_finding.remove_frames()  # only necessary for 4 frame tracking

# %%
# print(np.max(center_finding.frames))
# print(center_finding.frames.pixel_type)
# %%
center_finding.particle_diameter = 7
center_finding.threshold = 2
center_finding.separation = 3
# center_finding.test_parameters(100)

# %%
center_finding.find_centers(0, 12)
# %%
center_finding.check_mass_distribution()

# %%
center_finding.write_matches()
