# %%
from python_4dptv.center_finding.detect_centers import CenterFinding
import os
os.chdir('/workspaces/4d-ptv-mcflow')

file_path = 'raw_data/julian/2025-09-11-ParticleTracking/TTI_aligned_with_gravity/Run1/Camera1.cine'
output_path = 'data/julian/PTV_center/TTI_aligned_with_gravity/Run1/'

# %%
center_finding = CenterFinding(file_path, output_path, 7)
center_finding.remove_frames()
# %%
center_finding.test_parameters()
# %%
center_finding.find_centers(0, 10)
# %%
center_finding.check_mass_distribution()

# %%
center_finding.write_matches()
