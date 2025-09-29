from python_4be_eti import track
from python_4be_eti.utils.load_tracks import load_tracks

import os
import h5py

base_folder = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center'
case = 'TTI_aligned_with_gravity'

runs = ['Run1', 'Run2', 'Run3', 'Run4']

samples = []

with h5py.File(base_folder+f'/{case}/{case}_tracks.h5', 'w') as f:
    pass

for run in runs:
    filename = os.path.join(base_folder, case, run, 'tracks.h5')

    run_number = int(run.replace('Run', ''))
    samples.append(load_tracks(filename, 1e-3, 10, run_number))

# Combine all track objects from samples into a single list
all_tracks = [track_obj for sample in samples for track_obj in sample]

with h5py.File(base_folder+f'/{case}/{case}_tracks.h5', 'w') as f:
    for idx, track in enumerate(all_tracks):
        grp = f.create_group(f'track_{idx:07d}')
        grp.create_dataset('X', data=track.X)
        grp.create_dataset('Y', data=track.Y)
        grp.create_dataset('Z', data=track.Z)
        grp.create_dataset('vx', data=track.vx)
        grp.create_dataset('vy', data=track.vy)
        grp.create_dataset('vz', data=track.vz)
        grp.create_dataset('time', data=track.time)
        grp.create_dataset('idx', data=track.idx)
        grp.create_dataset('run', data=track.run)
