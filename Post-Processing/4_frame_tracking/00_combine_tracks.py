from python_4be_eti import track
from python_4be_eti.utils.load_tracks import load_tracks

import os
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

base_folder = '/workspaces/4d-ptv-mcflow/data/julian/PTV_center'
case = 'TTI_opposing_gravity'
min_length = 4

runs = ['Run1', 'Run2', 'Run3', 'Run4']

samples = []

with h5py.File(base_folder+f'/{case}/{case}_tracks.h5', 'w') as f:
    pass

for run in runs:
    print(f'Processing run {run}')
    filename = os.path.join(base_folder, case, run, 'tracks.h5')

    run_number = int(run.replace('Run', ''))
    samples.append(load_tracks(filename, 1e-3, 10,
                   run_number, min_length=min_length))

# Combine all track objects from samples into a single list
all_tracks = [track_obj for sample in samples for track_obj in sample]

# Convert all tracks to a DataFrame
track_dicts = []
for i, tr in enumerate(all_tracks):
    n = len(tr.X)
    track_dicts.append(pd.DataFrame({
        'track_id': [i],
        'run': [tr.run],
        'X_0': [tr.X[0]],
        'X_1': [tr.X[1]],
        'X_2': [tr.X[2]],
        'X_3': [tr.X[3]],
        'Y_0': [tr.Y[0]],
        'Y_1': [tr.Y[1]],
        'Y_2': [tr.Y[2]],
        'Y_3': [tr.Y[3]],
        'Z_0': [tr.Z[0]],
        'Z_1': [tr.Z[1]],
        'Z_2': [tr.Z[2]],
        'Z_3': [tr.Z[3]],
        'X': [tr.X[0]],
        'Y': [tr.Y[0]],
        'Z': [tr.Z[0]],
        'vx_0': [tr.vx[0]],
        'vx_1': [tr.vx[1]],
        'vx_2': [tr.vx[2]],
        'vy_0': [tr.vy[0]],
        'vy_1': [tr.vy[1]],
        'vy_2': [tr.vy[2]],
        'vz_0': [tr.vz[0]],
        'vz_1': [tr.vz[1]],
        'vz_2': [tr.vz[2]],
        'vmag_0': [tr.vmag[0]],
        'vmag_1': [tr.vmag[1]],
        'vmag_2': [tr.vmag[2]],
        'vmean': [np.mean(tr.v, axis=0) if tr.v is not None and tr.v.size > 0 else np.array([np.nan, np.nan, np.nan])],
        'vstd': [np.std(tr.v, axis=0) if tr.v is not None and tr.v.size > 0 else np.array([np.nan, np.nan, np.nan])],
        'ax_0': [tr.ax[0]],
        'ax_1': [tr.ax[1]],
        'ay_0': [tr.ay[0]],
        'ay_1': [tr.ay[1]],
        'az_0': [tr.az[0]],
        'az_1': [tr.az[1]],
        'amag_0': [tr.amag[0]],
        'amag_1': [tr.amag[1]],
        'amean': [np.mean(tr.a, axis=0) if tr.a is not None and tr.a.size > 0 else np.array([np.nan, np.nan, np.nan])],
        'astd': [np.std(tr.a, axis=0) if tr.a is not None and tr.a.size > 0 else np.array([np.nan, np.nan, np.nan])],
        'time': [tr.time]
    }))
df = pd.concat(track_dicts, ignore_index=True)

# Write to parquet
parquet_path = os.path.join(base_folder, case, f'{case}_tracks.parquet')
df.to_parquet(parquet_path, index=False)
print(f'Wrote all tracks to {parquet_path}')
