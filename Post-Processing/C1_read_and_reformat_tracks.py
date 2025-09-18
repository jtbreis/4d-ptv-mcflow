# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:43:22 2023

@author: ferran6am
This code is rewritting the trajectories data in another format. 
Might not be useful depending on the output of the new tracking code.
"""

import os
import sys
from pathlib import Path
import numpy as np
import h5py
# import constants as cst


def read_hdf5(filepath):
    output_file_path = filepath.parent / 'tracks.h5'
    # Read data and rearrange per pair
    with h5py.File(filepath, 'r') as input_file:
        data = list(input_file.keys())
        fields = [name for name in data if name not in ['ntraj', 'L', '/', 'trackingparameters']]
        
        L = input_file['/L'][0]  # number of frames for each trajectory
        id_traj = input_file['/ntraj'][0]  # indices of each trajectory
        Ntraj = len(L)  # number of trajctories
        
        with h5py.File(output_file_path, 'w') as output_file:
            tracks = []  # final list of all the trajectories
            
            c = 0  # indice for the beginning of each track
            for kt in range(Ntraj):  # loop on the trajectories
                if kt % 100 == 0:
                    print(f'{str(kt)}/{str(Ntraj)}')
                nframes = int(L[kt])  # number of frames in the trajectory
                # initialise info for that trajectory
                track_data = {'L': nframes, 'id_traj': int(id_traj[kt])} 
                
                for field in fields:  # loop over x, y, z 
                    tracks_temp = input_file['/' + field][0]
                    
                    track_data[field] = tracks_temp[c:c + nframes]
                #tracks.append(track_data)
                c += nframes
                group_name = f'traj_{str(int(id_traj[kt]))}'
                
                # Create a new dataset in the output file and save the modified data
                group = output_file.create_group(group_name)

                for key, value in track_data.items():
                    if isinstance(value, np.ndarray):
                        # If the value is an array, create a dataset within the group
                        group.create_dataset(key, data=value)
                    else:
                        # If the value is a scalar, create an attribute for the group
                        group.attrs[key] = value

    return tracks


# Local directory
path = Path.cwd()
expe = 'HIT_30V_qa900lpm_qw1.5lpm_23A_set1'  # name of the experiment
filepath = Path('data/amelie/Processed-DATA') / expe / 'tracks_rays_out_cpp.h5'

result = read_hdf5(filepath)

# Access data from the result
for track in result:
    print(track)
