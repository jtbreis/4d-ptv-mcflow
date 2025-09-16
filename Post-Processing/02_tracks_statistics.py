# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:15:32 2023

@author: ferran6am
Code to get some statistics on the tracks
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import constants as c

# Using tex's style
plt.style.use('tex')

def read_L_hdf5(filepath):
    # Read data and rearrange per pair
    with h5py.File(filepath, 'r') as input_file:
        data = list(input_file.keys())
        
        L = input_file['/L'][0]  # number of frames for each trajectory
    return L

# Local directory
path = Path.cwd()

# Path to the analysed data
expe = 'HIT_30V_qa900lpm_qw1.5lpm_23A_set1'  # name of the experiment
filepath = c.path_processed_data / expe / 'tracks_rays_out_cpp2.h5'

L = read_L_hdf5(filepath)

## PREPARATION FIGURE
SAVE = 1
figsize = (6.299, 2.5) # ps.set_size('thesis')

left, bottom = 0.1, 0.18
width, height = 0.95 - left, 0.95 - bottom

bins = np.arange(4,13)
hist, bin_edges = np.histogram(L, bins=bins, density=True)

fig = plt.figure()
ax = fig.add_axes([left, bottom, width, height])

ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='black')
#plt.hist(L, bins=bins, density=True, color='black', align='mid')

ax.grid(False)
ax.set_xlabel('Number of points per trajectory')
ax.set_ylabel('PDF density')
ax.set_xlim(3.5, 11.5)

if SAVE == 1:
    namefig = c.path_ptv_figs / 'Figure_number_points_per_track'
    fig.savefig(namefig.with_suffix('.pdf'), format='pdf', dpi=150)
    fig.savefig(namefig.with_suffix('.png'), format='png', dpi=150)