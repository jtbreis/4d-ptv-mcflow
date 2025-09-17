# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 02:27:52 2023

@author: ferran6am
"""

import os
import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import ListedColormap, Normalize
sys.path.append('C:/Users/ferran6am/Documents/09_PTV_UW/03_analysis/01_PTV')
sys.path.append('C:/Users/ferran6am/Documents/Python Scripts')
import constants as c
import Plot_settings as ps
# Using tex's style
plt.style.use('tex')

def get_number_trajectory(filepath):
    with h5py.File(filepath, 'r') as file:
        data = list(file.keys())
    return len(data)

def read_hdf5(filepath, ntrack):  # path to the file, number of the track
    # Read data and rearrange per pair
    with h5py.File(filepath, 'r') as file:
        data = list(file.keys())
        
        group_name = f'traj_{ntrack}'
        trajectory = file['/' + group_name]
        frames = trajectory['frames']
        x, y, z = trajectory['x'][:], trajectory['y'][:], trajectory['z'][:]
    return x, y, z, frames

# Local directory
path = Path.cwd()
expe = 'HIT_30V_qa900lpm_qw1.5lpm_23A_set1'  # name of the experiment
filepath = c.path_processed_data / expe / 'tracks.h5'

Ntraj = get_number_trajectory(filepath)

## PREPARATION FIGURE
SAVE = 1
cmap = plt.get_cmap('Reds')
figsize = ps.set_size('thesis')

left, bottom = 0.02, 0.02
width, height = 0.98 - left, 0.98 - bottom


# Create a 3D scatter plot
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d', position=[left, bottom, width, height])

# Scale the 3d plot
x_scale = 12
y_scale = 8
z_scale = 1

scale = np.diag([x_scale, y_scale, z_scale, 1.0])
scale = scale*(1.0/scale.max())
scale[3,3] = 0.7

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)
ax.get_proj=short_proj

# add trajectories
for id_frame in range(1, Ntraj, Ntraj//100):
    print(id_frame)
    x, y, z, frames = read_hdf5(filepath, id_frame)
    
    npoints = len(x)
    # plot in segments with each segments of a different color
    for i in range(npoints-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], alpha=float(i)/(npoints-1),
                color='blue')

ax.set_xlabel(r'$x ~ [mm]$')
ax.set_ylabel(r'$z ~ [mm]$')
ax.set_zlabel(r'$y ~ [mm]$')
ax.set_xlim(-60, 60)
ax.set_ylim(-40, 40)
ax.set_zlim(-5, 5)
#ax.set_proj_type('ortho')
ax.set_aspect('auto')

## Make panes transparent
ax.xaxis.pane.fill = False # Left pane
ax.yaxis.pane.fill = False # Right pane

## Ticks and ticks labels
ticksz = [-4, 4]
ticksz_labels = [r'$-4$', r'$4$']

ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
ax.set_zticks(ticksz)
ax.set_zticklabels(ticksz_labels)
# View point carefully adjusted
ax.view_init(elev=50, azim=-100)  # Adjust these angles as needed

if SAVE == 1:
    namefig = c.path_ptv_figs / 'Figure_tracks'
    fig.savefig(namefig.with_suffix('.pdf'), format='pdf', dpi=300)
    fig.savefig(namefig.with_suffix('.png'), format='png', dpi=150)

""" LEFT TO DO 
Make grid and frame pretty
Make axis pretty
Replace Y label by Z label
Put only long tracks
"""
