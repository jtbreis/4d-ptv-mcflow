# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:05:23 2023

@author: ferran6am
Plot the 3D positions of the calibration points on the target 
to make sure the calibration is accurate.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import struct
import matplotlib.pyplot as plt

import constants as cst
import plot_settings as ps

# Using tex's style
# plt.style.use('tex')

# Local directory
path = Path.cwd()
expe = 'MyTestCalibration1_y52cm'  # name of the experiment

# Path to save figures
path_fig = Path(data/amelie/Figures) / expe
path_fig.mkdir(parents=True, exist_ok=True)

filename =  Path(data/amelie/Processed-DATA) / expe / 'matchedcam3_1-12.dat'
minframes = 1
maxframes = 12
zpos = np.linspace(-5, 5, 11)

colors = ps.discrete_cmap(maxframes-minframes, base_cmap='Greys')
frameid = minframes
plot_frames = [i+1 for i in range(12)]

## PREPARATION FIGURE
SAVE = 1
figsize = (6.299, 2.5)  # ps.set_size('thesis')
left, bottom = -0.05, 0.15

width, height = 0.92/2 - left/2 - 0.05, 0.94 - bottom
l1 = left   + width  + 0.16


fig_thesis = plt.figure()
ax1_thesis = fig_thesis.add_axes([left, bottom, width, height], projection='3d')
ax2_thesis = fig_thesis.add_axes([l1  , bottom, width, height])

fig_calib = plt.figure()
ax_calib = fig_calib.add_subplot(projection='3d')

fin = open(filename, 'rb')
numpts = fin.read(4)                                                 # Read 4 bytes header
while(len(numpts)>0 and frameid < maxframes):                        # If something is read
    print("# frame", frameid)
    nmatch = struct.unpack('I', numpts)[0]                           # Interpret header as 4 byte uint
    print(nmatch)
    # nmatch is the number of matches
    # pack(fmt, values)  format, values to be stored as bytes data
    # = native standard no aligment
    
    data_bytes = fin.read(nmatch*26)                                 # 26 bytes per line 1+4*3+4+3*(1+2)
    matchdata  = struct.unpack('=' +('B4f'+3*"BH")*nmatch, data_bytes)              # Create string 'B4fBHBH'
    data = list(map(lambda i: list(matchdata[11*i:11*(i+1)]) ,range(len(matchdata)//11)))     
    # Reshape to 11*N np.arrayreshape converts everything to floats...
    
    
    pos3d = []                                                        # store 3d positions of the matches
    d_rms = []                                                        # store root mean square distances
    
    for match in data:
        pos3d.append(match[1:4])
        d_rms.append(match[4])
    
    pos3d = np.array(pos3d)
    d_rms = np.array(d_rms)
    
    if frameid in plot_frames:
        ## Preparation figure
        # Figure with the planes
        sc_calib = ax_calib.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], marker='.', color=colors[frameid-1], zorder=-frameid)
        ax1_thesis.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], color=colors[frameid-1], s=1, zorder=(frameid-7), alpha=0.8)
        
        if int(zpos[frameid-1]) == 0:
            sc = ax2_thesis.scatter(pos3d[:,0], pos3d[:,1], marker='.', c=d_rms, cmap='RdYlGn_r')
            sc = ax2_thesis.tricontourf(pos3d[:,0], pos3d[:,1], d_rms, levels=14, cmap='RdYlGn_r')
        
        # One figure for each plane
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(projection='3d')
        sc = ax_3d.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], marker='.', c=d_rms, cmap='RdYlGn_r')#olor=colors[frameid-1])
        cbar = fig_3d.colorbar(sc, pad=0.08)
        ax_3d.set_xlabel('x [mm]')
        ax_3d.set_ylabel('y [mm]')
        ax_3d.set_zlabel('z [mm]')
        ax_3d.set_title(f"Plan {frameid}, z={int(zpos[frameid-1])} mm")
        cbar.set_label(r'$\sqrt{\langle d^2 \rangle} ~ [mm]$')
        fig_3d.savefig(path_fig / f'Plan{str(frameid)}_3dpositions.png', format='png', dpi=150)
    
        fig_error = plt.figure()
        ax_error = fig_error.add_subplot()
        sc = ax_error.scatter(pos3d[:,0], pos3d[:,1], marker='.', c=d_rms, cmap='RdYlGn_r')
        sc = ax_error.tricontourf(pos3d[:,0], pos3d[:,1], d_rms, levels=14, cmap='RdYlGn_r')
        cbar = fig_error.colorbar(sc, pad=0.08)
        cbar.set_label(r'$\sqrt{\langle d^2 \rangle} ~ [mm]$')
        ax_error.set_title(f"Plan {frameid}, z={int(zpos[frameid-1])} mm")
        ax_error.set_xlabel('x[mm]')
        ax_error.set_ylabel('y [mm]')
        fig_error.savefig(path_fig / f'Plan{str(frameid)}_contour_error.png', format='png', dpi=150)
    
    frameid += 1
    numpts = fin.read(4)                                              # Read next header


fin.close()

ax_calib.set_xlabel('x [mm]')
ax_calib.set_ylabel('y [mm]')
ax_calib.set_zlabel('z [mm]')
fig_calib.savefig(path_fig / 'Calibration_plate.png', format='png', dpi=300)



ax1_thesis.set_xlabel(r'$x ~ [mm]$')
ax1_thesis.set_ylabel(r'$z ~ [mm]$')
ax1_thesis.set_zlabel(r'$y ~ [mm]$')

ax2_thesis.set_xlabel(r'$x ~ [mm]$')
ax2_thesis.set_ylabel(r'$z ~ [mm]$')
ax2_thesis.set_aspect('equal')
cbar_th = fig_thesis.colorbar(sc, pad=0.15, orientation="horizontal",fraction=0.07,anchor=(1.0,0.0))
cbar_th.set_label(r'$\sqrt{\langle d^2 \rangle} ~ [mm]$')
ax2_thesis.set_title(r"$y=0 ~ mm$")

axes = [ax1_thesis, ax2_thesis]
labels = ['a', 'b']

for ax,lbl in zip(axes, labels):
    ax.annotate(f'({lbl})', xy=(0.15, 0.99), xytext=(0, 10), 
                  xycoords='axes fraction', 
                  textcoords='offset points', fontsize=11, 
                  fontweight="bold")

if SAVE == 1:
    namefig = path_fig / 'Figure_calibration'
    # fig_thesis.savefig(namefig.with_suffix('.pdf'), format='pdf', dpi=300)
    fig_thesis.savefig(namefig.with_suffix('.png'), format='png', dpi=150)