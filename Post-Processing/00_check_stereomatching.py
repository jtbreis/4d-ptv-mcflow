# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:05:23 2023

@author: ferran6am
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import struct
import cv2 as cv
import mat73
import scipy.io as io
import matplotlib.pyplot as plt
from matplotlib import image
sys.path.append('C:/Users/ferran6am/Documents/Python Scripts')
sys.path.append('C:/Users/ferran6am/Documents/09_PTV_UW/03_analysis/01_PTV')
import constants as c
import Plot_settings as ps
# Using tex's style
plt.style.use('tex')

def image_show(image, nrows=1, ncols=1, cmap='gray', **kwargs):
    w, h = 2560, 1600
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w/300, h/300))
    ax.imshow(image, cmap='gray')
    ax.grid(False)
    # ax.axis('off')
    ax.axis((0, w, h, 0))
    ax.set_xlabel('X [pxl]')
    ax.set_ylabel('Y [pxl]')
    return fig, ax

# Local directory
path = Path.cwd()
expe = 'Experiment1'  # name of the experiment

# Path to the original images
path_imgs = c.path_data / expe

# Path to save figures
path_fig = c.path_processed_data / expe / 'Figures'
path_fig.mkdir(parents=True, exist_ok=True)

minframes = 1
maxframes = 10
frameid = minframes

file_matches = c.path_processed_data / expe / 'matched_th200_bkgdmean_T3cam3_1-10.dat'

plot_frames = [i+1 for i in range(12)]


file_calib = io.loadmat(c.path_data / 'MyCalibration1_y52cm' / f'coefficients_transformation.mat')
calib = file_calib['coeffs_trans']
z_calib = np.linspace(-5,5,11)

fin_m = open(file_matches, 'rb')
numpts = fin_m.read(4)                                       # Read 4 bytes header
while(len(numpts)>0 and frameid <= maxframes):               # If something is read
    print("# frame", frameid)
    nmatch = struct.unpack('I', numpts)[0]                   # Interpret header as 4 byte uint
    print(nmatch)
    # nmatch is the number of matches
    # pack(fmt, values)  format, values to be stored as bytes data
    # = native standard no aligment
    
    data_bytes = fin_m.read(nmatch*26)                                 # 26 bytes per line 1+4*3+4+3*(1+2)
    matchdata  = struct.unpack('=' +('B4f'+3*"BH")*nmatch, data_bytes)              # Create string 'B4fBHBH'
    data = list(map(lambda i: list(matchdata[11*i:11*(i+1)]) ,range(len(matchdata)//11)))     
    # Reshape to 11*N np.arrayreshape converts everything to floats...
    
    colors = plt.cm.get_cmap("gist_ncar", len(data))
    for c, camid in enumerate([1, 2, 3]):  # loop on cameras
        print(f'Camera {camid}')
        
        # load original image
        image_name = f'{expe}_cam{camid}.{str(frameid).zfill(6)}.tif'
        image_path = path_imgs / f'cam{camid}' / image_name
        img = image.imread(image_path)
        
        fig, ax = image_show(img)
        ax.set_title(f'\# frame: {frameid}, camera {camid}')
        for m,match in enumerate(data):
            pos3d = match[1:4]
            d_rms = match[4]
            rayID_camID = match[5:]  # camID rayID all cameras
            
            camrayID = rayID_camID[c*2:(c+1)*2]  # camID rayID only one camera
            rayID = camrayID[1]
            
            if camrayID[0] != camid:
                print('Problem !')
            #print(rayID)
            x = pos3d[0]
            y = pos3d[1]
            z_rw = pos3d[2]
            
            z_rd = np.round(z_rw) # rounded z position to be on a target plane
            print(z_rd)
            
            if z_rd > 5 or z_rd < -5:  # if the z position is outside the calibration volume
                continue
            
            
            id_zcalib = np.where(z_calib == z_rd)[0]
            cam_calib = calib[id_zcalib,c]
            T1px2rw = cam_calib[0][3]
            XY = np.array([x, y, 1]).T
            #XY = [1, x, y, x*y, x*x, y*y, x*x*y, x*y*y, x**3, y**3]
            x_pxl, y_pxl, _ = np.dot(T1px2rw.T, XY)
            
            ax.plot(x_pxl, y_pxl, color=colors(m), marker='o',  markeredgewidth=1.5, fillstyle='none')
    
    # if frameid in plot_frames:
    #     ## Preparation figure
    #     # Figure with the planes
    #     sc_calib = ax_calib.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], marker='.', color=colors[frameid-1])
        
    #     # One figure for each plane
    #     fig_3d = plt.figure()
    #     ax_3d = fig_3d.add_subplot(projection='3d')
    #     sc = ax_3d.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], marker='.', c=d_rms, cmap='RdYlGn_r')#olor=colors[frameid-1])
    #     cbar = fig_3d.colorbar(sc, pad=0.08)
    #     ax_3d.set_xlabel('x [mm]')
    #     ax_3d.set_ylabel('y [mm]')
    #     ax_3d.set_zlabel('z [mm]')
    #     ax_3d.set_title(f"Plan {frameid}, z={int(zpos[frameid-1])} mm")
    #     cbar.set_label(r'$\sqrt{\langle d^2 \rangle} ~ [mm]$')
    #     fig_3d.savefig(path_fig / f'Plan{str(frameid)}_3dpositions.png', format='png', dpi=150)
    
    #     fig_error = plt.figure()
    #     ax_error = fig_error.add_subplot()
    #     sc = ax_error.scatter(pos3d[:,0], pos3d[:,1], marker='.', c=d_rms, cmap='RdYlGn_r')
    #     sc = ax_error.tricontourf(pos3d[:,0], pos3d[:,1], d_rms, levels=14, cmap='RdYlGn_r')
    #     cbar = fig_error.colorbar(sc, pad=0.08)
    #     cbar.set_label(r'$\sqrt{\langle d^2 \rangle} ~ [mm]$')
    #     ax_error.set_title(f"Plan {frameid}, z={int(zpos[frameid-1])} mm")
    #     ax_error.set_xlabel('x[mm]')
    #     ax_error.set_ylabel('y [mm]')
    #     fig_error.savefig(path_fig / f'Plan{str(frameid)}_contour_error.png', format='png', dpi=150)
    
    frameid += 1
    numpts = fin_m.read(4)                                              # Read next header


fin_m.close()

