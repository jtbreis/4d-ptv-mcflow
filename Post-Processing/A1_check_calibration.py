# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:56:55 2023

@author: ferran6am
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import scipy.io as io
import mat73
import matplotlib.pyplot as plt
# Using tex's style
plt.style.use('tex')
plt.close('all')

# Local directory
path = Path.cwd()

# Calibration directory
file_calib = 'C:/Users/ferran6am/Documents/09_Stereomatching/03_Analysis/DATA/MyCalibration/calib.mat'

dat = mat73.loadmat(file_calib)  # load mat file
planes = range(-5,6,1)
cams = [1, 2 , 3]

for i,p in enumerate(planes):
    calib = dat['calib'][i]
    
   
    
    for c,cam in enumerate(cams):
        zpos, pimg, pos3D, *_ = calib[c]  # camera c 
        print(i, zpos, zpos)
    
        fig1  = plt.figure()   # figure camera c
        ax_x1 = fig1.add_subplot(121)
        ax_y1 = fig1.add_subplot(122)
        ax_x1.scatter(pimg[:,0], pimg[:,1], marker='.', c=pos3D[:,0], cmap='plasma')
        ax_y1.scatter(pimg[:,0], pimg[:,1], marker='.', c=pos3D[:,1], cmap='plasma')
    
    
        for ax in [ax_x1, ax_y1]:
            ax.set_xlabel('X [pxl]')
            ax.set_ylabel('Y [pxl]')
            
            ax.set_xlim(0, 2560)
            ax.set_ylim(0,1600)
            ax.invert_yaxis()
            ax.set_aspect('equal')