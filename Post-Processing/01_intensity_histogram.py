# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:34:03 2023

@author: ferran6am
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2  # opencv function
import scipy.io as io
import matplotlib.pyplot as plt
from matplotlib import image
sys.path.append('C:/Users/ferran6am/Documents/09_PTV_UW/03_analysis/01_PTV')
import constants as c
# Using tex's style
plt.style.use('tex')
plt.close('all')
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

cams = [1, 2, 3]
resolution = [2560, 1600]

# Local directory
path = Path.cwd()
expe = 'Experiment1'  # name of the experiment


# Path to save figures
path_fig = c.path_processed_data / expe / 'Figures'
path_fig.mkdir(parents=True, exist_ok=True)


def image_show(image, nrows=1, ncols=1, cmap='gray', **kwargs):
    w, h = resolution
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(w/300, h/300))
    ax.imshow(image, cmap='gray')
    ax.axis((0, resolution[0], resolution[1], 0))
    ax.grid(False)
    # ax.axis('off')
    ax.set_xlabel('X [pxl]')
    ax.set_ylabel('Y [pxl]')
    return fig, ax

for ca,camid in enumerate(cams):
    # Path to the original images
    folder_imgs = c.path_data / expe / f'cam{camid}'
    image_files = list(folder_imgs.glob('*.tif'))
    print(ca)
    # Background file
    file_bkgd = c.path_processed_data / expe / f'Background_cam{camid}.mat'
    dat_bkgd = io.loadmat(file_bkgd, squeeze_me=True)
	
    
    # For histogram
    bins = np.arange(0,4097,10)
    cumul_hist_raw = 0  # average histogram
    cumul_hist_prc = 0  # average histogram remove background mean
    cumul_hist_prc2 = 0  # average histogram remove background min
    for imf in image_files:
        # Read image file
        img = image.imread(imf)
        
        # Image histogramm
        vals = img.ravel()
        counts, bins = np.histogram(vals, bins=bins)
        cumul_hist_raw += counts  # compute mean histogram
        
        # Remove background Min
        img2  = img - dat_bkgd['BackgroundMin']
        #image_show(img2)
        vals2 = img2.ravel()
        counts, bins = np.histogram(vals2, bins=bins)
        cumul_hist_prc += counts  # compute mean histogram
        
        # Remove background Mean
        img3  = img - dat_bkgd['BackgroundMean']
        #image_show(img3)
        vals3 = img3.ravel()
        counts, bins = np.histogram(vals3, bins=bins)
        cumul_hist_prc2 += counts  # compute mean histogram
    
    mean_hist_raw  = cumul_hist_raw/len(image_files)  # divide by number of frames
    mean_hist_prc  = cumul_hist_prc/len(image_files)  # divide by number of frames
    mean_hist_prc2 = cumul_hist_prc2/len(image_files)  # divide by number of frames

    fig_hist, ax_hist = plt.subplots(1, 1)
    ax_hist.bar(bins[1:] , mean_hist_raw, width=-10, align='edge', color=cycle[0], label='Raw images')
    
    fig_hist_prc, ax_hist_prc = plt.subplots(1, 1)
    ax_hist_prc.bar(bins[1:] , mean_hist_prc, width=-10, align='edge', color=cycle[1], label='Images minus backgroundmin')
    
    fig_hist_prc2, ax_hist_prc2 = plt.subplots(1, 1)
    ax_hist_prc2.bar(bins[1:] , mean_hist_prc2, width=-10, align='edge', color=cycle[2], label='Images minus backgroundmean')
    
    
    for ax in [ax_hist, ax_hist_prc, ax_hist_prc2]:
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Counts')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'Camera {camid}')
        ax.legend()
        
    fig_hist.savefig(path_fig / f'Intensity_histogram_cam{camid}_rawimages.png', format='png', dpi=150)
    fig_hist_prc.savefig(path_fig / f'Intensity_histogram_cam{camid}_bkgdmin.png', format='png', dpi=150)
    fig_hist_prc2.savefig(path_fig / f'Intensity_histogram_cam{camid}_bkgdmean.png', format='png', dpi=150)
