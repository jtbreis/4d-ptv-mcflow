# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 04:19:57 2023

@author: ferran6am
"""

import os
import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline, splrep, splev
import constants as cst

# Using tex's style
# plt.style.use('tex')

class trajectory():
    """ Class giving information for one trajectory """
    def __init__(self, filepath, id_track):  # path to the file, number of the track
        self.filepath = filepath

        # Path to save the figures
        self.path_save_fig = path / 'Figures'

        # Read data and rearrange per pair
        with h5py.File(filepath, 'r') as file:
            data = list(file.keys())
        self.Ntraj = len(data)
        self.id_track = id_track  # storing the track id

        self.x, self.y, self.z, self.frames = self.read_track()
        
    def read_track(self):
        group_name = f'traj_{self.id_track}'
        with h5py.File(self.filepath, 'r') as file:
            track = file[group_name]  # a track is a group in the hdf5 file
            frames = track.attrs['L']  # number of frames for that track
            x_mm = track['x'][:]  # coordinates are datasets in the group "track_{id_track}"
            y_mm = track['y'][:]
            z_mm = track['z'][:]
            
            # Conversion from mm to meters
            x, y, z = x_mm*1e-3, y_mm*1e-3, z_mm*1e-3
        return x, y, z, frames

    def plot_track3d(self, fitted_track=False):
        """" Function to view the trajectory in 3d """
        fig = plt.figure()
        ax3d = fig.add_subplot(projection='3d')
        ax3d.plot(self.x, self.y, self.z, 'ro')
        if fitted_track:
            self.compute_velocity(method='polynomial_regression')
            ax3d.plot(self.x_fit, self.y_fit, self.z_fit, label='polynomial fit')

        # Decorations figure
        ax3d.set_xlabel(r'$x ~ [mm]$')
        ax3d.set_ylabel(r'$y ~ [mm]$')
        ax3d.set_zlabel(r'$z ~ [mm]$')
        ax3d.set_aspect('auto')

        ## Make panes transparent
        ax3d.xaxis.pane.fill = False  # Left pane
        ax3d.yaxis.pane.fill = False  # Right pane

    def plot_fitted_coordinates(self):
        #todo : ajouter l'intervalle de confiance
        t = np.arange(len(self.x)) * cst.dt  # time vector
        self.compute_velocity()

        fig_x = plt.figure()
        ax_x = fig_x.add_subplot()
        ax_x.plot(t, self.x, marker='o', markersize=3, color='C0', lw=0, label=r'\textbf{measurement data}')
        ax_x.plot(t, self.x_fit, color='royalblue', lw=2.0, label=r'\textbf{fitted data}')
        ax_x.set_ylabel(r'$x$ \textbf{-coordinate} $[m]$')

        fig_y = plt.figure()
        ax_y = fig_y.add_subplot()
        ax_y.plot(t, self.y, marker='o', markersize=3, color='C0', lw=0, label=r'\textbf{measurement data}')
        ax_y.plot(t, self.y_fit, color='royalblue', lw=2.0, label=r'\textbf{fitted data}')
        ax_y.set_ylabel(r'$y$ \textbf{-coordinate} $[m]$')

        fig_z = plt.figure()
        ax_z = fig_z.add_subplot()
        ax_z.plot(t, self.z, marker='o', markersize=3, color='C0', lw=0, label=r'\textbf{measurement data}')
        ax_z.plot(t, self.z_fit, color='royalblue', lw=2.0, label=r'\textbf{fitted data}')
        ax_z.set_ylabel(r'$z$ \textbf{-coordinate} $[m]$')

        for ax in [ax_x, ax_y, ax_z]:
            ax.set_xlabel('$t ~ [time]$')

        fig_x.savefig(self.path_save_fig / 'fitting_x_coordinate.png')
        fig_z.savefig(self.path_save_fig / 'fitting_z_coordinate.svg', format='svg', dpi=300)

        # plt.plot(t, self.y, 'ro', t, y_fit, 'b')

    def finite_difference(self, x, y, z):
        """ Compute Finite Differences """
        Vx = np.diff(x) / cst.dt
        Vy = np.diff(y) / cst.dt
        Vz = np.diff(z) / cst.dt
        return Vx, Vy, Vz

    def compute_velocity(self, method='finite_difference'):
        t = np.arange(len(self.x)) * cst.dt  # time vector
        # length_V = len(self.x) - 1

        if method == 'finite_difference':  # Direct finite difference of the positions
            vx, vy, vz = self.finite_difference(self.x, self.y, self.z)

        elif method == 'polynomial_regression': # Compute the velocity after conducting a polynomial regression

            """ Polynomial regression, one coordinate at a time """
            px = np.polynomial.polynomial.polyfit(t, self.x, deg=2)
            py = np.polynomial.polynomial.polyfit(t, self.y, deg=2)
            pz = np.polynomial.polynomial.polyfit(t, self.z, deg=2)

            self.x_fit = np.polynomial.polynomial.polyval(t, px)
            self.y_fit = np.polynomial.polynomial.polyval(t, py)
            self.z_fit = np.polynomial.polynomial.polyval(t, pz)
            vx, vy, vz = self.finite_difference(self.x_fit, self.y_fit, self.z_fit)

        elif method == 'b-spline':
            # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.interpolate.make_splrep.html
            tck = splrep(t, self.x, k=3)
            x_smooth = splev(t, tck)

            plt.plot(t, self.x, 'ro', t, x_smooth)

        #todo : Try with a spine
        # https: // stackoverflow.com / questions / 45179024 / scipy - bspline - fitting - in -python
        # print(len(vx), length_V)
        # print(vy_raw, vy)

        return vx, vy, vz

    # def compute_acceleration(self):
    #     length_A = len(self.x) - 2
    #
    #     """ Compute acceleration here """



class trajectories():
    """ Class processing all the trajectories of a dataset """
    def __init__(self, expe):  # path to the file, number of the track
        self.case = expe
        self.path_processed_data = Path('data/amelie/Processed-DATA') / self.case
        self.filepath = str(self.path_processed_data / 'tracks.h5')

        self.path_fig = Path('data/amelie/Figures') / self.case
        self.path_fig.mkdir(parents=True, exist_ok=True)

        # Read data and rearrange per pair
        with h5py.File(self.filepath, 'r') as file:
            data = list(file.keys())
        self.Ntraj = len(data)
    
    def compute_velocity(self, method='finite_difference'):
        V_ttx, V_tty, V_ttz = [], [], []
        Vel = []  # total velocity
        
        for id_track in range(1, self.Ntraj):
            if id_track%100 == 0: print(f'{id_track:.0f}/{self.Ntraj:.0f}')
            track = trajectory(self.filepath, id_track)
            Vx, Vy, Vz = track.compute_velocity(method=method)
            
            V_ttx.append(Vx)
            V_tty.append(Vy)
            V_ttz.append(Vz)
            
            V_tot = np.sqrt(Vx**2 + Vy**2 + Vz**2)
            Vel.append(V_tot)

        Vx_flat = np.hstack(V_ttx)
        Vy_flat = np.hstack(V_tty)
        Vz_flat = np.hstack(V_ttz)
        Vel_flat = np.hstack(Vel)

        # For now only saving the velocities as 3 big flattened arrays
        np.savez_compressed(self.path_processed_data / f'velocities_{method}.npz',
                            Vx=Vx_flat, Vy=Vy_flat, Vz=Vz_flat)

        return Vx_flat, Vy_flat, Vz_flat
    
    def compute_mean_velocities(self, Vx_flat, Vy_flat, Vz_flat):
        
        mVx = np.mean(Vx_flat)
        mVy = np.mean(Vy_flat)
        mVz = np.mean(Vz_flat)
        return mVx, mVy, mVz

    def compute_std_velocities(self, Vx_flat, Vy_flat, Vz_flat):

        sVx = np.std(Vx_flat)
        sVy = np.std(Vy_flat)
        sVz = np.std(Vz_flat)
        return sVx, sVy, sVz

    def plot_histogram(self, method, nbins):
        dat = np.load(self.path_processed_data / f'velocities_{method}.npz')
        Vx, Vy, Vz = dat['Vx'], dat['Vy'], dat['Vz']
        # Compute moments
        mVx, mVy, mVz = self.compute_mean_velocities(Vx, Vy, Vz)
        sVx, sVy, sVz = self.compute_std_velocities(Vx, Vy, Vz)

        bins_yz = np.linspace(-8, 8, nbins)
        
        figVx, axVx = plt.subplots()
        axVx.hist(Vx, bins=np.linspace(mVx - 3 * sVx, mVx + 3 * sVx, nbins), alpha=0.7, color='navy')
        axVx.set_xlabel(r'$V_x$')
        
        figVy, axVy = plt.subplots()
        axVy.hist(Vy, bins=bins_yz, alpha=0.7, color='navy')
        axVy.set_xlabel(r'$V_y$')
        
        figVz, axVz = plt.subplots()
        axVz.hist(Vz, bins=bins_yz, alpha=0.7, color='navy')
        axVz.set_xlabel(r'$V_z$')

        for ax in [axVx, axVy, axVz]:
            ax.set_ylabel('Counts')
            # ax.legend()
        
        figVx.savefig(self.path_fig / f'Fig_pdf_Vx_{method}.png', format='png', dpi=150)
        figVy.savefig(self.path_fig / f'Fig_pdf_Vy_{method}.png', format='png', dpi=150)
        figVz.savefig(self.path_fig / f'Fig_pdf_Vz_{method}.png', format='png', dpi=150)


# Local directory
path = Path.cwd()
path_processed_data = Path('data/amelie/Processed-DATA')  # Path to the processed data
expe = 'HIT_30V_qa900lpm_qw1.5lpm_23A_set1'  # name of the experiment
filepath = str(path_processed_data / expe / 'tracks.h5')

# one_track = trajectory(filepath, id_track=100)
# one_track.compute_velocity()
# one_track.plot_fitted_coordinates()
# Read all the tracks
traj = trajectories(expe)

# Velocity computation with the direct finite difference
# vx_fdiff, vy_fdiff, vz_fdiff = traj.compute_velocity(method='finite_difference')
traj.plot_histogram(method='finite_difference', nbins=128)

# Velocity computation after conducting a polynomial regression
# vx_poly, vy_poly, vz_poly = traj.compute_velocity(method='polynomial_regression')
traj.plot_histogram(method='polynomial_regression', nbins=128)

