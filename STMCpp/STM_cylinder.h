/*
 * Optional cylinder-based ray model (finite diameter from rays.h5) for STM.
 */

#ifndef STM_CYLINDER_H
#define STM_CYLINDER_H

// Thicken voxel traversal: extra isotropic shells beyond the original 6-neighbour halo,
// scaled by particle radius = diameter/2 in units of the smallest voxel edge length.
extern bool stm_cylinder_thicken_voxels;

// When candidate matches tie on camera count and RMS line distance, prefer larger
// pairwise cylinder overlap (sum of max(0, r_i+r_j - d_ij)) and larger min axis margin.
extern bool stm_cylinder_rank_matches;

// Cap on extra neighbour-expansion rounds (each round adds face-adjacent voxels).
extern int stm_cylinder_max_extra_shells;

// Shortest distance between two infinite lines (P1,u) and (P2,v); u,v assumed unit.
double stm_line_line_distance(double p1x, double p1y, double p1z, double ux, double uy, double uz,
                              double p2x, double p2y, double p2z, double vx, double vy, double vz);

#endif
