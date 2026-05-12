/*
 * Spatial partition of the stereo matching volume: assign rays to sub-boxes,
 * run voxel traversal / candidate generation per box in parallel, merge candidates,
 * then one global select-approved pass.
 */

#ifndef STM_SPATIAL_BOXES_H
#define STM_SPATIAL_BOXES_H

#include "STM_types.h"
#include <vector>

std::vector<candidatematch> SpaceTraversalMatchingSpatialBoxes(
    const std::vector<ray>& raydata,
    const boundingboxspec& bb,
    int n_boxes,
    int overlap_cells,
    int maxmatchesperray,
    unsigned int mincameras,
    double maxdistance,
    double multiplematchesperraymindistance,
    STMFrameTiming* timing_out,
    bool verbose);

#endif
