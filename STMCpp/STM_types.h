/*
 * Common STM type definitions
 *
 */

#ifndef STM_HEADER_H
#define STM_HEADER_H

#include <algorithm>
#include <vector>

struct cellvisit{
    double time;
    int dimension = -1;
    bool edgeQ = false;
};

struct cellid{
    long xi;
    long yi;
    long zi;
};

struct traversedcell{
    int camid = -1;
    int rayid = -1;
    struct cellid cellid;
};

struct camrayid{
    long camid;
    long rayid;
};

struct groupedcell{
    struct cellid cellid;
    std::vector<camrayid> camrayids;
};

struct candidatematch{
    double matchx = 0;
    double matchy = 0;
    double matchz = 0;
    double matcherror = -1;
    std::vector<camrayid> camrayids;
};

struct ray{
    int camid = -1;
    int rayid = -1;
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
};

struct transformedray{
    int camid = -1;
    int rayid = -1;
    double x;
    double y;
    double z;
    double vx;
    double vy;
    double vz;
    bool hit = false;
    bool inside = false;
};

struct boundingboxspec{
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    double zmin;
    double zmax;
    int nx;
    int ny;
    int nz;
};

struct hitpoint{
    double t;
    double posx;
    double posy;
    double posz;
};

struct STMFrameTiming {
    double import_rays_ms = 0;
    double prepare_rays_ms = 0;
    double voxel_traversal_ms = 0;
    double sort_traversed_ms = 0;
    double group_cells_ms = 0;
    double candidate_pairs_ms = 0;
    double permutations_dedup_ms = 0;
    double closest_point_ms = 0;
    double sort_candidate_matches_ms = 0;
    double select_approved_ms = 0;
    double write_output_ms = 0;
    double matching_total_ms = 0;

    void add(const STMFrameTiming& o) {
        import_rays_ms += o.import_rays_ms;
        prepare_rays_ms += o.prepare_rays_ms;
        voxel_traversal_ms += o.voxel_traversal_ms;
        sort_traversed_ms += o.sort_traversed_ms;
        group_cells_ms += o.group_cells_ms;
        candidate_pairs_ms += o.candidate_pairs_ms;
        permutations_dedup_ms += o.permutations_dedup_ms;
        closest_point_ms += o.closest_point_ms;
        sort_candidate_matches_ms += o.sort_candidate_matches_ms;
        select_approved_ms += o.select_approved_ms;
        write_output_ms += o.write_output_ms;
        matching_total_ms += o.matching_total_ms;
    }

    // Spatial sub-volumes run in parallel; take max per stage (approx. critical-path time), not sum.
    void merge_max_parallel_box_stages(const STMFrameTiming& o) {
        prepare_rays_ms = std::max(prepare_rays_ms, o.prepare_rays_ms);
        voxel_traversal_ms = std::max(voxel_traversal_ms, o.voxel_traversal_ms);
        sort_traversed_ms = std::max(sort_traversed_ms, o.sort_traversed_ms);
        group_cells_ms = std::max(group_cells_ms, o.group_cells_ms);
        candidate_pairs_ms = std::max(candidate_pairs_ms, o.candidate_pairs_ms);
        permutations_dedup_ms = std::max(permutations_dedup_ms, o.permutations_dedup_ms);
        closest_point_ms = std::max(closest_point_ms, o.closest_point_ms);
        sort_candidate_matches_ms = std::max(sort_candidate_matches_ms, o.sort_candidate_matches_ms);
   }
};

#endif
