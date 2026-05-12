#include "STM_spatial_boxes.h"
#include "STM.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

namespace {

void make_bounds_from_bb(const boundingboxspec& bb, std::vector<std::vector<double>>& bounds) {
    bounds.clear();
    bounds.resize(3);
    bounds[0].clear();
    for (int i = 0; i < bb.nx; i++) {
        bounds[0].push_back(bb.xmin + i * (bb.xmax - bb.xmin) / bb.nx);
    }
    bounds[0].push_back(bb.xmax);
    bounds[1].clear();
    for (int i = 0; i < bb.ny; i++) {
        bounds[1].push_back(bb.ymin + i * (bb.ymax - bb.ymin) / bb.ny);
    }
    bounds[1].push_back(bb.ymax);
    bounds[2].clear();
    for (int i = 0; i < bb.nz; i++) {
        bounds[2].push_back(bb.zmin + i * (bb.zmax - bb.zmin) / bb.nz);
    }
    bounds[2].push_back(bb.zmax);
}

// Balanced integer factorization n = bx * by * bz (minimize max(bx,by,bz) - min(...)).
void factor_box_grid(int n, int& bx, int& by, int& bz) {
    int best = std::numeric_limits<int>::max();
    bx = n;
    by = 1;
    bz = 1;
    for (int a = 1; a <= n; ++a) {
        if (n % a != 0) {
            continue;
        }
        const int n2 = n / a;
        for (int b = 1; b <= n2; ++b) {
            if (n2 % b != 0) {
                continue;
            }
            const int c = n2 / b;
            const int mx = std::max(a, std::max(b, c));
            const int mn = std::min(a, std::min(b, c));
            const int score = mx - mn;
            if (score < best) {
                best = score;
                bx = a;
                by = b;
                bz = c;
            }
        }
    }
}

bool build_sub_boxes(const boundingboxspec& global, int bx, int by, int bz, int overlap_cells,
                     std::vector<boundingboxspec>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(bx * by * bz));

    const double xw = global.xmax - global.xmin;
    const double yw = global.ymax - global.ymin;
    const double zw = global.zmax - global.zmin;
    const int ov = std::max(0, overlap_cells);

    for (int iz = 0; iz < bz; ++iz) {
        for (int iy = 0; iy < by; ++iy) {
            for (int ix = 0; ix < bx; ++ix) {
                boundingboxspec sub = global;

                const int x0i = ix * global.nx / bx;
                const int x1i = (ix + 1) * global.nx / bx;
                const int y0i = iy * global.ny / by;
                const int y1i = (iy + 1) * global.ny / by;
                const int z0i = iz * global.nz / bz;
                const int z1i = (iz + 1) * global.nz / bz;

                // Expand by ov global voxels on each face (clamped) so adjacent boxes share a halo region.
                const int x0e = std::max(0, x0i - ov);
                const int x1e = std::min(global.nx, x1i + ov);
                const int y0e = std::max(0, y0i - ov);
                const int y1e = std::min(global.ny, y1i + ov);
                const int z0e = std::max(0, z0i - ov);
                const int z1e = std::min(global.nz, z1i + ov);

                sub.xmin = global.xmin + xw * static_cast<double>(x0e) / static_cast<double>(global.nx);
                sub.xmax = global.xmin + xw * static_cast<double>(x1e) / static_cast<double>(global.nx);
                sub.ymin = global.ymin + yw * static_cast<double>(y0e) / static_cast<double>(global.ny);
                sub.ymax = global.ymin + yw * static_cast<double>(y1e) / static_cast<double>(global.ny);
                sub.zmin = global.zmin + zw * static_cast<double>(z0e) / static_cast<double>(global.nz);
                sub.zmax = global.zmin + zw * static_cast<double>(z1e) / static_cast<double>(global.nz);

                sub.nx = x1e - x0e;
                sub.ny = y1e - y0e;
                sub.nz = z1e - z0e;

                if (sub.nx < 1 || sub.ny < 1 || sub.nz < 1) {
                    return false;
                }
                out.push_back(sub);
            }
        }
    }
    return true;
}

std::vector<candidatematch> dedupe_matches_by_camrayids(std::vector<candidatematch> all) {
    for (size_t i = 0; i < all.size(); ++i) {
        std::sort(all[i].camrayids.begin(), all[i].camrayids.end(), comparecamrayidscamray);
    }
    std::sort(all.begin(), all.end(), [](const candidatematch& a, const candidatematch& b) {
        return comparecamrayidsordered(a.camrayids, b.camrayids);
    });
    std::vector<candidatematch> out;
    out.reserve(all.size());
    for (size_t i = 0; i < all.size(); ++i) {
        if (out.empty() || !comparecamrayidsidentical(all[i].camrayids, out.back().camrayids)) {
            out.push_back(all[i]);
        } else if (all[i].matcherror < out.back().matcherror) {
            out.back() = all[i];
        }
    }
    return out;
}

}  // namespace

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
    bool verbose) {
    if (n_boxes <= 1 || raydata.empty()) {
        std::vector<std::vector<double>> bounds;
        make_bounds_from_bb(bb, bounds);
        return SpaceTraversalMatching(raydata, bb, bounds, maxmatchesperray, mincameras, maxdistance,
                                      multiplematchesperraymindistance, timing_out);
    }

    int bx = 1;
    int by = 1;
    int bz = 1;
    factor_box_grid(n_boxes, bx, by, bz);

    std::vector<boundingboxspec> boxes;
    if (!build_sub_boxes(bb, bx, by, bz, overlap_cells, boxes)) {
        if (verbose) {
            std::cout << "[spatial-boxes] partition invalid for current nx,ny,nz; using single volume.\n";
        }
        std::vector<std::vector<double>> bounds;
        make_bounds_from_bb(bb, bounds);
        return SpaceTraversalMatching(raydata, bb, bounds, maxmatchesperray, mincameras, maxdistance,
                                      multiplematchesperraymindistance, timing_out);
    }

    const int B = static_cast<int>(boxes.size());
    if (B != n_boxes && verbose) {
        std::cout << "[spatial-boxes] note: built " << B << " sub-boxes for factorization of " << n_boxes << ".\n";
    }

    std::chrono::steady_clock::time_point t_match_wall{};
    if (timing_out) {
        t_match_wall = std::chrono::steady_clock::now();
    }

    std::vector<std::vector<ray>> rays_per_box(static_cast<size_t>(B));
    if (raydata.size() >= 256) {
        #pragma omp parallel
        {
            std::vector<std::vector<ray>> local(static_cast<size_t>(B));
            for (int bb = 0; bb < B; ++bb) {
                local[static_cast<size_t>(bb)].reserve(raydata.size() / static_cast<size_t>(B) + 8);
            }
            #pragma omp for schedule(static) nowait
            for (size_t ri = 0; ri < raydata.size(); ++ri) {
                const ray& r = raydata[ri];
                for (int b = 0; b < B; ++b) {
                    const boundingboxspec& box = boxes[static_cast<size_t>(b)];
                    transformedray tr = PrepareRay(r, box);
                    if (tr.hit) {
                        local[static_cast<size_t>(b)].push_back(r);
                    }
                }
            }
            #pragma omp critical(spatial_merge_rays)
            {
                for (int b = 0; b < B; ++b) {
                    std::vector<ray>& dst = rays_per_box[static_cast<size_t>(b)];
                    std::vector<ray>& src = local[static_cast<size_t>(b)];
                    if (!src.empty()) {
                        dst.insert(dst.end(), src.begin(), src.end());
                    }
                }
            }
        }
    } else {
        for (size_t ri = 0; ri < raydata.size(); ++ri) {
            const ray& r = raydata[ri];
            for (int b = 0; b < B; ++b) {
                const boundingboxspec& box = boxes[static_cast<size_t>(b)];
                transformedray tr = PrepareRay(r, box);
                if (tr.hit) {
                    rays_per_box[static_cast<size_t>(b)].push_back(r);
                }
            }
        }
    }

    std::vector<std::vector<candidatematch>> per_box(static_cast<size_t>(B));
    std::vector<STMFrameTiming> per_tim(static_cast<size_t>(B));

    const int maxt = omp_get_max_threads();
    const int n_outer = std::min(B, maxt);
    const int threads_per_box = std::max(1, maxt / n_outer);

    const int saved_levels = omp_get_max_active_levels();
    omp_set_max_active_levels(std::max(2, saved_levels));

    #pragma omp parallel for schedule(static) num_threads(n_outer)
    for (int b = 0; b < B; ++b) {
        omp_set_num_threads(threads_per_box);
        if (rays_per_box[static_cast<size_t>(b)].empty()) {
            omp_set_num_threads(maxt);
            continue;
        }
        std::vector<std::vector<double>> bounds_b;
        make_bounds_from_bb(boxes[static_cast<size_t>(b)], bounds_b);
        STMFrameTiming* tptr = timing_out ? &per_tim[static_cast<size_t>(b)] : nullptr;
        per_box[static_cast<size_t>(b)] = SpaceTraversalMatchingCandidatesOnly(
            rays_per_box[static_cast<size_t>(b)], boxes[static_cast<size_t>(b)], bounds_b, mincameras, tptr, false);
        omp_set_num_threads(maxt);
    }

    omp_set_max_active_levels(saved_levels);

    if (timing_out) {
        for (int b = 0; b < B; ++b) {
            timing_out->merge_max_parallel_box_stages(per_tim[static_cast<size_t>(b)]);
        }
    }

    size_t total_cands = 0;
    for (int b = 0; b < B; ++b) {
        total_cands += per_box[static_cast<size_t>(b)].size();
    }

    std::vector<candidatematch> merged;
    merged.reserve(total_cands);
    for (int b = 0; b < B; ++b) {
        std::vector<candidatematch>& v = per_box[static_cast<size_t>(b)];
        merged.insert(merged.end(), v.begin(), v.end());
    }

    std::vector<candidatematch> deduped = dedupe_matches_by_camrayids(std::move(merged));
    std::sort(deduped.begin(), deduped.end(), comparecandidatematches);

    std::vector<candidatematch> out = SelectApprovedMatchesFromSortedCandidates(
        deduped, maxmatchesperray, maxdistance, multiplematchesperraymindistance, timing_out, verbose);

    if (timing_out) {
        timing_out->matching_total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_match_wall).count();
    }

    if (verbose) {
        std::cout << "[spatial-boxes] OMP_MAX=" << maxt << " concurrent_boxes=" << n_outer
                  << " threads_per_box=" << threads_per_box << "; " << B << " boxes, overlap_cells="
                  << std::max(0, overlap_cells) << ", " << raydata.size() << " rays, " << total_cands
                  << " candidates (sum over boxes), " << deduped.size() << " after merge/dedupe, " << out.size()
                  << " approved.\n";
    }

    return out;
}
