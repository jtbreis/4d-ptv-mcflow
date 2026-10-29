#include "STM_cylinder.h"
#include <cmath>

bool stm_cylinder_thicken_voxels = false;
bool stm_cylinder_rank_matches = false;
int stm_cylinder_max_extra_shells = 32;

double stm_line_line_distance(double p1x, double p1y, double p1z, double ux, double uy, double uz,
                              double p2x, double p2y, double p2z, double vx, double vy, double vz) {
    const double wx = p1x - p2x;
    const double wy = p1y - p2y;
    const double wz = p1z - p2z;
    const double cx = uy * vz - uz * vy;
    const double cy = uz * vx - ux * vz;
    const double cz = ux * vy - uy * vx;
    const double clen = std::sqrt(cx * cx + cy * cy + cz * cz);
    const double eps = 1e-12;
    if (clen > eps) {
        return std::fabs(wx * cx + wy * cy + wz * cz) / clen;
    }
    // Parallel (or nearly): distance from P2 to line through P1 along u.
    const double wdu = wx * ux + wy * uy + wz * uz;
    const double rx = wx - wdu * ux;
    const double ry = wy - wdu * uy;
    const double rz = wz - wdu * uz;
    return std::sqrt(rx * rx + ry * ry + rz * rz);
}
