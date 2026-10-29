#include "import_rays_h5.h"

#include <cmath>
#include <sstream>
#include <iomanip>
#include <array>
#include <iostream>

std::vector<ray> import_rays_h5(H5::H5File &file, const unsigned int frame) {
    std::vector<ray> rays;
    std::cout << "Importing rays from H5 File for frame " << frame << "...\n";
    
    // Open input HDF5 file
    
    hsize_t numObjs = file.getNumObjs();
    size_t datasetCount = 0;
    for (hsize_t i = 0; i < numObjs; i++) {
        H5::Group group = file.openGroup(file.getObjnameByIdx(i));
        H5::Group frame_data = group.openGroup(group.getObjnameByIdx(frame));
        H5::DataSet xyz = frame_data.openDataSet("xyz0");
        H5::DataSet dd = frame_data.openDataSet("dd");

        H5::DataSpace xyz_dataspace = xyz.getSpace();
        int rank = xyz_dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        xyz_dataspace.getSimpleExtentDims(dims.data(), nullptr); // dims[0]: N, dims[1]: 3

        std::vector<double> diameter_per_ray(dims[0], 0.0);
        try {
            H5::DataSet ds_d = frame_data.openDataSet("diameter");
            H5::DataSpace ds_d_space = ds_d.getSpace();
            int drank = ds_d_space.getSimpleExtentNdims();
            std::vector<hsize_t> ddims(static_cast<size_t>(drank));
            ds_d_space.getSimpleExtentDims(ddims.data(), nullptr);
            hsize_t dn = 1;
            for (int di = 0; di < drank; ++di) {
                dn *= ddims[static_cast<size_t>(di)];
            }
            if (dn == dims[0]) {
                std::vector<double> dbuf(dims[0]);
                ds_d.read(dbuf.data(), H5::PredType::NATIVE_DOUBLE);
                for (hsize_t j = 0; j < dims[0]; ++j) {
                    double v = dbuf[j];
                    diameter_per_ray[static_cast<size_t>(j)] = (std::isfinite(v) && v > 0.0) ? v : 0.0;
                }
            }
        } catch (const H5::Exception&) {
            // No diameter dataset: leave zeros (pure line rays).
        }

        // Allocate a flat buffer
        size_t totalSize = 1;
        for (auto d : dims) totalSize *= d;
        std::vector<double> read_buffer(totalSize);

        std::vector<std::array<float, 3>> xyz_data(dims[0]);
        xyz.read(xyz_data.data(), H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dd_dataspace = dd.getSpace();
        std::vector<std::array<float, 3>> dd_data(dims[0]);
        dd.read(dd_data.data(), H5::PredType::NATIVE_FLOAT);

        struct ray* buffer = new ray[dims[0]];
        for (hsize_t j = 0; j < dims[0]; j++) {
            buffer[j].x = xyz_data[j][0];
            buffer[j].y = xyz_data[j][1];
            buffer[j].z = xyz_data[j][2];
            buffer[j].vx = dd_data[j][0];
            buffer[j].vy = dd_data[j][1];
            buffer[j].vz = dd_data[j][2];
            buffer[j].rayid = j;
            buffer[j].camid = i;
            buffer[j].diameter = diameter_per_ray[static_cast<size_t>(j)];
        }
        for (hsize_t j = 0; j < dims[0]; j++) {
            rays.push_back(buffer[j]);
        }
        delete[] buffer;
    }
    return rays;
}