/*
 * This file defines the STM_File class which represents a HDF5 file storing the
 * matches found by STM.
 *
 * Implementation file
 *
 */

#include <iostream>
#include <sys/stat.h>
#include "STM_hdf5.h"

namespace {
bool file_exists_regular(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
}  // namespace

STM_File::STM_File() {
    opened = false;
    f = nullptr;
    readonly_mode = true;
}

STM_File::STM_File(std::string filename) {
    opened = false;
    f = nullptr;
    readonly_mode = true;
    open(filename);
}

STM_File::STM_File(std::string filename, const unsigned int maxframes, const unsigned int mincameras, const double maxdistance, const unsigned int maxmatchesperray, const unsigned int nx, const unsigned int ny, const unsigned int nz) {
    opened = false;
    f = nullptr;
    readonly_mode = false;
    open(filename, maxframes, mincameras, maxdistance, maxmatchesperray, nx, ny, nz);
}

void STM_File::open(std::string filename) {
    /*
     * Constructor method
     * Open STM_file in readonly mode
     *
     */

    if (opened) {
        std::cerr << "Warning: STM HDF5 file was already open!" << std::endl;
    }

    f = new H5::H5File(H5std_string(filename), H5F_ACC_RDONLY);
    readonly_mode = true;
    std::cout << "File '" << filename << "' opened in readonly mode." << std::endl;

    // Read the attributes from the HDF5 file
    H5::Attribute attr_maxframes = f->openAttribute(H5std_string("maxframes"));
    attr_maxframes.read(H5::PredType::NATIVE_INT, &maxframes);
    std::cout << "maxframes = " << maxframes << std::endl;

    H5::Attribute attr_mincameras = f->openAttribute(H5std_string("mincameras"));
    attr_mincameras.read(H5::PredType::NATIVE_INT, &mincameras);
    std::cout << "mincameras = " << mincameras << std::endl;

    H5::Attribute attr_maxdistance = f->openAttribute(H5std_string("maxdistance"));
    attr_maxdistance.read(H5::PredType::NATIVE_DOUBLE, &maxdistance);
    std::cout << "maxdistance = " << maxdistance << std::endl;

    H5::Attribute attr_maxmatchesperray = f->openAttribute(H5std_string("maxmatchesperray"));
    attr_maxmatchesperray.read(H5::PredType::NATIVE_INT, &maxmatchesperray);
    std::cout << "maxmatchesperray = " << maxmatchesperray << std::endl;

    H5::Attribute attr_nx = f->openAttribute(H5std_string("nx"));
    attr_nx.read(H5::PredType::NATIVE_INT, &nx);
    std::cout << "nx = " << nx << std::endl;

    H5::Attribute attr_ny = f->openAttribute(H5std_string("ny"));
    attr_ny.read(H5::PredType::NATIVE_INT, &ny);
    std::cout << "ny = " << ny << std::endl;

    H5::Attribute attr_nz = f->openAttribute(H5std_string("nz"));
    attr_nz.read(H5::PredType::NATIVE_INT, &nz);
    std::cout << "nz = " << nz << std::endl;

    opened = true;
}

void STM_File::open(std::string filename, const unsigned int maxframes, const unsigned int mincameras, const double maxdistance, const unsigned int maxmatchesperray, const unsigned int nx, const unsigned int ny, const unsigned int nz) {
    /*
     * Constructor method
     * Open STM_file in write mode
     *
     */

    if (opened) {
        std::cerr << "Warning: STM HDF5 file was already open!" << std::endl;
    }

    f = new H5::H5File(H5std_string(filename), H5F_ACC_TRUNC);
    readonly_mode = false;
    std::cout << "File '" << filename << "' opened in write mode." << std::endl;

    this->maxframes = maxframes;
    this->mincameras = mincameras;
    this->maxdistance = maxdistance;
    this->maxmatchesperray = maxmatchesperray;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;

    // Write to HDF5 file
    const unsigned int RANK = 1;
    hsize_t dim_vector[RANK] = {1};
    H5::DataSpace attr_dataspace = H5::DataSpace(RANK, dim_vector);

    H5::Attribute attr_maxframes = f->createAttribute(H5std_string("maxframes"),
                                                      H5::PredType::NATIVE_INT,
                                                      attr_dataspace);
    attr_maxframes.write(H5::PredType::NATIVE_INT, &maxframes);

    H5::Attribute attr_mincameras = f->createAttribute(H5std_string("mincameras"),
                                                      H5::PredType::NATIVE_INT,
                                                      attr_dataspace);
    attr_mincameras.write(H5::PredType::NATIVE_INT,&mincameras);

    H5::Attribute attr_maxdistance = f->createAttribute(H5std_string("maxdistance"),
                                                      H5::PredType::NATIVE_DOUBLE,
                                                      attr_dataspace);
    attr_maxdistance.write(H5::PredType::NATIVE_DOUBLE, &maxdistance);

    H5::Attribute attr_maxmatchesperray = f->createAttribute(H5std_string("maxmatchesperray"),
                                                      H5::PredType::NATIVE_INT,
                                                      attr_dataspace);
    attr_maxmatchesperray.write(H5::PredType::NATIVE_INT, &maxmatchesperray);

    H5::Attribute attr_nx = f->createAttribute(H5std_string("nx"),
                                                      H5::PredType::NATIVE_INT,
                                                      attr_dataspace);
    attr_nx.write(H5::PredType::NATIVE_INT, &nx);

    H5::Attribute attr_ny = f->createAttribute(H5std_string("ny"),
                                                      H5::PredType::NATIVE_INT,
                                                      attr_dataspace);
    attr_ny.write(H5::PredType::NATIVE_INT, &ny);

    H5::Attribute attr_nz = f->createAttribute(H5std_string("nz"),
                                                      H5::PredType::NATIVE_INT,
                                                      attr_dataspace);
    attr_nz.write(H5::PredType::NATIVE_INT, &nz);

    opened = true;
}

// Destructor method
STM_File::~STM_File() {
    if (!opened || f == nullptr) {
        return;
    }
    f->flush(H5F_SCOPE_GLOBAL);
    f->close();
    delete f;
    f = nullptr;
    opened = false;
}

std::string STM_File::frame_group_path(unsigned int frameno) {
    std::string s = std::to_string(frameno);
    if (s.length() < 5) {
        s.insert(0, 5 - s.length(), '0');
    }
    return "/frame" + s;
}

bool STM_File::frame_is_complete(unsigned int frameno) const {
    if (!opened || f == nullptr) {
        return false;
    }
    const std::string gpath = frame_group_path(frameno);
    try {
        if (!f->nameExists(gpath)) {
            return false;
        }
        H5::Group g = f->openGroup(H5std_string(gpath));
        return g.nameExists("xyze") && g.nameExists("camrayids");
    } catch (const H5::Exception&) {
        return false;
    }
}

unsigned int STM_File::count_complete_frames(unsigned int nframes) const {
    unsigned int n = 0;
    for (unsigned int i = 0; i < nframes; ++i) {
        if (frame_is_complete(i)) {
            ++n;
        }
    }
    return n;
}

void STM_File::open_or_resume(std::string filename, const unsigned int maxframes, const unsigned int mincameras, const double maxdistance, const unsigned int maxmatchesperray, const unsigned int nx, const unsigned int ny, const unsigned int nz, bool resume) {
    if (resume && file_exists_regular(filename)) {
        if (opened) {
            std::cerr << "Warning: STM HDF5 file was already open!" << std::endl;
        }
        f = new H5::H5File(H5std_string(filename), H5F_ACC_RDWR);
        readonly_mode = false;
        opened = true;
        std::cout << "File '" << filename << "' opened in resume (read/write) mode." << std::endl;

        // Keep existing attributes when present; refresh maxframes to this run's target.
        try {
            H5::Attribute attr_mincameras = f->openAttribute(H5std_string("mincameras"));
            attr_mincameras.read(H5::PredType::NATIVE_INT, &this->mincameras);
            H5::Attribute attr_maxdistance = f->openAttribute(H5std_string("maxdistance"));
            attr_maxdistance.read(H5::PredType::NATIVE_DOUBLE, &this->maxdistance);
            H5::Attribute attr_maxmatchesperray = f->openAttribute(H5std_string("maxmatchesperray"));
            attr_maxmatchesperray.read(H5::PredType::NATIVE_INT, &this->maxmatchesperray);
            H5::Attribute attr_nx = f->openAttribute(H5std_string("nx"));
            attr_nx.read(H5::PredType::NATIVE_INT, &this->nx);
            H5::Attribute attr_ny = f->openAttribute(H5std_string("ny"));
            attr_ny.read(H5::PredType::NATIVE_INT, &this->ny);
            H5::Attribute attr_nz = f->openAttribute(H5std_string("nz"));
            attr_nz.read(H5::PredType::NATIVE_INT, &this->nz);
            if (this->mincameras != mincameras || this->maxdistance != maxdistance
                || this->maxmatchesperray != maxmatchesperray
                || this->nx != nx || this->ny != ny || this->nz != nz) {
                std::cerr << "Warning: resume file attributes differ from current STM parameters "
                          << "(mincameras/maxdistance/maxmatchesperray/nx/ny/nz). "
                          << "Skipping existing frames anyway; new frames use the current parameters.\n";
            }
        } catch (const H5::Exception& e) {
            std::cerr << "Warning: could not read some attributes from resume file: " << e.getDetailMsg() << std::endl;
            this->mincameras = mincameras;
            this->maxdistance = maxdistance;
            this->maxmatchesperray = maxmatchesperray;
            this->nx = nx;
            this->ny = ny;
            this->nz = nz;
        }
        this->maxframes = maxframes;
        set_last_frame(maxframes);
        const unsigned int already = count_complete_frames(maxframes);
        std::cout << "Resume: found " << already << " complete frame group(s) to skip (of " << maxframes << ")." << std::endl;
        return;
    }
    if (resume && !file_exists_regular(filename)) {
        std::cout << "Resume requested but '" << filename << "' does not exist; creating a new file.\n";
    }
    open(filename, maxframes, mincameras, maxdistance, maxmatchesperray, nx, ny, nz);
}

// Methods

void STM_File::write_matches(const unsigned int frameno, std::vector<candidatematch> &matches, bool write_cylinder_aux) {
    /*
     * Write matches to HDF5 file (one bulk write per dataset).
     *
     * Layout is (rows, cols) = (components, N_matches) in C (row-major) order, matching
     * the previous per-column hyperslab writes: buf[comp * N + match].
     */

    const hsize_t nmatches = static_cast<hsize_t>(matches.size());

    // Find the largest number of camrayids for this frame
    unsigned int maxcams = 0;
    for (const auto& match : matches) {
        if (match.camrayids.size() > maxcams) {
            maxcams = static_cast<unsigned int>(match.camrayids.size());
        }
    }
    // Empty frames still need a group with zero-width datasets so resume can skip them.
    if (maxcams == 0) {
        maxcams = 1;
    }

    const unsigned int RANK = 2;
    hsize_t xyze_dim_vector[RANK] = {4, nmatches};
    H5::DataSpace xyze_space(RANK, xyze_dim_vector);
    hsize_t camrayids_dim_vector[RANK] = {2 * maxcams, nmatches};
    H5::DataSpace camrayids_space(RANK, camrayids_dim_vector);

    // Chunk + light compression when large enough. Deflate 9 was a major time sink.
    H5::DSetCreatPropList xyze_plist;
    H5::DSetCreatPropList camrayids_plist;
    if (nmatches > 1) {
        hsize_t chunk_n = nmatches;
        if (chunk_n > 65536) {
            chunk_n = 65536;
        }
        hsize_t xyze_chunksize[RANK] = {4, chunk_n};
        xyze_plist.setChunk(RANK, xyze_chunksize);
        xyze_plist.setDeflate(1);

        hsize_t camrayids_chunksize[RANK] = {2 * maxcams, chunk_n};
        camrayids_plist.setChunk(RANK, camrayids_chunksize);
        camrayids_plist.setDeflate(1);
    }

    const std::string gpath = frame_group_path(frameno);
    if (f->nameExists(gpath)) {
        f->unlink(gpath);
    }
    H5::Group frameno_group = f->createGroup(H5std_string(gpath));
    H5::DataSet xyze_dset = frameno_group.createDataSet(
        H5std_string("xyze"), H5::PredType::NATIVE_DOUBLE, xyze_space, xyze_plist);
    H5::DataSet camrayids_dset = frameno_group.createDataSet(
        H5std_string("camrayids"), H5::PredType::NATIVE_LONG, camrayids_space, camrayids_plist);

    if (nmatches == 0) {
        return;
    }

    std::vector<double> xyze_buf(static_cast<size_t>(4 * nmatches));
    std::vector<long> camrayids_buf(static_cast<size_t>(2 * maxcams * nmatches), -1);
    for (hsize_t row = 0; row < nmatches; ++row) {
        const candidatematch& m = matches[static_cast<size_t>(row)];
        xyze_buf[0 * nmatches + row] = m.matchx;
        xyze_buf[1 * nmatches + row] = m.matchy;
        xyze_buf[2 * nmatches + row] = m.matchz;
        xyze_buf[3 * nmatches + row] = m.matcherror;
        for (unsigned int idno = 0; idno < maxcams; ++idno) {
            if (idno < m.camrayids.size()) {
                camrayids_buf[(2 * idno) * nmatches + row] = m.camrayids[idno].camid;
                camrayids_buf[(2 * idno + 1) * nmatches + row] = m.camrayids[idno].rayid;
            }
        }
    }

    xyze_dset.write(xyze_buf.data(), H5::PredType::NATIVE_DOUBLE);
    camrayids_dset.write(camrayids_buf.data(), H5::PredType::NATIVE_LONG);

    if (write_cylinder_aux) {
        const hsize_t CYR = 2;
        hsize_t cy_dim[2] = {CYR, nmatches};
        H5::DataSpace cy_space(2, cy_dim);
        std::vector<double> cy_buf(static_cast<size_t>(CYR * nmatches));
        for (hsize_t row = 0; row < nmatches; ++row) {
            cy_buf[row] = matches[static_cast<size_t>(row)].cylinder_pair_depth_sum;
            cy_buf[nmatches + row] = matches[static_cast<size_t>(row)].cylinder_min_axis_margin;
        }
        H5::DSetCreatPropList cy_plist;
        if (nmatches > 1) {
            hsize_t chunk_n = nmatches > 65536 ? 65536 : nmatches;
            hsize_t cy_chunk[2] = {CYR, chunk_n};
            cy_plist.setChunk(2, cy_chunk);
            cy_plist.setDeflate(1);
        }
        frameno_group.createDataSet(H5std_string("cylinder_aux"), H5::PredType::NATIVE_DOUBLE, cy_space, cy_plist)
            .write(cy_buf.data(), H5::PredType::NATIVE_DOUBLE);
    }
}

std::vector<candidatematch> STM_File::read_matches(const unsigned int frameno) {
    // Open datasets
    std::string xyze_name = "/frame";
    xyze_name.append(std::to_string(frameno));
    xyze_name.append("_xyze");
    H5::DataSet xyze_dset = f->openDataSet(H5std_string(xyze_name));

    std::string camrayids_name = "/frame";
    camrayids_name.append(std::to_string(frameno));
    camrayids_name.append("_camrayids");
    H5::DataSet camrayids_dset = f->openDataSet(H5std_string(camrayids_name));

    // Dataset sizes
    const unsigned int RANK = 2;
    H5::DataSpace xyze_space = xyze_dset.getSpace();
    if (xyze_space.getSimpleExtentNdims() != RANK) {
        std::cerr << "Wrong RANK in dataset!" << std::endl;
        exit(EXIT_FAILURE);
    }
    hsize_t xyze_dim_vector[RANK];
    xyze_space.getSimpleExtentDims(xyze_dim_vector, NULL);
    if (xyze_dim_vector[0] != 4) {
        std::cerr << "Wrong xyze dataset dimension" << std::endl;
        exit(EXIT_FAILURE);
    }
    // std::cout << "number of rows in xyze = " << xyze_dim_vector[1] << std::endl;

    H5::DataSpace camrayids_space = camrayids_dset.getSpace();
    if (camrayids_space.getSimpleExtentNdims() != RANK) {
        std::cerr << "Wrong RANK in dataset!" << std::endl;
        exit(EXIT_FAILURE);
    }
    hsize_t camrayids_dim_vector[RANK];
    camrayids_space.getSimpleExtentDims(camrayids_dim_vector, NULL);
    // std::cout << "number of rows in camrayids = " << camrayids_dim_vector[1] << std::endl;
    unsigned int maxcams = camrayids_dim_vector[0]/2;
    if (xyze_dim_vector[1] != camrayids_dim_vector[1]) {
        std::cerr << "xyze camrayids dimension mismatch" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read data
    const unsigned int nrows = xyze_dim_vector[1];
    std::vector<candidatematch> matches(nrows);
    hsize_t row_xyze_dim_vector[RANK] = {4, 1};
    H5::DataSpace row_xyze_space(RANK, row_xyze_dim_vector);
    hsize_t row_camrayids_dim_vector[RANK] = {2*maxcams, 1};
    H5::DataSpace row_camrayids_space(RANK, row_camrayids_dim_vector);

    for(unsigned int row = 0; row < nrows; row++) {
        double xyze_data[4];
        long camrayids_data[2*maxcams];

        hsize_t offset[2] = {0, row};
        H5::DataSpace xyze_subspace = xyze_dset.getSpace();
        xyze_subspace.selectHyperslab(H5S_SELECT_SET, row_xyze_dim_vector, offset);
        xyze_dset.read(xyze_data, H5::PredType::NATIVE_DOUBLE, row_xyze_space, xyze_subspace);

        H5::DataSpace camrayids_subspace = camrayids_dset.getSpace();
        camrayids_subspace.selectHyperslab(H5S_SELECT_SET, row_camrayids_dim_vector, offset);
        camrayids_dset.read(camrayids_data, H5::PredType::NATIVE_LONG, row_camrayids_space, camrayids_subspace);
        matches[row].matchx = xyze_data[0];
        matches[row].matchy = xyze_data[1];
        matches[row].matchz = xyze_data[2];
        matches[row].matcherror = xyze_data[3];

        for(unsigned int idno = 0; idno < maxcams; idno++) {
            struct camrayid camrayid;
            camrayid.camid = camrayids_data[2*idno];
            camrayid.rayid = camrayids_data[2*idno+1];
            if (camrayid.camid != -1 && camrayid.rayid != -1) {
                matches[row].camrayids.push_back(camrayid);
            }
        }
    }

    return matches;
}


void STM_File::flush() {
    if (opened && !readonly_mode) {
        f->flush(H5F_SCOPE_GLOBAL);
    }
}

void STM_File::set_last_frame(const unsigned int maxframes) {
    /*
     * Override maxframes attribute
     *
     */

    this->maxframes = maxframes;
    if (opened && !readonly_mode) {
        H5::Attribute attr_maxframes = f->openAttribute(H5std_string("maxframes"));
        attr_maxframes.write(H5::PredType::NATIVE_INT, &maxframes);
    }
}
