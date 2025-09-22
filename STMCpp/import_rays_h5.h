#ifndef IMPORT_RAYS_H5_H
#define IMPORT_RAYS_H5_H

#include <vector>
#include <string>
#include <H5Cpp.h>
#include "STM_types.h"

std::vector<ray> import_rays_h5(H5::H5File &file, const unsigned int frame);

#endif // IMPORT_RAYS_H5_H