/***
 * Write input std::vector to an HDF5
 * binary file
 */

#ifndef DATAWRITE_H
#define DATAWRITE_H

#include <string>
#include "hip_euler_1d.hpp"

// Writes the input data into an HDF5 file 
void write_file_h5(
    std::string filename,
    hip_euler::Primitive *hydro_data,
    double time,
    int nzones);

#endif