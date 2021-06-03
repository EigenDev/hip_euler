/***
 * Write input std::vector to an HDF5
 * binary file
 */

#ifndef DATAWRITE_HPP
#define DATAWRITE_HPP

#include <string>
#include "hip_euler_2d.hpp"

// Writes the input data into an HDF5 file 
void write_file_h5(
    const std::string filename,
    hip_euler2d::Primitive *hydro_data,
    double time,
    int nx,
    int ny);

#endif