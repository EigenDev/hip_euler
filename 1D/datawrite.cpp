/***
 * Implementation of the data writing
 * function
 */

#include "datawrite.hpp"
#include "H5Cpp.h"
#include <iostream>

using namespace H5;

void write_file_h5(
    const std::string filename,
    hip_euler::Primitive *hydro_data,
    double time,
    int nzones
)
{
  double rho[nzones];
  double vel[nzones];
  double pre[nzones];

  // Create temporary buffer for each con
  for (int ii = 0; ii < nzones; ii++){
      rho[ii] =   hydro_data[ii].rho;
      vel[ii] =   hydro_data[ii].v;
      pre[ii] =   hydro_data[ii].p;

  }

  
  // preparation of a dataset and a file.
  hsize_t dim[1];
  dim[0]   = nzones;                  
  int rank = 1;

  H5::DataSpace dataspace(rank, dim);

  H5:IntType datatype(H5::PredType::NATIVE_DOUBLE);

  H5::H5File  file(filename, H5F_ACC_TRUNC);
  H5::DataSet dataset(file.createDataSet("rho", datatype, dataspace));
  dataset.write(rho, H5::PredType::NATIVE_DOUBLE);
  dataset.close();
  dataset = file.createDataSet("vel", datatype, dataspace);
  dataset.write(vel, H5::PredType::NATIVE_DOUBLE);
  dataset.close();
  dataset = file.createDataSet("pre", datatype, dataspace);
  dataset.write(pre, H5::PredType::NATIVE_DOUBLE);
  dataset.close();

}

// int main(){
//   int nz = 40;
//   hip_euler::Primitive *pol = (hip_euler::Primitive*)malloc(nz * sizeof(hip_euler::Primitive));
//   for (size_t ii = 0; ii < nz; ii++)
//   {
//     pol[ii] =  hip_euler::Primitive{1.0, 0.0, 0.1};
//   }
  
//   write_file_h5("test.h5", pol, 0.0, nz);
//   free(pol);
//   return 0;
// }