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
    hip_euler2d::Primitive *hydro_data,
    double time,
    int nx,
    int ny
)
{
  double rho[nx * ny];
  double ve1[nx * ny];
  double ve2[nx * ny];
  double pre[nx * ny];
  
  // Create temporary buffer for each con
  for (int jj = 0; jj < ny; jj++)
  {
    for (int ii = 0; ii < nx; ii++){

        rho[jj*nx + ii] =   hydro_data[jj*nx + ii].rho;
        ve1[jj*nx + ii] =   hydro_data[jj*nx + ii].v1;
        ve2[jj*nx + ii] =   hydro_data[jj*nx + ii].v2;
        pre[jj*nx + ii] =   hydro_data[jj*nx + ii].p;
    }
  }

  
  // preparation of a dataset and a file.
  hsize_t dim[1];
  dim[0]   = nx*ny;                  
  int rank = 1;

  H5::DataSpace dataspace(rank, dim);

  H5:IntType datatype(H5::PredType::NATIVE_DOUBLE);

  H5::H5File  file(filename, H5F_ACC_TRUNC);
  H5::DataSet dataset(file.createDataSet("rho", datatype, dataspace));
  dataset.write(rho, H5::PredType::NATIVE_DOUBLE);
  dataset.close();
  dataset = file.createDataSet("v1", datatype, dataspace);
  dataset.write(ve1, H5::PredType::NATIVE_DOUBLE);
  dataset.close();
  dataset = file.createDataSet("v2", datatype, dataspace);
  dataset.write(ve2, H5::PredType::NATIVE_DOUBLE);
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