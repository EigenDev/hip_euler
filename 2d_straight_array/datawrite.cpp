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
    hip_euler2d::Primitive hydro_data,
    double time,
    int nx,
    int ny
)
{
  auto rho = new double[nx * ny];
  auto ve1 = new double[nx * ny];
  auto ve2 = new double[nx * ny];
  auto pre = new double[nx * ny];
  
  // Create temporary buffer for each con
  for (int jj = 0; jj < ny; jj++)
  {
    for (int ii = 0; ii < nx; ii++){

        rho[jj * nx + ii] = hydro_data[jj * nx + ii + 0];
        ve1[jj * nx + ii] = hydro_data[jj * nx + ii + 1];
        ve2[jj * nx + ii] = hydro_data[jj * nx + ii + 2];
        pre[jj * nx + ii] = hydro_data[jj * nx + ii + 3];
    }
  }
  
  // preparation of a dataset and a file.
  hsize_t dim[1];
  dim[0]   = nx*ny;                  
  int rank = 1;

  H5::DataSpace dataspace(rank, dim);

  H5::IntType double_type(H5::PredType::NATIVE_DOUBLE);
  H5::IntType int_type(H5::PredType::NATIVE_INT);

  H5::H5File  file(filename, H5F_ACC_TRUNC);
  H5::DataSet dataset(file.createDataSet("sim_info", double_type, dataspace));
  H5::DataSpace att_space(H5S_SCALAR);

  H5::Attribute att = dataset.createAttribute("nx", int_type, att_space );
  att.write(int_type, &nx);
  att = dataset.createAttribute("ny", int_type, att_space );
  att.write(int_type, &ny);

  dataset = file.createDataSet("rho", double_type, dataspace);
  dataset.write(rho, H5::PredType::NATIVE_DOUBLE);
  dataset.close();
  dataset = file.createDataSet("v1", double_type, dataspace);
  dataset.write(ve1, H5::PredType::NATIVE_DOUBLE);
  dataset.close();
  dataset = file.createDataSet("v2", double_type, dataspace);
  dataset.write(ve2, H5::PredType::NATIVE_DOUBLE);
  dataset.close();
  dataset = file.createDataSet("pre", double_type, dataspace);
  dataset.write(pre, H5::PredType::NATIVE_DOUBLE);
  dataset.close();

  delete[] rho;
  delete[] ve1;
  delete[] ve2;
  delete[] pre;
  

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