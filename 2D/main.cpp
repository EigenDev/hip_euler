/**
 * The main hip execution
 * that runs runs if there
 * i correct Kernel linkage
 * in the Euler component file
*/

#include <stdio.h>
#include <iostream>
#include "datawrite.hpp"
#include "hip_euler_2d.hpp"


using namespace hip_euler2d;

static constexpr int BLOCK_SIZE = 16;

int main(int argc, char ** argv)
{
    const int nx = 256;
    const int ny = 256;
    double xmin = -1.0;
    double xmax =  1.0;
    double ymin = -1.0;
    double ymax =  1.0;
    double dx   = (xmax - xmin)/nx;
    double dy   = (ymax - ymin)/ny;
    
    hip_euler2d::SimState sim_state;
    // create class storage on device and copy top level class
    hip_euler2d::SimState *d_sim;

    // Initialize the system on host
    sim_state.initializeModel(xmin, xmax, ymin, ymax, nx, ny);

    // Copy host class instance to device
    hipMalloc((void**)&d_sim,    sizeof(hip_euler2d::SimState));
    hipMemcpy(d_sim, &sim_state, sizeof(hip_euler2d::SimState), hipMemcpyHostToDevice);

    SimDualSpace dualMem;
    dualMem.copyStateToGPU(sim_state, d_sim);

    // Setup the system
    int nxBlocks = (nx + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int nyBlocks = (ny + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double tend = 0.1;
    double dt = dx * 0.001;

    // Evolve it 
    evolve(d_sim, nxBlocks, nyBlocks, BLOCK_SIZE, nx*ny, tend, dt);

    dualMem.copyGPUStateToHost(d_sim, sim_state);

    // Write it to a file
    std::cout << sim_state.nx << ", " << sim_state.ny << "\n";
    // std::cout << "middle dude: " << sim_state.prims "\n";
    write_file_h5("sod_test2d.h5", sim_state.prims, tend, sim_state.nx, sim_state.ny);

    // Clean up the memory space
    dualMem.cleanUp();


    hipError_t error = hipGetLastError();

    if (error !=0 ){
        printf("%s", hipGetErrorString(error));
    }

    hipFree(d_sim);

    return 0;
}

