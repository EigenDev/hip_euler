/**
 * The main hip execution
 * that runs runs if there
 * i correct Kernel linkage
 * in the Euler component file
*/

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "datawrite.hpp"

#include "hip_euler_1d.hpp"

using namespace hip_euler;
using namespace std::chrono;
static constexpr int BLOCK_SIZE = 32;

int main(int argc, char ** argv)
{
    const int nzones = 1 << 16;
    double xmin = 0.0;
    double xmax = 1.0;
    double dx   = (xmax - xmin)/nzones;
    
    hip_euler::SimState sim_state;
    // create class storage on device and copy top level class
    hip_euler::SimState *d_sim;

    // Initialize the system on host
    sim_state.initializeModel(xmin, xmax, nzones);

    // Copy host class instance to device
    hipMalloc((void**)&d_sim,    sizeof(hip_euler::SimState));
    hipMemcpy(d_sim, &sim_state, sizeof(hip_euler::SimState), hipMemcpyHostToDevice);

    SimDualSpace dualMem;
    dualMem.copyStateToGPU(sim_state, d_sim);

    // Setup the system
    int nBlocks = (nzones + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double tend = 0.1;
    double dt = dx * 0.01;

    // Evolve it 
    evolve(d_sim, nBlocks, BLOCK_SIZE, nzones, tend, dt);

    dualMem.copyGPUStateToHost(d_sim, sim_state);

    // Write it to a file
    write_file_h5("sod_test.h5", sim_state.prims, tend, sim_state.nzones);

    // Clean up the memory space
    dualMem.cleanUp();


    hipError_t error = hipGetLastError();

    if (error !=0 ){
        printf("%s", hipGetErrorString(error));
    }

    hipFree(d_sim);

    return 0;
}

