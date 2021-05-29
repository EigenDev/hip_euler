/***
 * The implementation zone 
 * for the hydro euler HIP variant
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "hip_euler_1d.hpp"
#include <math.h>


using namespace hip_euler;
using namespace std::chrono;

SimState::SimState(
    const int nzones, 
    const double xmin,
    const double xmax) 
    : 
    nzones(nzones),
    xmin(xmin),
    xmax(xmax)
    {

    }

SimState::SimState(){}


/** Initialize the model based on a Sod setup
 * x0 = left boundary
 * x1 = right boundary
 * numzones = number of zones
 * prims  = initial prim data
*/
void SimState::initializeModel(
    const double x0, 
    const double x1, 
    const int numzones
)
{
    double ds;
    dt     = 1e-4;
    xmin   = x0;
    xmax   = x1;
    nzones = numzones;
    dx     = (xmax - xmin) / nzones;

    // Allocate array of conservatives
    sys_state = (Conserved*) malloc(numzones * sizeof(Conserved));
    u1        = (Conserved*) malloc(numzones * sizeof(Conserved));
    du_dt     = (Conserved*) malloc(numzones * sizeof(Conserved));
    prims     = (Primitive*) malloc(numzones * sizeof(Primitive));


    for (int ii = 0; ii < nzones; ii++){
        ds = ii * dx;
        if (ds < 0.5 * (xmax - xmin)){
            prims[ii] = Primitive{1.0, 0.0, 1.0  };
        } else {
            prims[ii] = Primitive{0.125, 0.0, 0.1};
        }
    }

    // Construct the initial conservatives 
    prims2cons(prims);
}

SimState::~SimState(){
    free(sys_state);
    free(u1);
    free(du_dt);
    free(prims);
};

// Convert from the conservarive array to primitive and cache it
GPU_CALLABLE_MEMBER void SimState::cons2prim(const Conserved *u)
{
    for(int ii = 0; ii < nzones; ii++){
        double v = u[ii].m/u[ii].rho;

        double p = 
            (ADIABATIC_GAMMA - 1.0) * (u[ii].energy - 0.5 * u[ii].rho * v * v);

        prims[ii] = Primitive{u[ii].rho, v, p};

    }
    
}//-----End cons2prims



GPU_CALLABLE_MEMBER Conserved SimState::prims2cons(const Primitive &prims)
{
    double m = prims.rho * prims.v;
    double e = 
        prims.p/(ADIABATIC_GAMMA - 1.0) + 0.5 * prims.v * prims.v * prims.rho;

    return Conserved{prims.rho, m, e};

}//-----End prims2cons for single primitive struct

GPU_CALLABLE_MEMBER void SimState::prims2cons(const Primitive *prims)
{
    for (int ii = 0; ii < nzones; ii++){
        double mom = prims[ii].rho * prims[ii].v;
        double e = 
            prims[ii].p/(ADIABATIC_GAMMA - 1.0) + 0.5 * prims[ii].v * prims[ii].v * prims[ii].rho;

        sys_state[ii] = Conserved{prims[ii].rho, mom, e};
    }
    

}//-----End prims2cons for array of Primitive structs




GPU_CALLABLE_MEMBER EigenWave SimState::calc_waves(
    const Primitive &left_prims, 
    const Primitive &right_prims)
{
    double rhol = left_prims.rho;
    double vl   = left_prims.v  ;
    double pl   = left_prims.p  ;

    double rhor = right_prims.rho;
    double vr   = right_prims.v  ;
    double pr   = right_prims.p  ;

    double csl  = sqrt(ADIABATIC_GAMMA * pl / rhol);
    double csr  = sqrt(ADIABATIC_GAMMA * pr/  rhor);

    double aL   = min(vl - csl, vr - csr);
    double aR   = max(vr + csr, vl + csl);

    return EigenWave(aL, aR);

}//-------End calc_waves



GPU_CALLABLE_MEMBER Conserved SimState::calc_hll_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims)
{
    const EigenWave lambdas = calc_waves(left_prims, right_prims);
    const double aRp     = max(0.0, lambdas.aR);
    const double aLm     = min(0.0, lambdas.aL);

    return 
        (left_flux * aRp - right_flux * aLm + 
            (right_state - left_state) * aRp * aLm  ) / (aRp - aLm);

}// End HLL_FLUX


GPU_CALLABLE_MEMBER Conserved SimState::prims2flux(const Primitive &prims)
{
    double e = 
        prims.p/(ADIABATIC_GAMMA - 1.0) + 0.5 * prims.v * prims.v * prims.rho;
    
    double rhof = prims.rho * prims.v;
    double momf = prims.rho * prims.v * prims.v + prims.p;
    double engf = (e + prims.p)*prims.v;

    return Conserved{rhof, momf, engf};
    
}// End prims2flux


__global__ void hip_euler::gpu_evolve(SimState * s, double dt)
{
    int ii = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    Conserved ul, ur, fl, fr, frf, flf;
    Primitive pl, pr;

    if (ii < s->nzones){
        // i-1/2 face
        ul = (ii > 0) ? s->sys_state[ii - 1]   : s->sys_state[ii];
        ur =                                     s->sys_state[ii];
        pl = (ii > 0) ? s->prims[ii - 1]       : s->prims[ii];
        pr =                                     s->prims[ii];                  

        fl = s->prims2flux(pl);
        fr = s->prims2flux(pr);
        flf = s->calc_hll_flux(ul, ur, fl, fr, pl, pr);

        // i+1/2 face
        ul =                                               s->sys_state[ii];
        ur = (ii < s->nzones - 1) ? s->sys_state[ii + 1] : s->sys_state[ii];
        pl =                                               s->prims[ii];
        pr = (ii < s->nzones - 1) ? s->prims    [ii + 1] : s->prims[ii];

        fl  = s->prims2flux(pl);
        fr  = s->prims2flux(pr);
        frf = s->calc_hll_flux(ul, ur, fl, fr, pl, pr); 

        s->sys_state[ii] -= (frf - flf) / s->dx * dt ;
    }

}

__global__ void hip_euler::gpu_cons2prim(SimState *s){
    int ii = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (ii < s->nzones){
        double v = s->sys_state[ii].m/s->sys_state[ii].rho;

        double p = 
            (ADIABATIC_GAMMA - 1.0) * (s->sys_state[ii].energy - 0.5 * s->sys_state[ii].rho * v * v);

        s->prims[ii] = Primitive{s->sys_state[ii].rho, v, p};

    }
}

void hip_euler::evolve(SimState *s, int nBlocks, int block_size, int nzones, double tend, double dt)
{
    double t = 0.0;
    high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> delta_t;
    int n = 1;
    while (t < tend)
    {
        t1 = high_resolution_clock::now();
        hipLaunchKernelGGL(gpu_evolve,    dim3(nBlocks), dim3(block_size), 0, 0, s, dt);
        hipLaunchKernelGGL(gpu_cons2prim, dim3(nBlocks), dim3(block_size), 0, 0, s);

        t2 = high_resolution_clock::now();
        delta_t = t2 - t1;
        std::cout << std::fixed << std::setprecision(3) << std::scientific;
            std::cout << "\r"
                << "Iteration: " << std::setw(5) << n 
                << "\t"
                << "Time: " << std::setw(10) <<  t
                << "\t"
                << "Zones/sec: "<< nzones / delta_t.count() << std::flush;

        t += dt;
        n++;
    }
    std::cout << "\n";

}

SimDualSpace::SimDualSpace(){}

SimDualSpace::~SimDualSpace(){}
void SimDualSpace::copyStateToGPU(
    const SimState &host,
    SimState *device
)
{
    int nz = host.nzones;
    

    //--------Allocate the memory for pointer objects-------------------------
    hipMalloc((void **)&host_u0,     nz * sizeof(Conserved));
    hipMalloc((void **)&host_u1,     nz * sizeof(Conserved));
    hipMalloc((void **)&host_dudt,   nz * sizeof(Conserved));
    hipMalloc((void **)&host_prims,  nz * sizeof(Primitive));
    // hipMalloc((void **)&u0        , sizeof(Conserved));

    //--------Copy the host resources to pointer variables on host
    hipMemcpy(host_u0,    host.sys_state, nz * sizeof(Conserved), hipMemcpyHostToDevice);
    hipMemcpy(host_u1,    host.u1       , nz * sizeof(Conserved), hipMemcpyHostToDevice);
    hipMemcpy(host_dudt,  host.du_dt    , nz * sizeof(Conserved), hipMemcpyHostToDevice);
    hipMemcpy(host_prims, host.prims    , nz * sizeof(Primitive), hipMemcpyHostToDevice);

    // copy pointer to allocated device storage to device class
    if ( hipMemcpy(&(device->sys_state), &host_u0,    sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_u0 -> device_sys_tate");
    };
    if( hipMemcpy(&(device->u1),        &host_u1,    sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_u1 -> device_u1");
    };

    if( hipMemcpy(&(device->du_dt),     &host_dudt,  sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_dudt -> device_du_dt");
    };

    if( hipMemcpy(&(device->prims),     &host_prims, sizeof(Primitive *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_prims -> device_prims");
    };

    hipMemcpy(&(device->dt),     &host.dt     ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->dx),     &host.dx     ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->xmin),   &host.xmin   ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->xmax),   &host.xmax   ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->nzones), &host.nzones  , sizeof(int), hipMemcpyHostToDevice);

    
    // Verify that things match up
    hipMemcpy(&host_dx,   &(device->dx),   sizeof(double *), hipMemcpyDeviceToHost);
    hipMemcpy(&host_dt,   &(device->dt),   sizeof(double *), hipMemcpyDeviceToHost);
    hipMemcpy(&host_xmin, &(device->xmin), sizeof(double *), hipMemcpyDeviceToHost);
    hipMemcpy(&host_xmax, &(device->xmax), sizeof(double *), hipMemcpyDeviceToHost);
    assert(host.dx == host_dx);
    assert(host.dt == host_dt);
    assert(host.xmin == host_xmin);
    assert(host.xmax == host_xmax);
    
}

void SimDualSpace::copyGPUStateToHost(
    const SimState *device,
    SimState &host
)
{
    int nz = host.nzones;

    hipMemcpy(host.sys_state, host_u0,        nz * sizeof(Conserved), hipMemcpyDeviceToHost);
    hipCheckErrors("Memcpy failed at transferring device conservatives to host");
    hipMemcpy(host.prims,     host_prims ,    nz * sizeof(Conserved), hipMemcpyDeviceToHost);
    hipCheckErrors("Memcpy failed at transferring device prims to host");
    
}

void SimDualSpace::cleanUp()
{
    printf("Freeing Device Memory...\n");
    hipFree(host_u0);
    hipFree(host_u1);
    hipFree(host_dudt);
    hipFree(host_prims);
    printf("Memory Freed.");
    
}

