/***
 * The implementation zone 
 * for the hydro euler HIP variant
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "hip_euler_2d.hpp"
#include <math.h>


using namespace hip_euler2d;
using namespace std::chrono;

SimState::SimState(
    const int nx,
    const int ny, 
    const double xmin,
    const double xmax,
    const double ymin,
    const double ymax) 
    : 
    nx(nx),
    ny(ny),
    xmin(xmin),
    xmax(xmax),
    ymin(ymin),
    ymax(ymax)
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
    const double y0,
    const double y1,
    const int nxz,
    const int nyz
)
{
    double ds;
    dt     = 1e-4;
    xmin   = x0;
    xmax   = x1;
    ymin   = y0;
    ymax   = y1;
    nx     = nxz;
    ny     = nyz;
    dx     = (xmax - xmin) / nx;
    dy     = (ymax - ymin) / ny;

    double radius = sqrt( (xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin));
    // Allocate array of conservatives
    sys_state = (Conserved*) malloc(nx * ny * sizeof(Conserved));
    prims     = (Primitive*) malloc(nx * ny * sizeof(Primitive));

    for (int jj = 0; jj < ny; jj++)
    {
        for (int ii = 0; ii < nx; ii++){
            int gid = [&](){
                #ifdef __HIPCC__
                return ii * ny + jj;
                #else 
                return jj * nx + ii;
                #endif
            }();
            ds = sqrt( (ii * dx)*(ii * dx) + (jj * dy)*(jj * dy) ) ;
            if (ds < 0.5 * radius ){
                prims[gid] = Primitive{1.0, 0.0, 0.0, 1.0};
            } else {
                prims[gid] = Primitive{0.125, 0.0, 0.0, 0.1};
            }
        }
    }

    // Construct the initial conservatives 
    prims2cons(prims);
}

SimState::~SimState(){
    free(sys_state);
    free(prims);
};

GPU_CALLABLE_MEMBER void SimState::cons2prim(const Conserved &u)
{   
    const double rho = u.rho;
    const double v1  = u.m1 / rho;
    const double v2  = u.m2 / rho;
    const double p   = (ADIABATIC_GAMMA - 1.0) *  ( u.energy - 0.5 * rho * (v1 * v1 + v2 * v2));
    return Primitive {rho, v1, v2, p};
}
// Convert from the conservarive array to primitive and cache it
GPU_CALLABLE_MEMBER void SimState::cons2prim(const Conserved *u)
{   
    double v1, v2, p, rho;
    for (int jj = 0; jj < ny; jj++){

        for(int ii = 0; ii < nx; ii++){
            int gid = [&](){
                #ifdef __HIPCC__
                return ii * ny + jj;
                #else 
                return jj * nx + ii;
                #endif
            }();

            rho = u[gid].rho;
            v1 = u[gid].m1 / rho;
            v2 = u[gid].m2 / rho;
            p  = (ADIABATIC_GAMMA - 1.0) *  ( u[gid].energy - 0.5 * rho * (v1 * v1 + v2 * v2));
            prims[gid] = Primitive {rho, v1, v2, p};

        }
    }
}//-----End cons2prims



GPU_CALLABLE_MEMBER Conserved SimState::prims2cons(const Primitive &prims)
{
    const double rho = prims.rho;
    const double v1  = prims.v1;
    const double v2  = prims.v2;
    const double p   = prims.p;
    const double e   =  p  / (ADIABATIC_GAMMA - 1) + 0.5 * (v1 * v1 + v2 * v2) * rho;
    return Conserved{rho, rho * v1, rho * v2, e};
}//-----End prims2cons for single primitive struct

GPU_CALLABLE_MEMBER void SimState::prims2cons(const Primitive *prims)
{
    double m1, m2, v1, v2, e;
    for (int jj = 0; jj < ny; jj++)
    {
        for (int ii = 0; ii < nx; ii++){
            v1 = prims[jj*nx + ii].v1;
            v2 = prims[jj*nx + ii].v2;
            m1 = prims[jj*nx + ii].rho * v1;
            m2 = prims[jj*nx + ii].rho * v2;
            e = 
                prims[jj*nx + ii].p/(ADIABATIC_GAMMA - 1.0) + 0.5 * (v1 * v1 + v2 * v2) * prims[jj*nx + ii].rho;

            sys_state[jj*nx + ii] = Conserved{prims[jj*nx + ii].rho, m1, m2, e};
        }
    }
    

}//-----End prims2cons for array of Primitive structs




GPU_CALLABLE_MEMBER EigenWave SimState::calc_waves(
    const Primitive &left_prims, 
    const Primitive &right_prims,
    const int nhat)
{
    const double rhol = left_prims.rho;
    const double vl   = left_prims.vcomponent(nhat);
    const double pl   = left_prims.p;
    const double rhor = right_prims.rho;
    const double vr   = right_prims.vcomponent(nhat);
    const double pr   = right_prims.p;
    const double csl  = sqrt(ADIABATIC_GAMMA * pl / rhol);
    const double csr  = sqrt(ADIABATIC_GAMMA * pr/  rhor);
    const double aL   = min(vl - csl, vr - csr);
    const double aR   = max(vr + csr, vl + csl);
    return EigenWave{aL, aR};
}//-------End calc_waves



GPU_CALLABLE_MEMBER Conserved SimState::calc_hll_flux(
    const Conserved &left_state,
    const Conserved &right_state,
    const Conserved &left_flux,
    const Conserved &right_flux,
    const Primitive &left_prims,
    const Primitive &right_prims,
    const int nhat)
{
    const EigenWave lambdas = calc_waves(left_prims, right_prims, nhat);
    const double aRp     = max(0.0, lambdas.aR);
    const double aLm     = min(0.0, lambdas.aL);

    return 
        (left_flux * aRp - right_flux * aLm + 
            (right_state - left_state) * aRp * aLm  ) / (aRp - aLm);

}// End HLL_FLUX


GPU_CALLABLE_MEMBER Conserved SimState::prims2flux(const Primitive &prims, const int nhat)
{
    const double rho = prims.rho;
    const double v1  = prims.v1;
    const double v2  = prims.v2;
    const double p   = prims.p;
    const double e   = p / (ADIABATIC_GAMMA - 1.0) + 0.5 * (v1*v1 + v2*v2) * rho;

    switch (nhat)
    {
    case 1:
        {
            const double rhof = rho * v1;
            const double momf = rho * v1 * v1 + p;
            const double conv = rho * v1 * v2;
            const double engf = (e + p) * v1;
            return Conserved{rhof, momf, conv, engf};
        }
    case 2:
        {
            const double rhof = rho * v2;
            const double momf = rho * v2 * v2 + prims.p;
            const double conv = rho * v1 * v2;
            const double engf = (e + p) * v2;
            return Conserved{rhof, conv, momf, engf};
        }
    }
    
}// End prims2flux


__global__ void hip_euler2d::gpu_evolve(SimState * s, double dt)
{
    int jj  = blockDim.x * blockIdx.x + threadIdx.x;
    int ii  = blockDim.y * blockIdx.y + threadIdx.y;
    int ni  = s->get_max_i_stride();
    int nj  = s->get_max_j_stride();

    #ifdef __HIPCC_
    const int istride = s->ny;
    const int jstride = 1;
    #else 
    const int istride = 1;
    const int jstride = s->nx;
    #endif 

    Conserved uxl, uxr, uyl, uyr, fl, fr, gl, gr,  frf, flf, grf, glf;
    Primitive pxl, pxr, pyl, pyr;
    if (ii >= s->nx || jj >= s->ny) {
        return;
    }

    const auto cnull = Conserved{0, 0, 0, 0};
    const auto pnull = Primitive{0, 0, 0, 0};
    const int gid    = s->get_global_idx(ii, jj);
    const int limface = jj * jstride + (ii - 1 + (ii == 0)) * istride; 
    const int ljmface = (jj - 1 + (jj == 0)) * jstride + ii * istride; 
    const int lipface = jj * jstride + (ii + 1 - (ii == ni - 1)) * istride; 
    const int ljpface = (jj + 1 - (jj == nj - 1)) * jstride + ii * istride;
    // (i,j)-1/2 face
    pxl  = s->prims[limface]; 
    pxr  = s->prims[gid];
    pyl  = s->prims[ljmface]; 
    pyr  = pxr;        

    uxl  = s->prims2cons(pxl);
    uxr  = s->prims2cons(pxr);
    uyl  = s->prims2cons(pyl);
    uyr  = s->prims2cons(pyr);                         

    fl  = s->prims2flux(pxl, 1);
    fr  = s->prims2flux(pxr, 1);
    gl  = s->prims2flux(pyl, 2);
    gr  = s->prims2flux(pyr, 2);
    flf = s->calc_hll_flux(uxl, uxr, fl, fr, pxl, pxr, 1);
    glf = s->calc_hll_flux(uyl, uyr, gl, gr, pyl, pyr, 2);                  

    // i+1/2 face
    pxl  = s->prims[gid]; 
    pxr  = s->prims[lipface];
    pyl  = pxl; 
    pyr  = s->prims[ljpface];        

    uxl  = s->prims2cons(pxl);
    uxr  = s->prims2cons(pxr);
    uyl  = s->prims2cons(pyl);
    uyr  = s->prims2cons(pyr);                         

    fl  = s->prims2flux(pxl, 1);
    fr  = s->prims2flux(pxr, 1);
    gl  = s->prims2flux(pyl, 2);
    gr  = s->prims2flux(pyr, 2);
    frf = s->calc_hll_flux(uxl, uxr, fl, fr, pxl, pxr, 1);
    grf = s->calc_hll_flux(uyl, uyr, gl, gr, pyl, pyr, 2);                      

    s->sys_state[gid] -= (frf - flf) / s->dx * dt + (grf - glf) / s->dy * dt;

}

__global__ void hip_euler2d::shared_gpu_evolve(SimState * s, double dt)
{
    extern __shared__ Primitive primitive_buff[];
    int jj  = blockDim.x * blockIdx.x + threadIdx.x;
    int ii  = blockDim.y * blockIdx.y + threadIdx.y;
    int tj  = threadIdx.x;
    int ti  = threadIdx.y;
    int tja = tj + 1;
    int tia = ti + 1;
    int gid = s->get_global_idx(ii, jj);
    #ifdef __HIPCC__
    const int istride = s->ny;
    const int jstride = 1;
    #else 
    const int istride = 1;
    const int jstride = s->nx;
    #endif 

    if (ii >= s->nx || jj >= s->ny){
        return;
    }

    primitive_buff[tia * bj + tja * bi] = s->prims[gid];
    // If I'm at the thread block boundary, load the global neighbor
    if (tia == 1){
        const int limface = jj * jstride + (ii - 1 + (ii == 0)) * istride; 
        const int lipface = jj * jstride + (ii + SH_BLOCK_SIZE - (ii + SH_BLOCK_SIZE >= s->nx - 1) * (SH_BLOCK_SIZE + ii + 1 - s->nx)) * istride; 
        primitive_buff[(tia - 1)  * bj + (tja + 0) * bi] = s->prims[limface];
        primitive_buff[(tia + SH_BLOCK_SIZE) * bj + (tja + 0) * bi] = s->prims[lipface]; 
    }
    if (tja == 1){
        const int ljmface = (jj - 1 + (jj == 0)) * jstride + ii * istride; 
        const int ljpface = (jj + SH_BLOCK_SIZE - (jj + SH_BLOCK_SIZE >= s->ny - 1) * (SH_BLOCK_SIZE + jj + 1 - s->ny)) * jstride + ii * istride;
        primitive_buff[(tja - 1)  * bi + tia * bj] = s->prims[ljmface];
        primitive_buff[(tja + SH_BLOCK_SIZE) * bi + tia * bj] = s->prims[ljpface];
    }
        
    // synchronize threads (maybe)
    __syncthreads();

    // (i,j)-1/2 face
    Primitive pxl  = primitive_buff[(tia - 1) * bj + (tja + 0) * bi]; 
    Primitive pxr  = primitive_buff[(tia + 0) * bj + (tja + 0) * bi];
    Primitive pyl  = primitive_buff[(tia + 0) * bj + (tja - 1) * bi]; 
    Primitive pyr  = primitive_buff[(tia + 0) * bj + (tja + 0) * bi];        

    Conserved uxl  = s->prims2cons(pxl);
    Conserved uxr  = s->prims2cons(pxr);
    Conserved uyl  = s->prims2cons(pyl);
    Conserved uyr  = s->prims2cons(pyr);                         

    Conserved fl  = s->prims2flux(pxl, 1);
    Conserved fr  = s->prims2flux(pxr, 1);
    Conserved gl  = s->prims2flux(pyl, 2);
    Conserved gr  = s->prims2flux(pyr, 2);
    const Conserved flf = s->calc_hll_flux(uxl, uxr, fl, fr, pxl, pxr, 1);
    const Conserved glf = s->calc_hll_flux(uyl, uyr, gl, gr, pyl, pyr, 2);
    

    // (i, j)+1/2 face
    pxl  = primitive_buff[(tia + 0) * bj + (tja + 0) * bi];
    pxr  = primitive_buff[(tia + 1) * bj + (tja + 0) * bi];
    pyl  = primitive_buff[(tia + 0) * bj + (tja + 0) * bi];
    pyr  = primitive_buff[(tia + 0) * bj + (tja + 1) * bi];       

    uxl  = s->prims2cons(pxl);
    uxr  = s->prims2cons(pxr);
    uyl  = s->prims2cons(pyl);
    uyr  = s->prims2cons(pyr);     

    fl   = s->prims2flux(pxl, 1);
    fr   = s->prims2flux(pxr, 1);
    gl   = s->prims2flux(pyl, 2);
    gr   = s->prims2flux(pyr, 2);
    const Conserved frf  = s->calc_hll_flux(uxl, uxr, fl, fr, pxl, pxr, 1);
    const Conserved grf  = s->calc_hll_flux(uyl, uyr, gl, gr, pyl, pyr, 2); 

    s->sys_state[gid] -= ((frf - flf) / s->dx + (grf - glf) / s->dy) * dt;
    s->prims[gid]      = s->cons2prim(s->sys_state[gid]);

}

__global__ void hip_euler2d::gpu_cons2prim(SimState *s){
    const int jj  = blockDim.x * blockIdx.x + threadIdx.x;
    const int ii  = blockDim.y * blockIdx.y + threadIdx.y;
    const int ni  = s->get_max_i_stride();
    const int nj  = s->get_max_j_stride();
    const int gid = s->get_global_idx(ii, jj);
    if (ii < ni && jj < nj){
        double rho = s->sys_state[gid].rho;
        double v1  = s->sys_state[gid].m1/rho;
        double v2  = s->sys_state[gid].m2/rho;

        double p = 
            (ADIABATIC_GAMMA - 1.0) * (s->sys_state[gid].energy - 0.5 * rho * (v1 * v1 + v2*v2));

        s->prims[gid] = Primitive{rho, v1, v2, p};

    }
}

__global__ void hip_euler2d::shared_gpu_cons2prim(SimState *s){
    extern __shared__ Conserved  conserved_buff[];
    const int jj  = blockDim.x * blockIdx.x + threadIdx.x;
    const int ii  = blockDim.y * blockIdx.y + threadIdx.y;
    const int ni  = s->get_max_i_stride();
    const int nj  = s->get_max_j_stride();
    const int gid = s->get_global_idx(ii, jj);
    const int tj  = threadIdx.x;
    const int ti  = threadIdx.y;
    const int tid = tj * bi + ti * bj;
    const int nx  = s->nx;
    const int ny  = s->ny;

    // printf("%d, %d\n", jj, ii);
    if (ii < s->nx && jj < s->ny){
        // load shared memory
        conserved_buff[tid] = s->sys_state[gid];
        double rho = conserved_buff[tid].rho;
        double v1  = conserved_buff[tid].m1 / rho;
        double v2  = conserved_buff[tid].m2 / rho;

        double p = 
            (ADIABATIC_GAMMA - 1.0) * (conserved_buff[tid].energy 
                - 0.5 * rho * (v1 * v1 + v2*v2));

        // Write back to global
        s->prims[gid] = Primitive{rho, v1, v2, p};

    }
}

void hip_euler2d::evolve(SimState *s, int nxBlocks, int nyBlocks, int shared_blocks, int nzones, double tend, double dt)
{
    double t = 0.0;
    high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> delta_t;
    double zu_avg = 0;
    int n = 1;
    int nfold = 10;
    int ncheck = 0;
    dim3 group_size = dim3(nxBlocks, nyBlocks, 1);
    dim3 block_size = dim3(shared_blocks, shared_blocks, 1);
    int shared_memory = (block_size.x + 2) * (block_size.y + 2) * sizeof(Primitive);
    std::cout << "nzones: " << nzones << "\n";
    while (t < tend)
    {
        t1 = high_resolution_clock::now();
        for (int i = 0; i < nfold; i++)
        {
            #ifdef GLOBAL_EVOLVE
            gpu_evolve<<<group_size, block_size, 0, 0>>>(s, dt);
            gpu_cons2prim<<<group_size, block_size, 0, 0>>>(s);
            #else
            shared_gpu_evolve<<<group_size, block_size, shared_memory, 0>>>(s, dt);
            // gpu_cons2prim<<<group_size, block_size, 0, 0>>>(s);
            // shared_gpu_cons2prim<<<group_size, block_size, shared_memory, 0>>>(s);
            #endif 
            t += dt;
            n++;
        }
        hipDeviceSynchronize();
        t2 = high_resolution_clock::now();
        ncheck += 1;
        delta_t = t2 - t1;
        zu_avg += nzones * nfold / delta_t.count();
        std::cout << std::fixed << std::setprecision(3) << std::scientific;
            std::cout << "\r"
                << "Iteration: " << std::setw(5) << n 
                << "\t"
                << "Time: " << std::setw(10) <<  t
                << "\t"
                << "Zones/sec: "<< nzones * nfold / delta_t.count() << std::flush;
    }
    std::cout << "\n";
    std::cout << "Average zone_updates/sec for: " 
    << n << " iterations was " 
    << zu_avg / ncheck << " zones/sec" << "\n";
}

SimDualSpace::SimDualSpace(){}

SimDualSpace::~SimDualSpace(){}
void SimDualSpace::copyStateToGPU(
    const SimState &host,
    SimState *device
)
{
    int nx = host.nx;
    int ny = host.ny;
    

    //--------Allocate the memory for pointer objects-------------------------
    hipMalloc((void **)&host_u0,     nx * ny * sizeof(Conserved));
    hipMalloc((void **)&host_prims,  nx * ny * sizeof(Primitive));

    //--------Copy the host resources to pointer variables on host
    hipMemcpy(host_u0,    host.sys_state, nx * ny * sizeof(Conserved), hipMemcpyHostToDevice);
    hipMemcpy(host_prims, host.prims    , nx * ny * sizeof(Primitive), hipMemcpyHostToDevice);

    // copy pointer to allocated device storage to device class
    if ( hipMemcpy(&(device->sys_state), &host_u0,    sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_u0 -> device_sys_tate");
    };
    if( hipMemcpy(&(device->prims),     &host_prims, sizeof(Primitive *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_prims -> device_prims");
    };

    hipMemcpy(&(device->dt),     &host.dt     ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->dx),     &host.dx     ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->dy),     &host.dy     ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->xmin),   &host.xmin   ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->xmax),   &host.xmax   ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->ymin),   &host.ymin   ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->ymax),   &host.ymax   ,  sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(&(device->nx),     &host.nx     ,  sizeof(int),    hipMemcpyHostToDevice);
    hipMemcpy(&(device->ny),     &host.ny     ,  sizeof(int),    hipMemcpyHostToDevice);

    
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
    int nx = host.nx;
    int ny = host.ny;

    hipMemcpy(host.sys_state, host_u0,        nx * ny * sizeof(Conserved), hipMemcpyDeviceToHost);
    hipCheckErrors("Memcpy failed at transferring device conservatives to host");
    hipMemcpy(host.prims,     host_prims ,    nx * ny * sizeof(Primitive), hipMemcpyDeviceToHost);
    hipCheckErrors("Memcpy failed at transferring device prims to host");
    
}

void SimDualSpace::cleanUp()
{
    printf("Freeing Device Memory...\n");
    hipFree(host_u0);
    hipFree(host_prims);
    printf("Memory Freed.\n");
    
}

