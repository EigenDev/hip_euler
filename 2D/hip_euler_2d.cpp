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
    // u1        = (Conserved*) malloc(numzones * sizeof(Conserved));
    // du_dt     = (Conserved*) malloc(numzones * sizeof(Conserved));
    prims     = (Primitive*) malloc(nx * ny * sizeof(Primitive));

    for (int jj = 0; jj < ny; jj++)
    {
        for (int ii = 0; ii < nx; ii++){
            ds = sqrt( (ii * dx)*(ii * dx) + (jj * dy)*(jj * dy) ) ;
            if (ds < 0.5 * radius ){
                prims[jj * nx + ii] = Primitive{1.0, 0.0, 0.0, 1.0};
            } else {
                prims[jj * nx + ii] = Primitive{0.125, 0.0, 0.0, 0.1};
            }
        }
    }

    // Construct the initial conservatives 
    prims2cons(prims);
}

SimState::~SimState(){
    free(sys_state);
    // free(u1);
    // free(du_dt);
    free(prims);
};

// Convert from the conservarive array to primitive and cache it
GPU_CALLABLE_MEMBER void SimState::cons2prim(const Conserved *u)
{   
    double v1, v2, p;
    for (int jj = 0; jj < ny; jj++){

        for(int ii = 0; ii < nx; ii++){
            v1 = u[jj*nx + ii].m1/u[jj * nx + ii].rho;
            v2 = u[jj*nx + ii].m2/u[jj * nx + ii].rho;

            p = 
                (ADIABATIC_GAMMA - 1.0) * (u[jj * nx + ii].energy - 0.5 * u[jj * nx + ii].rho * (v1 * v1 + v2 * v2));

            prims[jj * nx + ii] = Primitive{u[jj * nx + ii].rho, v1, v2, p};

        }
    }
    
    
}//-----End cons2prims



GPU_CALLABLE_MEMBER Conserved SimState::prims2cons(const Primitive &prims)
{
    double m1 = prims.rho * prims.v1;
    double m2 = prims.rho * prims.v2;
    double e = 
        prims.p/(ADIABATIC_GAMMA - 1.0) + 0.5 * (prims.v1 * prims.v1 + prims.v2 * prims.v2) * prims.rho;

    return Conserved{prims.rho, m1, m2, e};

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
    double rhol = left_prims.rho;
    double v1l  = left_prims.v1 ;
    double v2l  = left_prims.v2 ;
    double pl   = left_prims.p  ;

    double rhor = right_prims.rho;
    double v1r  = right_prims.v1;
    double v2r  = right_prims.v2;
    double pr   = right_prims.p  ;

    double csl  = sqrt(ADIABATIC_GAMMA * pl / rhol);
    double csr  = sqrt(ADIABATIC_GAMMA * pr/  rhor);

    switch (nhat)
    {
    case 1:
        {
            double aL   = min(v1l - csl, v1r - csr);
            double aR   = max(v1r + csr, v1l + csl);

            return EigenWave(aL, aR);
        }
    
    case 2:
        {
            double aL   = min(v2l - csl, v2r - csr);
            double aR   = max(v2r + csr, v2l + csl);

            return EigenWave(aL, aR);
        }
        
    }
    

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
    double v1 = prims.v1;
    double v2 = prims.v2;
    double e = 
        prims.p/(ADIABATIC_GAMMA - 1.0) + 0.5 * (v1*v1 + v2*v2)* prims.rho;
    
    switch (nhat)
    {
    case 1:
        {
            double rhof = prims.rho * v1;
            double momf = prims.rho * v1 * v1 + prims.p;
            double conv = prims.rho*v1*v2;
            double engf = (e + prims.p)*v1;

            return Conserved{rhof, momf, conv, engf};
        }
        
    
    case 2:
        {
            double rhof = prims.rho * v2;
            double momf = prims.rho * v2 * v2 + prims.p;
            double conv = prims.rho*v1*v2;
            double engf = (e + prims.p)*v2;

            return Conserved{rhof, conv, momf, engf};
        }
        
    }
    
}// End prims2flux


__global__ void hip_euler2d::gpu_evolve(SimState * s, double dt)
{
    int ii = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int jj = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    Conserved uxl, uxr, uyl, uyr, fl, fr, gl, gr,  frf, flf, grf, glf;
    Primitive pxl, pxr, pyl, pyr;
    int nx = s->nx;
    int ny = s->ny;

    // printf("Thread coord: (%d, %d)\n", hipThreadIdx_x, hipThreadIdx_y);
    // printf("Global Index: %d    \n",  jj*s->nx + ii);
    if (ii < s->nx && jj < s->ny){
        // (i,j)-1/2 face
        uxl = (ii > 0) ? s->sys_state[jj*nx + ii - 1]   : s->sys_state[jj * nx + ii];
        uxr =                                             s->sys_state[jj * nx + ii];
        uyl = (jj > 0) ? s->sys_state[(jj - 1)*nx + ii] : s->sys_state[jj * nx + ii];
        uyr =                                             s->sys_state[jj * nx + ii];
        pxl  = (ii > 0) ? s->prims[jj*nx + ii - 1]      : s->prims[jj*nx + ii];
        pxr  =                                            s->prims[jj*nx + ii];
        pyl  = (jj > 0) ? s->prims[(jj - 1)*nx + ii]    : s->prims[jj*nx + ii];
        pyr  =                                            s->prims[jj*nx + ii];                    

        fl  = s->prims2flux(pxl, 1);
        fr  = s->prims2flux(pxr, 1);
        gl  = s->prims2flux(pyl, 2);
        gr  = s->prims2flux(pyr, 2);
        flf = s->calc_hll_flux(uxl, uxr, fl, fr, pxl, pxr, 1);
        glf = s->calc_hll_flux(uyl, uyr, gl, gr, pyl, pyr, 2);
        

        // i+1/2 face
        uxl =                                                    s->sys_state[jj * nx + ii];
        uxr = (ii < nx - 1) ? s->sys_state[jj * nx + (ii + 1)]:  s->sys_state[jj * nx + ii];
        uyl =                                                    s->sys_state[jj * nx + ii];
        uyr = (jj < ny - 1) ? s->sys_state[(jj + 1)* nx + ii] : s->sys_state[jj * nx + ii];
        pxl  =                                                   s->prims[jj*nx + ii];
        pxr  = (ii < nx - 1) ? s->prims[jj*nx + (ii + 1)]      : s->prims[jj*nx + ii];
        pyl  =                                                   s->prims[jj*nx + ii];
        pyr  = (jj < ny - 1) ? s->prims[(jj + 1)*nx + ii]      : s->prims[jj*nx + ii];                      

        fl  = s->prims2flux(pxl, 1);
        fr  = s->prims2flux(pxr, 1);
        gl  = s->prims2flux(pyl, 2);
        gr  = s->prims2flux(pyr, 2);
        frf = s->calc_hll_flux(uxl, uxr, fl, fr, pxl, pxr, 1);
        grf = s->calc_hll_flux(uyl, uyr, gl, gr, pyl, pyr, 2); 

        s->sys_state[jj*nx + ii] -= (frf - flf) / s->dx * dt + (grf - glf) / s->dy * dt ;
    }

}

__global__ void hip_euler2d::shared_gpu_evolve(SimState * s, double dt)
{
    __shared__ Conserved conserved_buff[4 + 2][4 + 2];
    __shared__ Primitive primitive_buff[4 + 2][4 + 2];

    int ii = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int jj = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    int txa = threadIdx.x + 1;
    int tya = threadIdx.y + 1;
    int wx = hipBlockDim_x;
    int wy = hipBlockDim_y;
    int nx = s->nx;
    int ny = s->ny;

    Conserved uxl, uxr, uyl, uyr, fl, fr, gl, gr,  frf, flf, grf, glf;
    Primitive pxl, pxr, pyl, pyr;

    // printf("TY: %d, TY roll to left: %d Block_dim: %d\n", ty, (unsigned)(ty - 1)%wy, wy);

    if (ii < s->nx && jj < s->ny){
        int gid = jj*nx + ii;

        // load shared memory
        conserved_buff[tya][txa] = s->sys_state[gid];
        primitive_buff[tya][txa] = s->prims[gid];

        conserved_buff[tya][txa - 1] = 
            (ii > 0) ? s->sys_state[jj*nx + ii - 1]  : conserved_buff[tya][txa];

        conserved_buff[tya][txa + 1] = 
            (ii < nx - 1) ? s->sys_state[jj*nx + ii + 1]  : conserved_buff[tya][txa];

        conserved_buff[tya - 1][txa] = 
            (jj > 0) ? s->sys_state[(jj - 1)*nx + ii]  : conserved_buff[tya][txa];

        conserved_buff[tya + 1][txa] = 
            (jj < ny - 1) ? s->sys_state[(jj + 1) * nx + ii]  : conserved_buff[tya][txa];

        primitive_buff[tya][txa - 1] = 
            (ii > 0) ? s->prims[jj*nx + ii - 1]  : primitive_buff[tya][txa];

        primitive_buff[tya][txa + 1] = 
            (ii < nx - 1) ? s->prims[jj*nx + ii + 1]  : primitive_buff[tya][txa];

        primitive_buff[tya - 1][txa] = 
            (jj > 0) ? s->prims[(jj - 1)*nx + ii]  : primitive_buff[tya][txa];

        primitive_buff[tya + 1][txa] = 
            (jj < ny - 1) ? s->prims[(jj + 1) * nx + ii]  : primitive_buff[tya][txa];

        // synchronize threads (maybe)
        __syncthreads();

        // (i,j)-1/2 face
        uxl  = conserved_buff[tya][txa - 1]; 
        uxr  = conserved_buff[tya][txa];
        uyl  = conserved_buff[tya - 1][txa]; 
        uyr  = conserved_buff[tya][txa];

        pxl  = primitive_buff[tya][txa - 1]; 
        pxr  = primitive_buff[tya][txa];
        pyl  = primitive_buff[tya - 1][txa]; 
        pyr  = primitive_buff[tya][txa];                                      

        fl  = s->prims2flux(pxl, 1);
        fr  = s->prims2flux(pxr, 1);
        gl  = s->prims2flux(pyl, 2);
        gr  = s->prims2flux(pyr, 2);
        flf = s->calc_hll_flux(uxl, uxr, fl, fr, pxl, pxr, 1);
        glf = s->calc_hll_flux(uyl, uyr, gl, gr, pyl, pyr, 2);
        

        // // i+1/2 face
        uxl = conserved_buff[tya][txa];
        uxr = conserved_buff[tya][txa + 1];
        uyl = conserved_buff[tya][txa];
        uyr = conserved_buff[tya + 1][txa];

        pxl  = primitive_buff[tya][txa];
        pxr  = primitive_buff[tya][txa + 1];
        pyl  = primitive_buff[tya][txa];
        pyr  = primitive_buff[tya + 1][txa];                      

        fl  = s->prims2flux(pxl, 1);
        fr  = s->prims2flux(pxr, 1);
        gl  = s->prims2flux(pyl, 2);
        gr  = s->prims2flux(pyr, 2);
        frf = s->calc_hll_flux(uxl, uxr, fl, fr, pxl, pxr, 1);
        grf = s->calc_hll_flux(uyl, uyr, gl, gr, pyl, pyr, 2); 

        s->sys_state[gid] -= (frf - flf) / s->dx * dt + (grf - glf) / s->dy * dt ;
    }

}

__global__ void hip_euler2d::gpu_cons2prim(SimState *s){
    int ii = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int jj = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    int nx = s->nx;
    int ny = s->ny;

    // printf("%d, %d\n", jj, ii);
    if (ii < nx && jj < ny){
        double v1 = s->sys_state[jj*nx + ii].m1/s->sys_state[jj*nx + ii].rho;
        double v2 = s->sys_state[jj*nx + ii].m2/s->sys_state[jj*nx + ii].rho;

        double p = 
            (ADIABATIC_GAMMA - 1.0) * (s->sys_state[jj*nx + ii].energy - 0.5 * s->sys_state[jj * nx + ii].rho * (v1 * v1 + v2*v2));

        s->prims[jj*nx + ii] = Primitive{s->sys_state[jj*nx + ii].rho, v1, v2, p};

    }
}

__global__ void hip_euler2d::shared_gpu_cons2prim(SimState *s){
    __shared__ Conserved  conserved_buff[4][4];
    __shared__ Primitive  primitive_buff[4][4];

    int ii = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int jj = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int wx = hipBlockDim_x;
    int wy = hipBlockDim_y;

    int nx = s->nx;
    int ny = s->ny;

    // printf("%d, %d\n", jj, ii);
    if (ii < nx && jj < ny){
        int gid = jj*nx + ii;

        // load shared memory
        conserved_buff[ty][tx] = s->sys_state[gid];
        primitive_buff[ty][tx] = s->prims[gid];
        
        double v1 = conserved_buff[ty][tx].m1/conserved_buff[ty][tx].rho;
        double v2 = conserved_buff[ty][tx].m2/conserved_buff[ty][tx].rho;

        double p = 
            (ADIABATIC_GAMMA - 1.0) * (conserved_buff[ty][tx].energy - 0.5 * conserved_buff[ty][tx].rho * (v1 * v1 + v2*v2));

        // Write back to global
        s->prims[gid] = Primitive{conserved_buff[ty][tx].rho, v1, v2, p};

    }
}

void hip_euler2d::evolve(SimState *s, int nxBlocks, int nyBlocks, int block_size, int nzones, double tend, double dt)
{
    double t = 0.0;
    high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> delta_t;
    double zu_avg = 0;
    int n = 1;
    int nfold = 1000;
    int ncheck = 0;
    while (t < tend)
    {
        t1 = high_resolution_clock::now();
        hipLaunchKernelGGL(shared_gpu_evolve,    dim3(nxBlocks, nyBlocks), dim3(block_size, block_size), 0, 0, s, dt);
        hipLaunchKernelGGL(shared_gpu_cons2prim, dim3(nxBlocks, nyBlocks), dim3(block_size, block_size), 0, 0, s);
        hipDeviceSynchronize();
        // std::cout << n << "\n";
        // hipCheckErrors("Kernel died: ");
        if (n >= nfold){
            ncheck += 1;
            t2 = high_resolution_clock::now();
            delta_t = t2 - t1;
            zu_avg += nzones / delta_t.count();
            std::cout << std::fixed << std::setprecision(3) << std::scientific;
                std::cout << "\r"
                    << "Iteration: " << std::setw(5) << n 
                    << "\t"
                    << "Time: " << std::setw(10) <<  t
                    << "\t"
                    << "Zones/sec: "<< nzones / delta_t.count() << std::flush;
            nfold += 1000;
        }

        t += dt;
        n++;
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
    // hipMalloc((void **)&host_u1,     nz * sizeof(Conserved));
    // hipMalloc((void **)&host_dudt,   nz * sizeof(Conserved));
    hipMalloc((void **)&host_prims,  nx * ny * sizeof(Primitive));
    // hipMalloc((void **)&u0        , sizeof(Conserved));

    //--------Copy the host resources to pointer variables on host
    hipMemcpy(host_u0,    host.sys_state, nx * ny * sizeof(Conserved), hipMemcpyHostToDevice);
    // hipMemcpy(host_u1,    host.u1       , nz * sizeof(Conserved), hipMemcpyHostToDevice);
    // hipMemcpy(host_dudt,  host.du_dt    , nz * sizeof(Conserved), hipMemcpyHostToDevice);
    hipMemcpy(host_prims, host.prims    , nx * ny * sizeof(Primitive), hipMemcpyHostToDevice);

    // copy pointer to allocated device storage to device class
    if ( hipMemcpy(&(device->sys_state), &host_u0,    sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
    {
        printf("Hip Memcpy failed at: host_u0 -> device_sys_tate");
    };
    // if( hipMemcpy(&(device->u1),        &host_u1,    sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
    // {
    //     printf("Hip Memcpy failed at: host_u1 -> device_u1");
    // };

    // if( hipMemcpy(&(device->du_dt),     &host_dudt,  sizeof(Conserved *),  hipMemcpyHostToDevice) != hipSuccess )
    // {
    //     printf("Hip Memcpy failed at: host_dudt -> device_du_dt");
    // };

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
    // hipFree(host_u1);
    // hipFree(host_dudt);
    hipFree(host_prims);
    printf("Memory Freed.\n");
    
}

