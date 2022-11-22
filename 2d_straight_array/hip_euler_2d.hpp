/**
 *  A Header file for inlined operator overloading
 * for the 1D Euler code in HIP
 */

#ifndef HIP_EULER_2D_HPP
#define HIP_EULER_2D_HPP

/**
 * Check if the compiler define _CUDACC__
 * If not, use standard CPU calls
*/
#include "hip/hip_runtime.h"
#include "gpu_error_check.h"

#ifdef __HIPCC__
#define GPU_CALLABLE_MEMBER __host__ __device__
#else
#define GPU_CALLABLE_MEMBER 
#endif 




// Static const values 
constexpr double PI = 3.14159265358979323846;
constexpr double ADIABATIC_GAMMA = 5.0/3.0;
constexpr int SH_BLOCK_SIZE = 32;
constexpr auto bz = SH_BLOCK_SIZE + 2;
constexpr auto bi = []() {
    #ifdef __HIPCC__
    return 1;
    #else
        return SH_BLOCK_SIZE + 2;
    #endif
}();

constexpr auto bj = []() {
    #ifdef __HIPCC__
    return SH_BLOCK_SIZE + 2;
    #else
        return 1;
    #endif
}();

namespace CONS_VAR {
    enum value {
        RHO  = 0,
        MOM1 = 1,
        MOM2 = 2,
        ENERGY = 3
    };
};

namespace PRIM_VAR {
    enum value {
        RHO = 0,
        V1 = 1,
        V2 = 2,
        P = 3
    };

    GPU_CALLABLE_MEMBER
    constexpr int vcomponent(const int ehat){
        return (ehat == 1) ? V1 : V2;
    };
};

namespace EIGEN_WAVE {
    enum value {
        AR = 0,
        AL = 1,
    };
};

template <typename T>
__host__ __device__ inline double* map_variables(T *arr, int idx){
    static double var[4];
    var[0] = arr[idx + 0];
    var[1] = arr[idx + 1];
    var[2] = arr[idx + 2];
    var[3] = arr[idx + 3];
    return var;
};

template <typename T>
__host__ __device__ inline void map_variables(T *arr, double var[4], int idx){
    arr[idx + 0] = var[0];
    arr[idx + 1] = var[1];
    arr[idx + 2] = var[2];
    arr[idx + 3] = var[3];
};


namespace hip_euler2d
{
    using Conserved = double*;
    using Primitive = double*;
    using EigenWave = double*;
    //=====================================
    struct SimState
    {
        const static int nvars = 4;
        double dx, dy, xmin, xmax, ymin, ymax, tend, dt;
        int nx, ny;
        Conserved sys_state;
        Primitive prims;


        /* Below is how it can be done in standard CPP for example */
        // std::vector<Conserved> sys_state, du_dt, u1;
        // std::vector<Primitive> prims;

        SimState();
        SimState(
            const int nx,
            const int ny, 
            const double xmin, 
            const double xmax,
            const double ymin,
            const double ymax);
        ~SimState();

        void initializeModel(
            const double x0, 
            const double x1,
            const double y0,
            const double y1, 
            const int nx,
            const int ny);

        // Convert from the conservarive array to primitive and cache it
        GPU_CALLABLE_MEMBER void cons2prim(const Conserved u_state);
        GPU_CALLABLE_MEMBER Primitive cell_cons2prim(const Conserved);
        GPU_CALLABLE_MEMBER Conserved cell_prims2cons(const Primitive prims);
        GPU_CALLABLE_MEMBER void prims2cons(const Primitive prims);
        GPU_CALLABLE_MEMBER EigenWave calc_waves(const Primitive left_prims, const Primitive right_prims, const int nhat);

        GPU_CALLABLE_MEMBER Conserved calc_hll_flux(
            const Conserved left_state,
            const Conserved right_state,
            const Conserved left_flux,
            const Conserved right_flux,
            const Primitive left_prims,
            const Primitive right_prims,
            const int nhat);

        GPU_CALLABLE_MEMBER Conserved prims2flux(const Primitive prims, int nhat);

        GPU_CALLABLE_MEMBER
        auto get_max_j_stride(){
            #ifdef __HIPCC__
            return nx;
            #else 
            return ny;
            #endif 
        }

        GPU_CALLABLE_MEMBER
        auto get_max_i_stride(){
            #ifdef __HIPCC__
            return ny;
            #else 
            return nx;
            #endif 
        }

        GPU_CALLABLE_MEMBER
        auto get_global_idx(const int ii, const int jj) {
            #ifdef __HIPCC__
            return ii * ny + jj;
            #else 
            return jj * nx + ii;
            #endif 
        }
    };

    __global__ void gpu_evolve(SimState * s, double dt);
    __global__ void shared_gpu_evolve(SimState * s, double dt);
    __global__ void gpu_cons2prim(SimState *s);
    __global__ void shared_gpu_cons2prim(SimState *s);

    void evolve(SimState *s, int nxBlocks, int nyBlocks, int block_size, int nzones, double tend, double dt);
    
    struct SimDualSpace{
        SimDualSpace();
        ~SimDualSpace();

        double *host_prims;
        double *host_u0;
        double *host_u1;
        double *host_dudt;

        double host_dt;
        double host_xmin; 
        double host_xmax;
        double host_dx;

        void copyStateToGPU(const SimState &host, SimState *device);
        void copyGPUStateToHost(const SimState *device, SimState &host);
        void cleanUp();

    };


}

#endif