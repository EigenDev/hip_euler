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
constexpr int SH_BLOCK_SIZE = 8;
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

namespace hip_euler2d
{

    struct Conserved
    {
        double rho, m1, m2, energy;
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() = default;
        GPU_CALLABLE_MEMBER Conserved(double rho, double m1, double m2, double energy) : rho(rho), m1(m1), m2(m2), energy(energy) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &u) : rho(u.rho), m1(u.m1), m2(u.m2), energy(u.energy) {}

        //---------- Operator overloading for laziness ------------------
        GPU_CALLABLE_MEMBER Conserved operator+(const Conserved &u) const { return Conserved{rho + u.rho, m1 + u.m1, m2 + u.m2, energy + u.energy}; }
        GPU_CALLABLE_MEMBER Conserved operator-(const Conserved &u) const { return Conserved{rho - u.rho, m1 - u.m1, m2 - u.m2, energy - u.energy}; }
        GPU_CALLABLE_MEMBER Conserved operator*(const double c) const { return Conserved{rho * c, m1 * c, m2 * c, energy * c}; }
        GPU_CALLABLE_MEMBER Conserved operator/(const double c) const { return Conserved{rho / c, m1 / c, m2 / c, energy / c}; }
        GPU_CALLABLE_MEMBER Conserved& operator-=(const Conserved &u) {
            this->rho    -= u.rho;
            this->m1     -= u.m1;
            this->m2     -= u.m2;
            this->energy -= u.energy;
            return *this; }

    }; //---------End Conserved

    //===============================
    struct Primitive
    {
        double rho, v1, v2, p;
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() = default;
        GPU_CALLABLE_MEMBER Primitive(double rho, double v1, double v2, double p) : rho(rho), v1(v1), v2(v2), p(p) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &u) : rho(u.rho), v1(u.v1), v2(u.v2), p(u.p) {}

        //---------- Operator overloading for lazine ------------------
        GPU_CALLABLE_MEMBER Primitive operator+(const Primitive &u) const { return Primitive{rho + u.rho, v1 + u.v1, v2 + u.v2, p + u.p}; }
        GPU_CALLABLE_MEMBER Primitive operator-(const Primitive &u) const { return Primitive{rho - u.rho, v1 - u.v1, v2 - u.v2, p - u.p}; }
        GPU_CALLABLE_MEMBER Primitive operator*(const double c) const { return Primitive{rho * c, v1 * c, v2 * c, p * c}; }
        GPU_CALLABLE_MEMBER Primitive operator/(const double c) const { return Primitive{rho / c, v1 / c, v2 / c, p / c}; }

    }; //---------End Primitive

    //================================
    struct EigenWave
    {
        double aL, aR;
        GPU_CALLABLE_MEMBER EigenWave() {}
        GPU_CALLABLE_MEMBER ~EigenWave() = default;
        GPU_CALLABLE_MEMBER EigenWave(double aL, double aR) : aL(aL), aR(aR) {}
    }; //----------End EigenWave



    //=====================================
    struct SimState
    {
        double dx, dy, xmin, xmax, ymin, ymax, tend, dt;
        int nx, ny;
        Conserved *sys_state;
        Conserved *du_dt    ;
        Conserved *u1       ;
        Primitive *prims    ;


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
        GPU_CALLABLE_MEMBER void cons2prim(const Conserved *u_state);

        GPU_CALLABLE_MEMBER Conserved prims2cons(const Primitive &prims);

        GPU_CALLABLE_MEMBER void      prims2cons(const Primitive *prims);

        GPU_CALLABLE_MEMBER EigenWave calc_waves(const Primitive &left_prims, const Primitive &right_prims, const int nhat);

        GPU_CALLABLE_MEMBER Conserved calc_hll_flux(
            const Conserved &left_state,
            const Conserved &right_state,
            const Conserved &left_flux,
            const Conserved &right_flux,
            const Primitive &left_prims,
            const Primitive &right_prims,
            const int nhat);

        GPU_CALLABLE_MEMBER Conserved prims2flux(const Primitive &prims, int nhat);

        void udot (int nBlocks, int block_size);

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

        Primitive *host_prims;
        Conserved *host_u0;
        Conserved *host_u1;
        Conserved *host_dudt;

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