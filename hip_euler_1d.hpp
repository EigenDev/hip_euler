/**
 *  A Header file for inlined operator overloading
 * for the 1D Euler code in HIP
 */

#ifndef EULER_1D_HPP
#define EULER_1D_HPP

/**
 * Check if the compiler define _CUDACC__
 * If not, use standard CPU calls
*/
#include "hip/hip_runtime.h"
#include "gpu_error_check.h"

#ifdef __HIPCC__
#define GPU_CALLABLE_MEMBER __host__ __device__
#else
#define GPU_CALLABLE_MEMBER __host__ __device__
#endif 




// Static const values 
constexpr double PI = 3.14159265358979323846;
constexpr double ADIABATIC_GAMMA = 5.0/3.0;

namespace hip_euler
{

    struct Conserved
    {
        double rho, m, energy;
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() = default;
        GPU_CALLABLE_MEMBER Conserved(double rho, double m, double energy) : rho(rho), m(m), energy(energy) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &u) : rho(u.rho), m(u.m), energy(u.energy) {}

        //---------- Operator overloading for laziness ------------------
        GPU_CALLABLE_MEMBER Conserved operator+(const Conserved &u) const { return Conserved{rho + u.rho, m + u.m, energy + u.energy}; }
        GPU_CALLABLE_MEMBER Conserved operator-(const Conserved &u) const { return Conserved{rho - u.rho, m - u.m, energy - u.energy}; }
        GPU_CALLABLE_MEMBER Conserved operator*(const double c) const { return Conserved{rho * c, m * c, energy * c}; }
        GPU_CALLABLE_MEMBER Conserved operator/(const double c) const { return Conserved{rho / c, m / c, energy / c}; }
        GPU_CALLABLE_MEMBER Conserved& operator-=(const Conserved &u) {
            this->rho    -= u.rho;
            this->m      -= u.m;
            this->energy -= u.energy;
            return *this; }

    }; //---------End Conserved

    //===============================
    struct Primitive
    {
        double rho, v, p;
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() = default;
        GPU_CALLABLE_MEMBER Primitive(double rho, double v, double p) : rho(rho), v(v), p(p) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &u) : rho(u.rho), v(u.v), p(u.p) {}

        //---------- Operator overloading for lazine ------------------
        GPU_CALLABLE_MEMBER Primitive operator+(const Primitive &u) const { return Primitive{rho + u.rho, v + u.v, p + u.p}; }
        GPU_CALLABLE_MEMBER Primitive operator-(const Primitive &u) const { return Primitive{rho - u.rho, v - u.v, p - u.p}; }
        GPU_CALLABLE_MEMBER Primitive operator*(const double c) const { return Primitive{rho * c, v * c, p * c}; }
        GPU_CALLABLE_MEMBER Primitive operator/(const double c) const { return Primitive{rho / c, v / c, p / c}; }

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
        double dx, xmin, xmax, tend, dt;
        int nzones;
        Conserved *sys_state;
        Conserved *du_dt    ;
        Conserved *u1       ;
        Primitive *prims    ;


        /* Below is how it can be done in standard CPP for example */
        // std::vector<Conserved> sys_state, du_dt, u1;
        // std::vector<Primitive> prims;

        SimState();
        SimState(const int nzones, const double xmin, const double xmax);
        ~SimState();

        void initializeModel(const double x0, const double x1, const int num_zones);

        // Convert from the conservarive array to primitive and cache it
        GPU_CALLABLE_MEMBER void cons2prim(const Conserved *u_state);

        GPU_CALLABLE_MEMBER Conserved prims2cons(const Primitive &prims);

        GPU_CALLABLE_MEMBER void      prims2cons(const Primitive *prims);

        GPU_CALLABLE_MEMBER EigenWave calc_waves(const Primitive &left_prims, const Primitive &right_prims);

        GPU_CALLABLE_MEMBER Conserved calc_hll_flux(
            const Conserved &left_state,
            const Conserved &right_state,
            const Conserved &left_flux,
            const Conserved &right_flux,
            const Primitive &left_prims,
            const Primitive &right_prims);

        GPU_CALLABLE_MEMBER Conserved prims2flux(const Primitive &prims);

        void udot (int nBlocks, int block_size);
    };

    __global__ void gpu_evolve(SimState * s, double dt);
    __global__ void gpu_cons2prim(SimState *s);

    void evolve(SimState *s, int nBlocks, int block_size, int nzones, double tend, double dt);
    
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