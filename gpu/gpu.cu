#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

/**
 * This is your initialization function! We pass in h0, u0, and v0, which are
 * your initial height, u velocity, and v velocity fields. You should send these
 * grids to the GPU so you can do work on them there, and also these other fields.
 * Here, length and width are the length and width of the domain, and nx and ny are
 * the number of grid points in the x and y directions. H is the height of the water
 * column, g is the acceleration due to gravity, and dt is the time step size.
 * The rank and num_procs variables are unused here, but you will need them
 * when doing the MPI version.
 */
// Device pointers
double *c_h, *c_u, *c_v, *c_dh, *c_du, *c_dv, *c_dh1, *c_du1, *c_dv1, *c_dh2, *c_du2, *c_dv2;

// Simulation parameters
double H, g, dt, dx, dy;
int nx, ny;
int t = 0;

void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    // Assign values to simulation parameters
    nx = nx_;
    ny = ny_;
    H = H_;
    g = g_;
    dt = dt_;
    dx = length_ / nx;
    dy = width_ / ny;

    size_t size = nx * ny * sizeof(double);

    // Allocate device memory
    cudaMalloc((void**)&c_h, size);
    cudaMalloc((void**)&c_u, size);
    cudaMalloc((void**)&c_v, size);

    cudaMalloc((void**)&c_dh, size);
    cudaMalloc((void**)&c_du, size);
    cudaMalloc((void**)&c_dv, size);

    cudaMalloc((void**)&c_dh1, size);
    cudaMalloc((void**)&c_du1, size);
    cudaMalloc((void**)&c_dv1, size);

    cudaMalloc((void**)&c_dh2, size);
    cudaMalloc((void**)&c_du2, size);
    cudaMalloc((void**)&c_dv2, size);

    // Copy initial data to device
    cudaMemcpy(c_h, h0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_u, u0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_v, v0, size, cudaMemcpyHostToDevice);

    // Initialize derivative arrays to zero
    cudaMemset(c_dh, 0, size);
    cudaMemset(c_du, 0, size);
    cudaMemset(c_dv, 0, size);

    cudaMemset(c_dh1, 0, size);
    cudaMemset(c_du1, 0, size);
    cudaMemset(c_dv1, 0, size);

    cudaMemset(c_dh2, 0, size);
    cudaMemset(c_du2, 0, size);
    cudaMemset(c_dv2, 0, size);
}

/**
 * This is your step function! Here, you will actually numerically solve the shallow
 * water equations. You should update the h, u, and v fields to be the solution after
 * one time step has passed.
 */
void step()
{
    // @TODO: Your code here
}

/**
 * This is your transfer function! You should copy the h field back to the host
 * so that the CPU can check the results of your computation.
 */
void transfer(double *h_host)
{
    size_t size = nx * ny * sizeof(double);
    cudaMemcpy(h_host, c_h, size, cudaMemcpyDeviceToHost);
}

/**
 * This is your finalization function! You should free all of the memory that you
 * allocated on the GPU here.
 */
void free_memory()
{
    cudaFree(c_h);
    cudaFree(c_u);
    cudaFree(c_v);

    cudaFree(c_dh);
    cudaFree(c_du);
    cudaFree(c_dv);

    cudaFree(c_dh1);
    cudaFree(c_du1);
    cudaFree(c_dv1);

    cudaFree(c_dh2);
    cudaFree(c_du2);
    cudaFree(c_dv2);
}