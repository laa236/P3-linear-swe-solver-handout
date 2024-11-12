#include <cuda.h>
#include <cuda_runtime.h>
//#include <stdio.h>
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
double *h, *u, *v, *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;

// Simulation parameters
double H, g, dt, dx, dy;
int nx, ny;
int t = 0;

//
int numblocks_x, numblocks_y;

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
    size_t size_h = (nx + 1) * (ny + 1) * sizeof(double);
    size_t size_u = (nx + 2) * ny * sizeof(double);
    size_t size_v = nx * (ny + 2) * sizeof(double);

    // Allocate device memory
    cudaMalloc((void**)&h, size_h);
    cudaMalloc((void**)&u, size_u);
    cudaMalloc((void**)&v, size_v);

    cudaMalloc((void**)&dh, size);
    cudaMalloc((void**)&du, size);
    cudaMalloc((void**)&dv, size);

    cudaMalloc((void**)&dh1, size);
    cudaMalloc((void**)&du1, size);
    cudaMalloc((void**)&dv1, size);

    cudaMalloc((void**)&dh2, size);
    cudaMalloc((void**)&du2, size);
    cudaMalloc((void**)&dv2, size);

    // Copy initial data to device
    cudaMemcpy(h, h0, size_h, cudaMemcpyHostToDevice);
    cudaMemcpy(u, u0, size_u, cudaMemcpyHostToDevice);
    cudaMemcpy(v, v0, size_v, cudaMemcpyHostToDevice);

    // Initialize derivative arrays to zero
    cudaMemset(dh, 0, size);
    cudaMemset(du, 0, size);
    cudaMemset(dv, 0, size);

    cudaMemset(dh1, 0, size);
    cudaMemset(du1, 0, size);
    cudaMemset(dv1, 0, size);

    cudaMemset(dh2, 0, size);
    cudaMemset(du2, 0, size);
    cudaMemset(dv2, 0, size);

    numblocks_x = (nx + 31) / 32;
    numblocks_y = (ny + 31) / 32;

}

__global__ void ghost_setup(int nx, int ny, double* h) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    //bottom two will only execute on the edges
    //set the top boundary to equal the bottom
    if (i < nx && j == ny) {
        h(i, ny) = h(i, 0);
    }
    //set the right boundary to equal the left
    if (i == nx && j < ny) {
        h(nx, j) = h(0, j);
    }
}

__global__ void calc_derivs(int nx, int ny, double* dh, double* du, double* dv, double* h, double* u, double* v, double H, double g, double dx, double dy) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }

    dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
    du(i, j) = -g * dh_dx(i, j);
    dv(i, j) = -g * dh_dy(i, j);
}

__global__ void multistep(int nx, int ny, double a1, double a2, double a3, double* dh, double* du, double* dv, double* h, double* u, double* v,
    double* dh1, double* du1, double* dv1, double* dh2, double* du2, double* dv2, double dt)
{
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    
    if (i >= nx || j >= ny) {
        return;
    }

    h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
    u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
    v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;
}

__global__ void compute_boundary(int nx, int ny, double* h, double* u, double* v) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int j = blockIdx.y * 32 + threadIdx.y;
    if (i < nx && j == ny) {
        v(i, 0) = v(i, ny);
    }
    if (i == nx && j < ny) {
        u(0, j) = u(nx, j);
    }
}

void swap_buffers()
{
    double *tmp;

    tmp = dh2;
    dh2 = dh1;
    dh1 = dh;
    dh = tmp;

    tmp = du2;
    du2 = du1;
    du1 = du;
    du = tmp;

    tmp = dv2;
    dv2 = dv1;
    dv1 = dv;
    dv = tmp;
}

/**
 * This is your step function! Here, you will actually numerically solve the shallow
 * water equations. You should update the h, u, and v fields to be the solution after
 * one time step has passed.
 */
void step()
{
    //cuda apparently synchs between kernel calls so no need for the synchs

    /*
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max block dimensions: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    */

    //this block is max threads in a block
    dim3 blockDim(32, 32);
    dim3 gridDim(numblocks_x, numblocks_y);
    
    ghost_setup<<<gridDim, blockDim>>>(nx, ny, h);
    //cudaDeviceSynchronize();
    calc_derivs<<<gridDim, blockDim>>>(nx, ny, dh, du, dv, h, u, v, H, g, dx, dy);
    //cudaDeviceSynchronize();
    
    double a1, a2, a3;
    if (t == 0)
    {
        a1 = 1.0;
    }
    else if (t == 1)
    {
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
    }
    else
    {
        a1 = 23.0 / 12.0;
        a2 = -16.0 / 12.0;
        a3 = 5.0 / 12.0;
    }

    multistep<<<gridDim, blockDim>>>(
        nx, ny, a1, a2, a3, dh, du, dv, h, u, v,
        dh1, du1, dv1, dh2, du2, dv2, dt);
    compute_boundary<<<gridDim, blockDim>>>(nx, ny, h, u, v);
    //cudaDeviceSynchronize();
    swap_buffers();
    t++;
}

/**
 * This is your transfer function! You should copy the h field back to the host
 * so that the CPU can check the results of your computation.
 */
void transfer(double *h_host)
{
    size_t size = (nx + 1) * (ny + 1) * sizeof(double);
    cudaMemcpy(h_host, h, size, cudaMemcpyDeviceToHost);
}

/**
 * This is your finalization function! You should free all of the memory that you
 * allocated on the GPU here.
 */
void free_memory()
{
    cudaFree(h);
    cudaFree(u);
    cudaFree(v);

    cudaFree(dh);
    cudaFree(du);
    cudaFree(dv);

    cudaFree(dh1);
    cudaFree(du1);
    cudaFree(dv1);

    cudaFree(dh2);
    cudaFree(du2);
    cudaFree(dv2);
}