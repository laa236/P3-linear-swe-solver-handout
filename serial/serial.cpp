#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

// Here we hold the number of cells we have in the x and y directions
int nx, ny;

// This is where all of our points are. We need to keep track of our active
// height and velocity grids, but also the corresponding derivatives. The reason
// we have 2 copies for each derivative is that our multistep method uses the
// derivative from the last 2 time steps.
double *h, *u, *v, *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;
double H, g, dx, dy, dt;
__m256d dtv;

/**
 * This is your initialization function!
 */
void init(double *h0, double *u0, double *v0, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    // We set the pointers to the arrays that were passed in
    h = h0;
    u = u0;
    v = v0;

    nx = nx_;
    ny = ny_;

    // We allocate memory for the derivatives
    dh = (double *)calloc(nx * ny, sizeof(double));
    du = (double *)calloc(nx * ny, sizeof(double));
    dv = (double *)calloc(nx * ny, sizeof(double));

    dh1 = (double *)calloc(nx * ny, sizeof(double));
    du1 = (double *)calloc(nx * ny, sizeof(double));
    dv1 = (double *)calloc(nx * ny, sizeof(double));

    dh2 = (double *)calloc(nx * ny, sizeof(double));
    du2 = (double *)calloc(nx * ny, sizeof(double));
    dv2 = (double *)calloc(nx * ny, sizeof(double));

    H = H_;
    g = g_;

    dx = length_ / nx;
    dy = width_ / ny;

    dt = dt_;

    dtv = _mm256_set1_pd(dt);
}

/**
 * This is your step function!
 */
int t = 0;

void step()
{
    for (int j = 0; j < ny; j++)
    {
        h(nx, j) = h(0, j);
    }
    for (int i = 0; i < nx; i++)
    {
        h(i, ny) = h(i, 0);
    }
    
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            dh(i, j) = -H * (du_dx(i, j) + dv_dy(i, j));
            du(i, j) = -g * dh_dx(i, j);
            dv(i, j) = -g * dh_dy(i, j);
        }
    }

    double a1, a2, a3;
    __m256d a1v, a2v, a3v;
    if (t == 0)
    {
        a1 = 1.0;
        a1v = _mm256_set1_pd(1.0);
    }
    else if (t == 1)
    {
        a1 = 3.0 / 2.0;
        a1v = _mm256_set1_pd(1.5);
        a2 = -1.0 / 2.0;
        a2v = _mm256_set1_pd(-0.5);
    }
    else
    {
        a1 = 23.0 / 12.0;
        a1v = _mm256_set1_pd(23.0 / 12.0);
        a2 = -16.0 / 12.0;
        a2v = _mm256_set1_pd(-16.0 / 12.0);
        a3 = 5.0 / 12.0;
        a3v = _mm256_set1_pd(5.0 / 12.0);
    }
    

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            if (j + 3 < ny) {
                __m256d hv = _mm256_loadu_pd(&h(i, j));
                __m256d dhv = _mm256_loadu_pd(&dh(i, j));
                __m256d dh1v = _mm256_loadu_pd(&dh1(i, j));
                __m256d dh2v = _mm256_loadu_pd(&dh2(i, j));

                __m256d result = _mm256_fmadd_pd(a1v, dhv, _mm256_fmadd_pd(a2v, dh1v, _mm256_mul_pd(a3v, dh2v)));
                result = _mm256_fmadd_pd(result, dtv, hv);
                _mm256_storeu_pd(&h(i, j), result);
                j += 3;
            } else {
                h(i, j) += (a1 * dh(i, j) + a2 * dh1(i, j) + a3 * dh2(i, j)) * dt;
            }
        }

        for (int j = 0; j < ny; j++)
        {
            u(i + 1, j) += (a1 * du(i, j) + a2 * du1(i, j) + a3 * du2(i, j)) * dt;
            v(i, j + 1) += (a1 * dv(i, j) + a2 * dv1(i, j) + a3 * dv2(i, j)) * dt;
        }
    }

    for (int j = 0; j < ny; j++)
    {
        u(0, j) = u(nx, j);
    }
    for (int i = 0; i < nx; i++)
    {
        v(i, 0) = v(i, ny);
    }

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

    t++;
}

/**
 * This is your transfer function! Since everything is running on the same node,
 * you don't need to do anything here.
 */
void transfer(double *h)
{
    return;
}

/**
 * This is your finalization function! Free whatever memory you've allocated.
 */
void free_memory()
{
    free(dh);
    free(du);
    free(dv);

    free(dh1);
    free(du1);
    free(dv1);

    free(dh2);
    free(du2);
    free(dv2);
}