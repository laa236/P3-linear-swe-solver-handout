#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "../common/common.hpp"
#include "../common/solver.hpp"

int nx, ny;
int rank, num_procs;
int local_nx;  // Number of rows per process
int *sendcounts, *displs;  // Arrays for scattering/gathering data

// Local arrays for this process
double *h, *u, *v, *dh, *du, *dv, *dh1, *du1, *dv1, *dh2, *du2, *dv2;
// Global arrays (only used by rank 0)
double *global_h, *global_u, *global_v;
double H, g, dx, dy, dt;

void init(double *h0, double *u0, double *v0, double length_, double width_, 
          int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_)
{
    rank = rank_;
    num_procs = num_procs_;
    nx = nx_;
    ny = ny_;
    
    // Calculate local grid size
    local_nx = nx / num_procs;
    if (rank < nx % num_procs) {
        local_nx++;
    }

    // Allocate local arrays
    h = (double*)calloc((local_nx+1) * (ny), sizeof(double));  // +1 for ghost rows and +1 ghost col
    u = (double*)calloc((local_nx+1) * (ny), sizeof(double));
    v = (double*)calloc((local_nx+1) * (ny), sizeof(double));

    dh = (double*)calloc(local_nx * ny, sizeof(double));
    du = (double*)calloc(local_nx * ny, sizeof(double));
    dv = (double*)calloc(local_nx * ny, sizeof(double));

    dh1 = (double*)calloc(local_nx * ny, sizeof(double));
    du1 = (double*)calloc(local_nx * ny, sizeof(double));
    dv1 = (double*)calloc(local_nx * ny, sizeof(double));

    dh2 = (double*)calloc(local_nx * ny, sizeof(double));
    du2 = (double*)calloc(local_nx * ny, sizeof(double));
    dv2 = (double*)calloc(local_nx * ny, sizeof(double));

    // Allocate arrays for scattering/gathering
    if (rank == 0) {
        sendcounts = (int*)malloc(num_procs * sizeof(int));
        displs = (int*)malloc(num_procs * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < num_procs; i++) {
            int cols = nx / num_procs;
            if (i < nx % num_procs) cols++;
            sendcounts[i] = cols * ny;
            displs[i] = offset;
            offset += cols * ny;
        }

        global_h = h0;
        global_u = u0;
        global_v = v0;


    MPI_Scatterv(global_h, sendcounts, displs, MPI_DOUBLE,
                 h, local_nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_u, sendcounts, displs, MPI_DOUBLE,
                 u, local_nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(global_v, sendcounts, displs, MPI_DOUBLE,
                 v, local_nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    }
    else{
    MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                 h, local_nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                 u, local_nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                 v, local_nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    H = H_;
    g = g_;
    dx = length_ / nx;
    dy = width_ / ny;
    dt = dt_;
}


void exchange_ghost_rows()
{

    int prev_rank = (rank - 1 + num_procs) % num_procs;
    int next_rank = (rank + 1) % num_procs;

    // Send bottom row to next process's top ghost row
    MPI_Sendrecv(h, ny, MPI_DOUBLE, prev_rank, 0,
                 h + local_nx*ny, ny, MPI_DOUBLE, next_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Repeat for u and v arrays
    MPI_Sendrecv(u, ny, MPI_DOUBLE, prev_rank, 2,
                 u + local_nx * ny, ny, MPI_DOUBLE, next_rank, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(v, ny, MPI_DOUBLE, prev_rank, 2,
                 v + local_nx * ny, ny, MPI_DOUBLE, next_rank, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void compute_dh()
{
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            // Using local indexing: i + j * local_n
            int idx = i*ny + j;
            dh[idx] = -H * (
                // du_dx needs to account for local indexing
                (u[(i + 1)*ny + j] - u[idx]) / dx +
                // dv_dy remains similar but uses local_nx+1 for stride
                (v[(i*ny)+((j+1)%ny)] - v[idx]) / dy
            );
        }
    }
}

void compute_du()
{
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            int idx = i*ny + j;
            // Using local indexing for h array
            du[idx] = -g * (
                (h[(i + 1)*ny + j] - h[idx]) / dx
            );
        }
    }
}

void compute_dv()
{
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {  // ny-1 because we're looking one ahead in y
            int idx = i*ny + j;
            // Using local indexing for h array
            dv[idx] = -g * (
                (h[(i*ny)+((j+1)%ny)] - h[idx]) / dy
            );
        }
    }
}

// void compute_ghost_horizontal()
// {
//     // Handle periodic boundaries for the first and last processes
//     if (rank == 0) {
//         for (int j = 0; j < ny; j++) {
//             // Copy from last real cell to first ghost cell
//             h[j] = h[j + (local_nx - 1) * ny];
//             u[j] = u[j + (local_nx - 1) * ny];
//             v[j] = v[j + (local_nx - 1) * ny];
//         }
//     }
//     if (rank == num_procs - 1) {
//         for (int j = 0; j < ny; j++) {
//             // Copy from first real cell to last ghost cell
//             h[j + local_nx * ny] = h[j];
//             u[j + local_nx * ny] = u[j];
//             v[j + local_nx * ny] = v[j];
//         }
//     }
// }

void multistep(double a1, double a2, double a3)
{
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < ny; j++) {
            int idx = i*ny + j;
            
            // Update h
            h[idx] += (a1 * dh[idx] + a2 * dh1[idx] + a3 * dh2[idx]) * dt;
            u[(i + 1) + j * (local_nx + 1)] += 
                    (a1 * du[idx] + a2 * du1[idx] + a3 * du2[idx]) * dt;
            
            // Update v (offset by 1 in y-direction) // Don't update the last v point
            v[(i*ny)+((j+1)%ny)] += 
                    (a1 * dv[idx] + a2 * dv1[idx] + a3 * dv2[idx]) * dt;
        }
    }
     int prev_rank = (rank - 1 + num_procs) % num_procs;
    int next_rank = (rank + 1) % num_procs;
    MPI_Sendrecv(u+(local_nx*ny), ny, MPI_DOUBLE, next_rank, 2,
                 u, ny, MPI_DOUBLE, prev_rank, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h+(local_nx*ny), ny, MPI_DOUBLE, next_rank, 0,
                 h, ny, MPI_DOUBLE, prev_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void swap_buffers()
{
    double *tmp;

    tmp = dh2; dh2 = dh1; dh1 = dh; dh = tmp;
    tmp = du2; du2 = du1; du1 = du; du = tmp;
    tmp = dv2; dv2 = dv1; dv1 = dv; dv = tmp;
}

int t = 0;

void step()
{
    // Print values of h, u, v before calculations
        // printf("Before step:\n");
        // for (int i = 0; i < local_nx; i++) {
        //     for (int j = 0; j < ny; j++) {
        //         int idx = i * ny + j;
        //         printf("Rank %d h[%d][%d] = %f, u[%d][%d] = %f, v[%d][%d] = %f\n", 
        //                rank, i, j, h[idx], i, j, u[idx], i, j, v[idx]);
        //     }
        // }

    // Exchange ghost rows with neighboring processes
    exchange_ghost_rows();

    // Compute horizontal ghost cells
    // compute_ghost_horizontal();

    // Compute derivatives
    compute_dh();
    compute_du();
    compute_dv();
    // Set multistep coefficients
    double a1, a2, a3;
    if (t == 0) {
        a1 = 1.0;
        a2 = a3 = 0.0;
    } else if (t == 1) {
        a1 = 3.0 / 2.0;
        a2 = -1.0 / 2.0;
        a3 = 0.0;
    } else {
        a1 = 23.0 / 12.0;
        a2 = -16.0 / 12.0;
        a3 = 5.0 / 12.0;
    }

    // Compute next time step
    multistep(a1, a2, a3);

    // Print values of h, u, v after calculations
        // printf("After step:\n");
        // for (int i = 0; i < local_nx; i++) {
        //     for (int j = 0; j < ny; j++) {
        //         int idx = i * ny + j;
        //         printf("Rank %d h[%d][%d] = %f, u[%d][%d] = %f, v[%d][%d] = %f\n", 
        //                rank, i, j, h[idx], i, j, u[idx], i, j, v[idx]);
        //     }
        // }

    // Compute boundaries
    // compute_boundaries_horizontal();

    // Swap derivative buffers
    swap_buffers();
    t++;
}

void transfer(double *h_out)
{

    // Gather results back to rank 0
    MPI_Gatherv(h,                  // Send buffer (skip ghost row)
                local_nx * ny,            // Number of elements to send
                MPI_DOUBLE,               // Data type
                h_out,                    // Receive buffer (only used at root)
                sendcounts,               // Array of receive counts
                displs,                   // Array of displacements
                MPI_DOUBLE,               // Data type
                0,                        // Root process
                MPI_COMM_WORLD);


    // Print h, u, v and timestep t (only on rank 0)
    // if (rank == 0) {
    //     printf("Timestep: %d\n", t);
    //     for (int i = 0; i < nx; i++) {
    //         for (int j = 0; j < ny; j++) {
    //             int idx = i + j * nx;
    //             printf("h[%d][%d] = %f, u[%d][%d] = %f, v[%d][%d] = %f\n", 
    //                    i, j, h_out[idx], i, j, global_u[idx], i, j, global_v[idx]);
    //         }
    //     }
    // }
}

void free_memory()
{
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }

    free(h); free(u); free(v);
    free(dh); free(du); free(dv);
    free(dh1); free(du1); free(dv1);
    free(dh2); free(du2); free(dv2);
}