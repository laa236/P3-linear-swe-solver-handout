#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef CUDA
#include <cuda_runtime.h>
#endif

#ifdef MPI
#include <mpi.h>
#endif

#include "scenarios.hpp"
#include "solver.hpp"

int main(int argc, char **argv)
{

    int length = 1.0e7, width = 1.0e7, nx = 256, ny = 258, num_iterations = 1000, save_iter = 20;
    double depth = 100.0, g = 1.0, r = 2.0e5, max_height = 10.0, dt = 100.0;

    char scenario[256] = "water_drop", output_file[256];
    bool output = false;

    int rank = 0, num_procs = 1;

#ifdef MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    int cur_arg = 1;
    int num_args = argc - 1;

    while (num_args > 0)
    {
        if (num_args == 1)
        {
            fprintf(stderr, "Missing argument value for %s\n", argv[cur_arg]);
            return 1;
        }

        if (strcmp(argv[cur_arg], "--length") == 0)
        {
            length = atoi(argv[cur_arg + 1]);
        }
        else if (strcmp(argv[cur_arg], "--width") == 0)
        {
            width = atoi(argv[cur_arg + 1]);
        }
        else if (strcmp(argv[cur_arg], "--nx") == 0)
        {
            nx = atoi(argv[cur_arg + 1]);
        }
        else if (strcmp(argv[cur_arg], "--ny") == 0)
        {
            ny = atoi(argv[cur_arg + 1]);
        }
        else if (strcmp(argv[cur_arg], "--scenario") == 0)
        {
            strcpy(scenario, argv[cur_arg + 1]);
        }
        else if (strcmp(argv[cur_arg], "--radius") == 0)
        {
            r = atof(argv[cur_arg + 1]);
        }
        else if (strcmp(argv[cur_arg], "--height") == 0)
        {
            max_height = atof(argv[cur_arg + 1]);
        }
        else if (strcmp(argv[cur_arg], "--num_iter") == 0)
        {
            num_iterations = atoi(argv[cur_arg + 1]);
        }
        else if (strcmp(argv[cur_arg], "--output") == 0)
        {
            strcpy(output_file, argv[cur_arg + 1]);
            output = true;
        }
        else if (strcmp(argv[cur_arg], "--dt") == 0)
        {
            dt = atof(argv[cur_arg + 1]);
        }
        else if (strcmp(argv[cur_arg], "--save_iter") == 0)
        {
            save_iter = atoi(argv[cur_arg + 1]);
        }
        else
        {
            fprintf(stderr, "Unknown argument: %s\n", argv[cur_arg]);
            return 1;
        }

        cur_arg += 2;
        num_args -= 2;
    }

    double *h = nullptr;
    double *u = nullptr;
    double *v = nullptr;

    if (rank == 0)
    {
        h = (double *)calloc((nx + 1) * (ny + 1), sizeof(double));
        u = (double *)calloc((nx + 2) * ny, sizeof(double));
        v = (double *)calloc(nx * (ny + 2), sizeof(double));

        if (strcmp(scenario, "water_drop") == 0)
        {
            water_drop(length, width, nx, ny, r, max_height, h, u, v);
        }
        else if (strcmp(scenario, "dam_break") == 0)
        {
            dam_break(length, width, nx, ny, r, max_height, h, u, v);
        }
        else if (strcmp(scenario, "wave") == 0)
        {
            wave(length, width, nx, ny, max_height, h, u, v);
        }
        else if (strcmp(scenario, "river") == 0)
        {
            river(length, width, nx, ny, max_height, h, u, v);
        }
        else
        {
            fprintf(stderr, "Unknown scenario: %s\n", scenario);
            return 1;
        }
    }

    clock_t init_start = clock();

    init(h, u, v, length, width, nx, ny, depth, g, dt, rank, num_procs);

    clock_t init_end = clock();
    fprintf(stderr, "Initialization time for rank %d: %f\n", rank, (double)(init_end - init_start) / CLOCKS_PER_SEC);

    FILE *fptr;

    if (rank == 0 && output)
    {
        fptr = fopen(output_file, "w");

        fwrite(&length, sizeof(int), 1, fptr);
        fwrite(&width, sizeof(int), 1, fptr);

        fwrite(&nx, sizeof(int), 1, fptr);
        fwrite(&ny, sizeof(int), 1, fptr);

        fwrite(&depth, sizeof(double), 1, fptr);
        fwrite(&g, sizeof(double), 1, fptr);
        fwrite(&r, sizeof(double), 1, fptr);
        fwrite(&max_height, sizeof(double), 1, fptr);
        fwrite(&dt, sizeof(double), 1, fptr);

        fwrite(&num_iterations, sizeof(int), 1, fptr);
        fwrite(&save_iter, sizeof(int), 1, fptr);
    }

    clock_t start = clock();

    for (int i = 0; i < num_iterations; i++)
    {
        if (output && i % save_iter == 0)
        {
            transfer(h);

            if (rank == 0)
            {
                fwrite(h, sizeof(double), (nx + 1) * (ny + 1), fptr);
            }
        }

        step();
    }

#ifdef CUDA
    cudaDeviceSynchronize();
#endif

    clock_t end = clock();
    fprintf(stderr, "Execution time for rank %d: %f\n", rank, (double)(end - start) / CLOCKS_PER_SEC);

    clock_t free_start = clock();
    free_memory();
    clock_t free_end = clock();
    fprintf(stderr, "Free memory time for rank %d: %f\n", rank, (double)(free_end - free_start) / CLOCKS_PER_SEC);

    if (rank == 0)
    {
        free(h);
        free(u);
        free(v);

        if (output)
        {
            fclose(fptr);
        }
    }

#ifdef MPI
    MPI_Finalize();
#endif

    return 0;
}
