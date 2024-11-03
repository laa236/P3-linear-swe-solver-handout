#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import math, re, sys
from subprocess import run, PIPE

num_iters = 10000

serial_sizes = [ 100, 250, 500, 750, 1000 ]
gpu_sizes = [ 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 ]

mpi_size_strong = 2048
mpi_sizes_weak = [ 512, 1024, 2048, 4096 ]

mpi_tasks_per_node = 128
mpi_ranks = [ 8, 32, 128, 512 ]

def time(time_type, executable_name, sizes):
    sizes_arr = np.array(sizes)
    times = np.zeros(len(sizes))

    for i in range(len(sizes_arr)):
        size = sizes_arr[i]

        print(f"Timing {time_type} size {size}")
        command = [executable_name, "--nx", str(size), "--ny", str(size), "--num_iter", str(num_iters)]
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        
        init_time = re.search(r"Initialization time for rank 0: (\d+\.\d+)", result.stderr).group(1)
        exec_time = re.search(r"Execution time for rank 0: (\d+\.\d+)", result.stderr).group(1)
        free_time = re.search(r"Free memory time for rank 0: (\d+\.\d+)", result.stderr).group(1)

        total_time = float(init_time) + float(exec_time) + float(free_time)
        times[i] = total_time

        print(f"Total time: {total_time}")

    plt.plot(sizes_arr, times)
    plt.xlabel("Size")
    plt.ylabel("Time (s)")
    plt.savefig(f"{executable_name}_timing.png")

def time_mpi_strong_scaling(executable_name, max_size, ranks_list):
    ranks_arr = np.array(ranks_list)
    times = np.zeros(len(ranks_list))

    for i in range(len(ranks_arr)):
        ranks = ranks_arr[i]
        tasks = min(ranks, mpi_tasks_per_node)
        nodes = math.ceil(ranks / tasks)

        print(f"Timing MPI strong scaling with size {max_size} and {ranks} ranks (divided in {nodes} nodes with {tasks} tasks each)")
        command = ["srun", f"--nodes={nodes}", f"--ntasks-per-node={tasks}", executable_name, "--nx", str(max_size), "--ny", str(max_size), "--num_iter", str(num_iters)]
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

        for r in range(ranks):
            init_time = re.search(f"Initialization time for rank {r}: (\d+\.\d+)", result.stderr).group(1)
            exec_time = re.search(f"Execution time for rank {r}: (\d+\.\d+)", result.stderr).group(1)
            free_time = re.search(f"Free memory time for rank {r}: (\d+\.\d+)", result.stderr).group(1)

            total_time = float(init_time) + float(exec_time) + float(free_time)
            times[i] = max(times[i], total_time)

            # print(f"Total time for rank {r}: {total_time}")
        
        print(f"Total time: {times[i]}")
        print(f"Ideal time: {times[0] / (ranks / ranks_arr[0])}")

    plt.plot(ranks_arr, times, marker="o")
    plt.plot(ranks_arr, times[0] / (ranks / ranks_arr[0]), linestyle="--")
    plt.xlabel("Nodes")
    plt.ylabel("Time (s)")
    plt.savefig(f"{executable_name}_strong_scaling_timing.png")

def time_mpi_weak_scaling(executable_name, sizes, ranks_list):
    sizes_arr = np.array(sizes)
    ranks_arr = np.array(ranks_list)
    times = np.zeros(len(sizes))

    for i in range(len(sizes_arr)):
        size = sizes_arr[i]

        ranks = ranks_arr[i]
        tasks = min(ranks, mpi_tasks_per_node)
        nodes = math.ceil(ranks / tasks)

        print(f"Timing MPI weak scaling with size {size} and {ranks} ranks (divided in {nodes} nodes with {tasks} tasks each)")

        command = ["srun", f"--nodes={nodes}", f"--ntasks-per-node={tasks}", executable_name, "--nx", str(size), "--ny", str(size), "--num_iter", str(num_iters)]
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

        for r in range(ranks):
            init_time = re.search(f"Initialization time for rank {r}: (\d+\.\d+)", result.stderr).group(1)
            exec_time = re.search(f"Execution time for rank {r}: (\d+\.\d+)", result.stderr).group(1)
            free_time = re.search(f"Free memory time for rank {r}: (\d+\.\d+)", result.stderr).group(1)

            total_time = float(init_time) + float(exec_time) + float(free_time)
            times[i] = max(times[i], total_time)

            # print(f"Total time for rank {r}: {total_time}")
            
        print(f"Total time: {times[i]}")
        print(f"Ideal time: {times[0]}")

    plt.plot(ranks_arr, times, marker="o")
    plt.plot(ranks_arr, times[0] * np.ones(shape=(len(ranks_arr))), linestyle="--")
    plt.xlabel("Nodes")
    plt.ylabel("Time (s)")
    plt.savefig(f"{executable_name}_weak_scaling_timing.png")

if __name__ == "__main__":
    time_type = sys.argv[1]
    exec_name = sys.argv[2]

    if time_type == "serial":
        time(time_type, exec_name, serial_sizes)
    elif time_type == "gpu":
        time(time_type, exec_name, gpu_sizes)
    elif time_type == "mpi_strong":
        time_mpi_strong_scaling(exec_name, mpi_size_strong, mpi_ranks)
    elif time_type == "mpi_weak":
        time_mpi_weak_scaling(exec_name, mpi_sizes_weak, mpi_ranks)
