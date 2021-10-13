#!/bin/sh
#SBATCH -J mpi
#SBATCH -o mpi_%j.out
#SBATCH -e mpi_%j.err
## 4 processes & 2 processes/node
#SBATCH -n 10
#SBATCH --tasks-per-node=4
mpirun ./HelloMPI
