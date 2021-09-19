#!/bin/bash
#SBATCH --job-name=hello-world-mpi
#SBATCH --output=/mnt/graphcore-ipu-demo/hellompi/helloworld_mpi-%j.out
#SBATCH --error=/mnt/graphcore-ipu-demo/hellompi/helloworld_mpi-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --time=72:00:00
#SBATCH --mem=1G

PATH=/mnt/graphcore-ipu-demo/hellompi/:$PATH
WORKDIR=/mnt/graphcore-ipu-demo/hellompi

cd $WORKDIR
mpiexec -np 8 ./hellompi
