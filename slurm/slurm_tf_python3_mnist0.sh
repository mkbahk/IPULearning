#!/bin/bash
#SBATCH --job-name="MNIST-00"
#SBATCH -D .
#SBATCH --output=MNIST_%j.out
#SBATCH --error=MNIST_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:10:00
module load tensorflow/2.4.1 numpy/1.19.5 keras/2.4.3 matplotlib/3.3.4
srun python3 slurm_tf_python3_mnist.py
