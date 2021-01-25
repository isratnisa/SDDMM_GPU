#!/bin/bash
#PBS -l nodes=2:ppn=12:gpus=1
#PBS -l walltime=0:5:59

module load cuda/8.0.44
nvidia-smi

bin=/users/PAS0134/osu1601/israt/git/SDDMM_GPU/big_tile_MPI

# make && nvprof --profile-child-processes  --metrics all mpiexec -n 1 -ppn 1 /users/PAS0134/osu1601/israt/git/SDDMM_GPU/big_tile_MPI/sddmm ~/israt/graph_data/nips.mtx 32 256 999999 &> nvporf_1node_on2nodemachine.csv



make && mpiexec -n 2 -ppn 1 /users/PAS0134/osu1601/israt/git/SDDMM_GPU/big_tile_MPI/sddmm ~/israt/graph_data/nytimes.mtx 32 256 9999999