#!/bin/bash
#PBS -l nodes=16:ppn=28:gpus=1
#PBS -l walltime=0:59:59

module load cuda/8.0.44
nvidia-smi

bin=/users/PAS0134/osu1601/israt/git/SDDMM_GPU/big_tile_MPI/sddmm

# make && nvprof --profile-child-processes  --metrics all mpiexec -n 1 -ppn 1 /users/PAS0134/osu1601/israt/git/SDDMM_GPU/big_tile_MPI/sddmm ~/israt/graph_data/nips.mtx 32 256 999999 &> nvporf_1node_on2nodemachine.csv

dataset=~/israt/graph_data/nytimes.mtx

for K in 32 64 128 256 512 1024; do
	log1=`mpiexec -n 16 -ppn 1 $bin $dataset $K 256 9999999`
	echo $dataset, $K, $log1 >> mnew_pi_data_16node.csv
done

# dataset=~/israt/graph_data/pubmed_row_sorted.mtx
# for K in 32 64 128 256 512 1024; do
# 	log2=`mpiexec -n 16 -ppn 1 $bin $dataset $K 256 9999999`
# 	echo $dataset, $K, $log2 >> mpi_data_16node.csv
# done