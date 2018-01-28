#!/bin/bash
#SBATCH --time=23:30:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out
#SBATCH -p batch-gpu
module load cuda
bin=/home/nisa/SDDMM/big_tile/sddmm
K=256
# for item in /home/nisa/SDDMM/dataset/syn_*sorted.mtx; do
for item in /home/nisa/SpMM/dataset/*.mtx; do 
   echo "Processing $item"
    log01=`$bin $item $K 192 10000` 
    log02=`$bin $item $K 96 10000` 
    log03=`$bin $item $K 48 10000` 
    log04=`$bin $item $K 24 10000` 
    # log05=`$bin $item $K 12 10000`

    echo "$item,no tile,$log01,$log02,$log03,$log04,$log05"

    log06=`$bin $item $K 192 25000` 
    log07=`$bin $item $K 96 25000` 
    log08=`$bin $item $K 48 25000` 
    log09=`$bin $item $K 24 25000`
    # log10=`$bin $item $K 12 25000`

    echo "Processing $item"
    log11=`$bin $item $K 192 50000` 
    log12=`$bin $item $K 96 50000` 
    log13=`$bin $item $K 48 50000` 
    log14=`$bin $item $K 24 50000` 
    # log15=`$bin $item $K 12 50000`

    echo "Processing $item"
    log16=`$bin $item $K 192 100000` 
    log17=`$bin $item $K 96 100000` 
    log18=`$bin $item $K 48 100000` 
    log19=`$bin $item $K 24 100000` 
    # log20=`$bin $item $K 12 100000`

    echo "Processing $item"
    log21=`$bin $item $K 192 250000` 
    log22=`$bin $item $K 96 250000` 
    log23=`$bin $item $K 48 250000` 
    log24=`$bin $item $K 24 250000` 
    # log25=`$bin $item $K 12 250000`

    echo "Processing $item"
    log26=`$bin $item $K 192 9999999` 
    log27=`$bin $item $K 96 9999999` 
    log28=`$bin $item $K 48 9999999` 
    log29=`$bin $item $K 24 9999999` 
    # log30=`$bin $item $K 12 9999999`

    echo "
    $item, 
    10000, $log01,$log02,$log03,$log04, $log05,
    25000, $log06,$log07,$log08, $log09,$log10,
    50000, $log11, $log12,$log13, $log14,$log15,
    100000, $log16,$log17,$log18, $log19, $log20, 
    250000, ,$log21,$log22,$log23,$log24, $log25, 
    9999999, $log26,$log27,$log28,$log29, $log30" >> /home/nisa/SDDMM/big_tile/sch1_graph_k256_k80c.csv
done
