#!/bin/bash
#PBS -l nodes=1:ppn=24:gpus=1
#PBS -l walltime=23:59:59
module load cuda
bin=/users/PAS0134/osu1601/israt/git/SDDMM_GPU/WH_shared_pascal/sddmm
K=32
# for item in /home/nisa/SDDMM/dataset/syn_*sorted.mtx; do
for item in /users/PAS0134/osu1601/israt/dense_syn/*.mtx; do 
   echo "Processing $item"
    log01=`$bin $item $K 32 128` 
    log02=`$bin $item $K 64 128` 
    # log02=`$bin $item $K 96 10000` 
    # log03=`$bin $item $K 48 10000` 
    # log04=`$bin $item $K 24 10000` 
    # log05=`$bin $item $K 12 10000`

    echo "$item,no tile,$log01,$log02,$log03,$log04,$log05"

    # log06=`$bin $item $K 192 25000` 
    # log07=`$bin $item $K 96 25000` 
    # log08=`$bin $item $K 48 25000` 
    # log09=`$bin $item $K 24 25000`
    # log10=`$bin $item $K 12 25000`

    # echo "Processing $item"
    # log11=`$bin $item $K 192 50000` 
    # log12=`$bin $item $K 96 50000` 
    # log13=`$bin $item $K 48 50000` 
    # log14=`$bin $item $K 24 50000` 
    # log15=`$bin $item $K 12 50000`

    # echo "Processing $item"
    # log16=`$bin $item $K 192 100000` 
    # log17=`$bin $item $K 96 100000` 
    # log18=`$bin $item $K 48 100000` 
    # log19=`$bin $item $K 24 100000` 
    # log20=`$bin $item $K 12 100000`

    # echo "Processing $item"
    # log21=`$bin $item $K 192 250000` 
    # log22=`$bin $item $K 96 250000` 
    # log23=`$bin $item $K 48 250000` 
    # log24=`$bin $item $K 24 250000` 
    # log25=`$bin $item $K 12 250000`

    # echo "Processing $item"
    # log26=`$bin $item $K 192 9999999` 
    # log27=`$bin $item $K 96 9999999` 
    # log28=`$bin $item $K 48 9999999` 
    # log29=`$bin $item $K 24 9999999` 
    # log30=`$bin $item $K 12 9999999`

    echo "
    $item, $log01,$log02" >> /users/PAS0134/osu1601/israt/dense_syn/sch1_denseSyn_k256_P100.csv
    # 10000, $log01,$log02,$log03,$log04, $log05,
    # 25000, $log06,$log07,$log08, $log09,$log10,
    # 50000, $log11, $log12,$log13, $log14,$log15,
    # 100000, $log16,$log17,$log18, $log19, $log20, 
    # 250000, ,$log21,$log22,$log23,$log24, $log25, 
    # 9999999, $log26,$log27,$log28,$log29, $log30" >> /users/PAS0134/osu1601/israt/dense_syn/sch1_denseSyn_k256_P100.csv
done
