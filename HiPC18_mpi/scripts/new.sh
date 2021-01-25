#!/bin/bash
#PBS -l nodes=1:ppn=24:gpus=1
#PBS -l walltime=23:59:59
module load cuda
bin=/users/PAS0134/osu1601/israt/git/SDDMM_GPU/big_tile/sddmm
# path=/users/PAS0134/osu1601/israt/graph_data
path=/users/PAS0134/osu1601/israt/graph_data/
K=32
counter=0
# mat=(2cubes_sphere.asym.mtx	cage12.mtx	cant.asym.mtx	cit-HepPh.mtx	com-amazon.ungraph.mtx	com-dblp.ungraph.mtx	com-youtube.ungraph.mtx	consph.asym.mtx	cop20k_A.asym.mtx	email-Enron.mtx	facebook_combined.mtx	filter3D.asym.mtx	hood.asym.mtx	loc-gowalla_edges.mtx	m133-b3.mtx	mac_econ_fwd500.mtx	majorbasis.mtx	mario002.asym.mtx	mc2depi.mtx	nytimes.mtx	offshore.asym.mtx	patents_main.mtx	pdb1HYS.asym.mtx	poisson3Da.mtx	pubmed_row_sorted.mtx	pwtk.asym.mtx	rma10.mtx	roadNet-CA.mtx	scircuit.mtx	shipsec1.asym.mtx	webbase-1M.mtx	web-BerkStan.mtx	web-Google.mtx	web-NotreDame.mtx)
# TILE=(40000	45000	15000	25000	285000	210000	335000	15000	35000	30000	5000	30000	35000	70000	110000	90000	60000	135000	180000	10000	65000	160000	10000	10000	25000	30000	15000	420000	85000	25000	285000	125000	205000	135000)
# for item in ${mat[@]}; do 
# 	echo "$item ${TILE[$counter]}"
# 	log01=`$bin $path/$item $K 256 ${TILE[$counter]}` 
#     log02=`$bin $path/$item $K 128 ${TILE[$counter]}` 
#     log03=`$bin $path/$item $K 64 ${TILE[$counter]}` 
#     log04=`$bin $path/$item $K 32 ${TILE[$counter]}` 
#     log05=`$bin $path/$item $K 16 ${TILE[$counter]}` 

#     echo "
#     $item, ${TILE[$counter]}, $log01,$log02,$log03,$log04, $log05" >> /users/PAS0134/osu1601/israt/git/SDDMM_GPU/big_tile/bestT_sch1_graph_k"$K"_P100.csv
#     let counter=counter+1
# done

echo "K = $K"
mat=(kos.mtx nips.mtx  enron.mtx)   
for item in ${mat[@]}; do 
    TILE=(16500 10000  35000)
    echo "Processing $item"
    log01=`$bin $path/$item $K 256 10000` 
    # log02=`$bin $path/$item $K 128 10000` 
    # log03=`$bin $path/$item $K 64 10000` 
    # log04=`$bin $path/$item $K 32 10000` 
    # log05=`$bin $path/$item $K 16 10000`

    echo "$item,10 tile,$log01,$log02,$log03,$log04,$log05"

    log06=`$bin $path/$item $K 256 25000` 
    # log07=`$bin $path/$item $K 128 25000` 
    # log08=`$bin $path/$item $K 64 25000` 
    # log09=`$bin $path/$item $K 32 25000`
    # log10=`$bin $path/$item $K 16 25000`

    echo "Processing $item"
    log11=`$bin $path/$item $K 256 50000` 
    # log12=`$bin $path/$item $K 128 50000` 
    # log13=`$bin $path/$item $K 64 50000` 
    # log14=`$bin $path/$item $K 32 50000` 
    # log15=`$bin $path/$item $K 16 50000`

    echo "Processing $item"
    log16=`$bin $path/$item $K 256 ${TILE[$counter]}` 
    # log17=`$bin $path/$item $K 128 ${TILE[$counter]}` 
    # log18=`$bin $path/$item $K 64 ${TILE[$counter]}` 
    # log19=`$bin $path/$item $K 32 ${TILE[$counter]}` 
    # log20=`$bin $path/$item $K 16 ${TILE[$counter]}`

    # echo "Processing $item"
    # log16=`$bin $path/$item $K 256 100000` 
    # log17=`$bin $path/$item $K 128 100000` 
    # log18=`$bin $path/$item $K 64 100000` 
    # log19=`$bin $path/$item $K 32 100000` 
    # log20=`$bin $path/$item $K 16 100000`

    # echo "Processing $item"
    # log21=`$bin $path/$item $K 256 250000` 
    # log22=`$bin $path/$item $K 128 250000` 
    # log23=`$bin $path/$item $K 64 250000` 
    # log24=`$bin $path/$item $K 32 250000` 
    # log25=`$bin $path/$item $K 16 250000`

    # echo "Processing $item"
    # log26=`$bin $path/$item $K 256 9999999` 
    # log27=`$bin $path/$item $K 128 9999999` 
    # log28=`$bin $path/$item $K 64 9999999` 
    # log29=`$bin $path/$item $K 32 9999999` 
    # log30=`$bin $path/$item $K 16 9999999`

    echo "
    $item, 
    10000, $log01,$log02,$log03,$log04, $log05,
    25000, $log06,$log07,$log08, $log09,$log10,
    50000, $log11, $log12,$log13, $log14,$log15,
    ${TILE[$counter]}, $log16,$log17,$log18, $log19, $log20" >> /users/PAS0134/osu1601/israt/git/SDDMM_GPU/big_tile/kos_nips_enron.csv
    let counter=counter+1
done