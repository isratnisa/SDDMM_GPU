echo "processing"

nvprof --metrics all  --events all ./sddmm ~/israt/syn_500K_12500000_random.mtx_sorted.mtx 256 256 9999999 &> syn_005_k256_slc_32.csv
nvprof --metrics all  --events all ./sddmm ~/israt/syn_500K_12500000_random.mtx_sorted.mtx 256 128 9999999 &> syn_005_k256_slc_64.csv
nvprof --metrics all  --events all  ./sddmm ~/israt/syn_500K_12500000_random.mtx_sorted.mtx 256 64 9999999 &> syn_005_k256_slc_128.csv
nvprof --metrics all  --events all ./sddmm ~/israt/syn_500K_12500000_random.mtx_sorted.mtx 256 32 9999999 &> syn_005_k256_slc_256.csv

echo "processing"
nvprof --metrics all  --events all ./sddmm ~/israt/syn_500K_125000000_random.mtx_sorted.mtx 256 256 9999999 &> syn_05_k256_slc_32.csv
nvprof --metrics all  --events all ./sddmm ~/israt/syn_500K_125000000_random.mtx_sorted.mtx 256 128 9999999 &> syn_05_k256_slc_64.csv
nvprof --metrics all  --events all  ./sddmm ~/israt/syn_500K_125000000_random.mtx_sorted.mtx 256 64 9999999 &> syn_05_k256_slc_128.csv
nvprof --metrics all  --events all ./sddmm ~/israt/syn_500K_125000000_random.mtx_sorted.mtx 256 32 9999999 &> syn_05_k256_slc_256.csv