Sampled Dense-Dense Matrix Multiplication (SDDMM) is a core component in reformulations of many machine learning algorithms. It requires the computation of the product of two input dense matrices but only at locations of the result matrix corresponding to nonzero entries in a sparse third input matrix. In this work, we develop of cuSDDMM, a multi-node GPU-accelerated implementation for SDDMM. This work is published under the title "Sampled Dense Matrix Multiplication for High-Performance Machine Learning" (https://ieeexplore.ieee.org/abstract/document/8638042) in 2018 IEEE 25th International Conference on High Performance Computing (HiPC). Please refer to HiPC18 folder.


## Input format

Supoprts Matrix Market (https://sparse.tamu.edu/about) input format. Currently support row sorted matrices. Example avaialable in dataset folder. Conversion code from col sorted to row sorted can be found inside sddmm.cu

## Build requirements:
- GCC Compiler 
- CUDA SDK
- OpenMP


## Build 

`cd HiPC18/src`
`$ make`  

## Run

`$ ./sddmm ../../dataset/nips.mtx K tile_size_X tile_size_Y`

Example:
`./sddmm nytimes.mtx 128 192 50000`

       


