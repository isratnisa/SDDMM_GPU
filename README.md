Sampled Dense-Dense Matrix Multiplication (SDDMM) is a primitive that has been shown to be usable as a core component in reformulations of many machine learning algorithms. It requires the computation of the product of two input dense matrices but only at locations of the result matrix corresponding to nonzero entries in a sparse third input matrix. In this work, we develop of cuSDDMM, a multi-node GPU-accelerated implementation for SDDMM. This work is published under the title "Sampled Dense Matrix Multiplication for High-Performance Machine Learning" (https://ieeexplore.ieee.org/abstract/document/8638042) in 2018 IEEE 25th International Conference on High Performance Computing (HiPC).


## Input format

Suports Matrix Market (https://sparse.tamu.edu/about) input format. 

## Build requirements:
- GCC Compiler 
- CUDA SDK


## Build 

`$ make`  

## Run

`$ ./sddmm input K tile_size_X tile_size_Y`

Example:
`./sddmm nytimes.mtx 128 192 50000`

       


