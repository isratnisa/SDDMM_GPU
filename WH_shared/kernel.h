__global__ void comp_kernel_COO(int const* __restrict__ row_ind, int const* __restrict__ col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
  int nnz, int n_rows, int n_cols, int k, int *active_row, int *tile_limIdx, int * lastIdx_block_tile, 
  int *no_block_tile, int max_active_block, int tile_size, int max_active_row){
    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 1;
    int tile_st = tile_limIdx[blockIdx.x];
    int tile_end = tile_limIdx[blockIdx.x+1];
    //__shared__ float sh_c[180*32];
    __shared__ float sh_c[96*32];
    __shared__ float sh_r[96*32];
    int tile_no = blockIdx.x;
    int sh_tile = 96;
    int sh_tile_r = 96;
    int WARP_ID = tId >> 5;
    int tid_in_WARP = tId & 31;
    int step = blockDim.x >> 5;
    int o_t = 0;
    //for (int o_t = 0; o_t < k; o_t+=32)
    {
        unsigned int c = tId + tile_st;
        int tile  = blockIdx.x;// col/tile_size;
        {   int t = tid_in_WARP;
            for (int i = WARP_ID; i < sh_tile && (i +(tile_no*sh_tile)) < n_cols; i+=step){
                sh_c[i *  k + t] = v[(tile * sh_tile + i) *k + t];
            }
        }           
        __syncthreads();

        int g_n_blocks = no_block_tile[blockIdx.x]; int n_blocks=0;
        if(blockIdx.x == 0) n_blocks =  no_block_tile[blockIdx.x];
        else n_blocks =  no_block_tile[blockIdx.x] - no_block_tile[blockIdx.x-1];
        //if(blockIdx.x == 1 && tId ==0 ) printf("GPU %d : %d %d %d %d\n",  tile, n_blocks, tile_st, tile_end );

        int block_st = tile_st;// lastIdx_block_tile[g_n_blocks + block-1];
        for (int block = 0; block < n_blocks ; block++){          
            int block_end = lastIdx_block_tile[tile * max_active_block + block];

            // for (int t = 0; t < k; ++t){
            //     for (int i = tId; i < sh_tile_r && (i +(tile*sh_tile)) < n_rows; i+=blockDim.x){
            //         sh_r[t * sh_tile + i] = u[ (t * n_rows) + (active_row[(tile*max_active_row) + (block * sh_tile_r + i)])];                   
            //     }
            // }
            WARP_ID = tId >> 5;
            tid_in_WARP = tId & 31;
            step = blockDim.x >> 5;    
            int t = tid_in_WARP;
            for (int i = WARP_ID; i < sh_tile && i < n_rows; i+=step){
                //if(tile_no > 1005) printf("I am in \n");
                    sh_r[i *  k + t] = u[(active_row[(tile*max_active_row) + (block * sh_tile_r + i)]) * k + t];
            }  
            __syncthreads();
            // int c = tId + block_st;
            // if(c < block_end && c< nnz)
            for (int c = (tId >> 1)  + block_st; c < block_end && c< nnz; c+=(blockDim.x >> 1)){
                float sm =0 ;           float sm1 =0, sm2=0, sm3=0, sm4=0;
                // int row = row_ind[c];
                // int col = col_ind[c];
                int row = row_ind[c] >> 8;
                int col = tile*sh_tile + (row_ind[c] & 0xff);

                int active_row_ = active_row[(tile*max_active_row) + row];

                int passive_row = row;
                int sh_row = row - block * sh_tile;
                int sh_col = col - tile * sh_tile;
                for (int t = laneId*16; t < (laneId+1)*16; t+=8)
               // for (int t = 0; t < k; t+=8)
                {
                    float4 rtmp1 = *((float4*) &sh_r[sh_row * k + t]); 
                    float4 ctmp1 = *((float4*) &sh_c[ sh_col * k + t]);
                    sm1+= rtmp1.x * ctmp1.x + rtmp1.y * ctmp1.y +  rtmp1.z * ctmp1.z +  rtmp1.w * ctmp1.w ; 
                    
                    float4 rtmp2 = *((float4*) &sh_r[sh_row * k + t+4]); 
                    float4 ctmp2 = *((float4*) &sh_c[ sh_col * k + t+4]);
                    sm2+= rtmp2.x * ctmp2.x + rtmp2.y * ctmp2.y +  rtmp2.z * ctmp2.z +  rtmp2.w * ctmp2.w ; 
                }
                sm1 += __shfl_xor(sm1, 1);
                sm2 += __shfl_xor(sm2, 1);
                val[c] = val[c] * (sm1 + sm2);

                // for (int t = 0; t < k; ++t){
                //     //sm += u[row*k+t] * v[col*k+t];
                //     // sm += u[t*n_rows+row] * v[t*n_cols+cc];
                //     //sm += u[active_row_ * k +t] * sh_c[ sh_col * k + t]; 
                //     //sm += sh_r[t * sh_tile + sh_row] * sh_c[ sh_col * k + t]; 
                //     sm += sh_r[sh_row * k + t] * sh_c[ sh_col * k + t]; 
                // }
                // val[c] = val[c] * sm; 
            }
            block_st = block_end;   
            __syncthreads();         
        }  
   }  
}

//no vWARP otherwise perfect
// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int const* __restrict__ col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *active_row, int *tile_limIdx, int * lastIdx_block_tile, 
//   int *no_block_tile, int max_active_block, int tile_size, int max_active_row){
//     unsigned int tId = threadIdx.x;
//     unsigned int gId = (blockIdx.x * blockDim.x + tId) ;
//     int nnz_tile = tile_limIdx[blockIdx.x+1] - tile_limIdx[blockIdx.x];
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     //__shared__ float sh_c[180*32];
//     __shared__ float sh_c[96*32];
//     __shared__ float sh_r[96*32];
//     int tile_no = blockIdx.x;
//     int sh_tile = 96;
//     int sh_tile_r = 96;
//     int WARP_ID = tId >> 5;
//     int tid_in_WARP = tId & 31;
//     int step = blockDim.x >> 5;
//     int o_t = 0;
//     //for (int o_t = 0; o_t < k; o_t+=32)
//     {
//         unsigned int c = tId + tile_st;
//         int tile  = blockIdx.x;// col/tile_size;
//         {   int t = tid_in_WARP;
//             for (int i = WARP_ID; i < sh_tile && (i +(tile_no*sh_tile)) < n_cols; i+=step){
//                 sh_c[i *  k + t] = v[(tile * sh_tile + i) *k + t];
//             }
//         }           
//         __syncthreads();

//         int g_n_blocks = no_block_tile[blockIdx.x]; int n_blocks=0;
//         if(blockIdx.x == 0) n_blocks =  no_block_tile[blockIdx.x];
//         else n_blocks =  no_block_tile[blockIdx.x] - no_block_tile[blockIdx.x-1];
//         //if(blockIdx.x == 1 && tId ==0 ) printf("GPU %d : %d %d %d %d\n",  tile, n_blocks, tile_st, tile_end );

//         int block_st = tile_st;// lastIdx_block_tile[g_n_blocks + block-1];
//         for (int block = 0; block < n_blocks ; block++){          
//             int block_end = lastIdx_block_tile[tile * max_active_block + block];

//             // for (int t = 0; t < k; ++t){
//             //     for (int i = tId; i < sh_tile_r && (i +(tile*sh_tile)) < n_rows; i+=blockDim.x){
//             //         sh_r[t * sh_tile + i] = u[ (t * n_rows) + (active_row[(tile*max_active_row) + (block * sh_tile_r + i)])];                   
//             //     }
//             // }
//             WARP_ID = tId >> 5;
//             tid_in_WARP = tId & 31;
//             step = blockDim.x >> 5;    
//             int t = tid_in_WARP;
//             for (int i = WARP_ID; i < sh_tile && i < n_rows; i+=step){
//                 //if(tile_no > 1005) printf("I am in \n");
//                     sh_r[i *  k + t] = u[(active_row[(tile*max_active_row) + (block * sh_tile_r + i)]) * k + t];
//             }  
//             __syncthreads();
//             // int c = tId + block_st;
//             // if(c < block_end && c< nnz)
//             for (int c = tId + block_st; c < block_end && c< nnz; c+=blockDim.x ){
//                 float sm =0 ;           float sm1 =0, sm2=0, sm3=0, sm4=0;
//                 // int row = row_ind[c];
//                 // int col = col_ind[c];
//                 int row = row_ind[c] >> 8;
//                 int col = tile*sh_tile + (row_ind[c] & 0xff);

//                 int active_row_ = active_row[(tile*max_active_row) + row];

//                 int passive_row = row;
//                 int sh_row = row - block * sh_tile;
//                 int sh_col = col - tile * sh_tile;
//                 for (int t = 0; t < k; t+=8)
//                 {
//                     float4 rtmp1 = *((float4*) &sh_r[sh_row * k + t]); 
//                     float4 ctmp1 = *((float4*) &sh_c[ sh_col * k + t]);
//                     sm1+= rtmp1.x * ctmp1.x + rtmp1.y * ctmp1.y +  rtmp1.z * ctmp1.z +  rtmp1.w * ctmp1.w ; 
                    
//                     float4 rtmp2 = *((float4*) &sh_r[sh_row * k + t+4]); 
//                     float4 ctmp2 = *((float4*) &sh_c[ sh_col * k + t+4]);
//                     sm2+= rtmp2.x * ctmp2.x + rtmp2.y * ctmp2.y +  rtmp2.z * ctmp2.z +  rtmp2.w * ctmp2.w ; 
//                 }
//                 val[c] = val[c] * (sm1 + sm2);

//                 // for (int t = 0; t < k; ++t){
//                 //     //sm += u[row*k+t] * v[col*k+t];
//                 //     // sm += u[t*n_rows+row] * v[t*n_cols+cc];
//                 //     //sm += u[active_row_ * k +t] * sh_c[ sh_col * k + t]; 
//                 //     //sm += sh_r[t * sh_tile + sh_row] * sh_c[ sh_col * k + t]; 
//                 //     sm += sh_r[sh_row * k + t] * sh_c[ sh_col * k + t]; 
//                 // }
//                 // val[c] = val[c] * sm; 
//             }
//             block_st = block_end;   
//             __syncthreads();         
//         }  
//    }  
// }
// _

// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int const* __restrict__ col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *active_row, int *tile_limIdx, int * lastIdx_block_tile, 
//   int *no_block_tile, int max_active_block, int tile_size, int max_active_row){
//     unsigned int tId = threadIdx.x;
//     unsigned int gId = (blockIdx.x * blockDim.x + tId) ;
//     int nnz_tile = tile_limIdx[blockIdx.x+1] - tile_limIdx[blockIdx.x];
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     //__shared__ float sh_c[180*32];
//     __shared__ float sh_c[95*32];
//     __shared__ float sh_r[95*32];

//     int sh_tile = 95;
//     int sh_tile_r = 95;
//     int o_t = 0;
//     //for (int o_t = 0; o_t < k; o_t+=32)
//     {
//         unsigned int c = tId + tile_st;
//         int tile  = blockIdx.x;// col/tile_size;
//         if(c < tile_end && c < nnz)
//         {
//             for (int t = 0; t < k; ++t){
//                 for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_cols; i+=blockDim.x){
//                     sh_c[t * sh_tile + i] = v[ (t * n_cols) + (tile * sh_tile + i)];
//                 }
//             }
//         }           
//         __syncthreads();

//         int g_n_blocks = no_block_tile[blockIdx.x]; int n_blocks=0;
//         if(blockIdx.x == 0) n_blocks =  no_block_tile[blockIdx.x];
//         else n_blocks =  no_block_tile[blockIdx.x] - no_block_tile[blockIdx.x-1];
//         //if(blockIdx.x == 1 && tId ==0 ) printf("GPU %d : %d %d %d %d\n",  tile, n_blocks, tile_st, tile_end );

//         int block_st = tile_st;// lastIdx_block_tile[g_n_blocks + block-1];
//         for (int block = 0; block < n_blocks ; block++){          
//             int block_end = lastIdx_block_tile[tile * max_active_block + block];

//             for (int t = 0; t < k; ++t){
//                 for (int i = tId; i < sh_tile_r && (i +(tile*sh_tile)) < n_rows; i+=blockDim.x){
//                     sh_r[t * sh_tile + i] = u[ (t * n_rows) + (active_row[(tile*max_active_row) + (block * sh_tile_r + i)])];                   
//                 }
//             }    
//             __syncthreads();
//             // int c = tId + block_st;
//             // if(c < block_end && c< nnz)
//             for (int c = tId + block_st; c < block_end && c< nnz; c+=blockDim.x ){
//                 float sm =0 ;
//                 // int row = row_ind[c];
//                 // int col = col_ind[c];
//                 int row = row_ind[c] >> 8;
//                 int col = tile*sh_tile + (row_ind[c] & 0xff);

//                 int active_row_ = active_row[(tile*max_active_row) + row];
//                 int passive_row = row;
//                 int sh_row = row - block * sh_tile;
//                 int sh_col = col - tile * sh_tile;
//                 // if(blockIdx.x == 1 && tId == 5 && block == 1) 
//                 //      printf("***GPU*** %d %d %d : %d %d  \n",c,  row, col, sh_row, sh_col  );
//                 //int row = row_ind[c];
              
//                 for (int t = 0; t < k; ++t){
//                     //sm += u[row*k+t] * v[col*k+t];
//                     // sm += u[t*n_rows+row] * v[t*n_cols+cc];
//                     //sm += u[t*n_rows+active_row_] ;//* sh_c[t* sh_tile + sh_col]; 
//                     sm += sh_r[t * sh_tile + sh_row] * sh_c[t* sh_tile + sh_col]; 
//                 }
//                 val[c] = val[c] * sm; 
//             }
//             block_st = block_end;   
//             __syncthreads();         
//         }  
//    }  
// }
// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int const* __restrict__ col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *active_row, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int gId = (blockIdx.x * blockDim.x + tId) ;
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     __shared__ float sh_c[96*32];
//     __shared__ float sh_r[96*32];
//     //__shared__ float sh_c[192*32];
//     int sh_tile = 96;
//     unsigned int c = tId + tile_st;
//     int col = col_ind[c];
//     int tile  = col/sh_tile;
//     if(c < tile_end && c< nnz){
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_cols; i+=blockDim.x){
//                 sh_c[t * sh_tile + i] = v[ (t * n_cols) + (tile * sh_tile + i)];
//             }
//         }
//     }           
    
//     __syncthreads();
//     int offset[96];
