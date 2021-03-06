
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

//     for (int c = tId + tile_st; c < tile_end && c< nnz; ){
        
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_rows; i+=blockDim.x){
//                 sh_r[t * sh_tile + i] = u[ (t * n_rows) + (tile * sh_tile + active_row[i])];
//                 offset[i] = active_row[i] - i;
//             }
//         }    
//         __syncthreads();

//         float sm =0 ;
//         int row = row_ind[c];
//         row = offset[row];
//         int cc = col_ind[c];
//         for (int t = 0; t < k; ++t)
//             //sm += u[row*k+t] * v[col*k+t];
//             // sm += u[row*k+t] * sh[(col - tile*180) * k + t];
//             //sm += u[row*k+t] * v[t*n_cols+col];
//             float u_val = sh_r[(t * sh_tile) + (cc - tile * sh_tile)];
//             //u[t*n_rows+row];
//             sm += u_val * sh_c[(t * sh_tile) + (cc - tile * sh_tile)];
//         val[c] = val[c] * sm;  
//         c+=blockDim.x;      
//     }    
// }


// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *tiled_ind, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     __shared__ float sh[180*32];
//     int sh_tile = 180;
//     c = tId + tile_st;
//     int col = col_ind[c];
//     int tile  = col/sh_tile;

//     //if(tId < sh_tile && (tId +(tile*sh_tile)) < n_cols){
//     // if( c < tile_end && c < nnz ){ 
//     //     for (int t = 0; t < k; ++t){
//     //         for (int i = tId; i < 180 && (i +(tile*180)) < n_cols; i+=blockDim.x)
//     //             sh[t * 180 + i] = v[ (t * n_cols) + (tile * 180 + i)];
//     //     }           
//     // }
//     // __syncthreads();

//         //if( c < nnz ){

//     for (c = tId + tile_st; c < tile_end && c< nnz; c+=blockDim.x){
//         int col = col_ind[c];
//         int tile  = col/sh_tile;
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_cols; i+=blockDim.x)
//                 sh[t * sh_tile + i] = v[ (t * n_cols) + (tile * sh_tile + i)];

//         }    
//         __syncthreads();
//         float sm =0 ;
//         int row = row_ind[c];
//         //int col = col_ind[c];
//         for (int t = 0; t < k; ++t)
//             //sm += u[row*k+t] * v[col*k+t];
//             // sm += u[row*k+t] * sh[(col - tile*180) * k + t];
//             //sm += u[row*k+t] * v[t*n_cols+col];
//             sm += u[row*k+t] * sh[(t * sh_tile) + (col - tile * sh_tile)];
//         val[c] = val[c] * sm;
//         // if(col>130 && col< 135 && c <10000)
//             // printf("from gpu %d %d %d val: %f %f \n",c, col, tile,  sh[col/tile * k ], v[col*k] ); 
        
//     }
    
// }
// __global__ void comp_kernel_col_sorted_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *tiled_ind, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     __shared__ float sh[180*32];
//     int sh_tile = 180;
//     c = tId + tile_st;
//     int col = col_ind[c];
//     int tile  = col/sh_tile;

//     //if(tId < sh_tile && (tId +(tile*sh_tile)) < n_cols){
//     //if( c < tile_end && c< nnz )
//     // if(c < tId + tile_st + blockDim.x)
//     // { 
//     //     for (int t = 0; t < k; ++t){
//     //         for (int i = tId; i < 180 && (i +(tile*180)) < n_cols; i+=blockDim.x)
//     //             sh[t * 180 + i] = v[ (t * n_cols) + (tile * 180 + i)];
//     //     }           
//     // }
//     // __syncthreads();

//         //if( c < nnz ){

//     for (c = tId + tile_st; c < tile_end && c< nnz; c+=blockDim.x){
//         int col = col_ind[c];
//         int tile  = col/sh_tile;
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_cols; i+=blockDim.x)
//                 sh[t * sh_tile + i] = v[ (t * n_cols) + (tile * sh_tile + i)];

//         }    
//         __syncthreads();
//         float sm =0 ;
//         int row = row_ind[c];
//         //int col = col_ind[c];
//         for (int t = 0; t < k; ++t)
//             //sm += u[row*k+t] * v[col*k+t];
//             // sm += u[row*k+t] * sh[(col - tile*180) * k + t];
//             //sm += u[row*k+t] * v[t*n_cols+col];
//             sm += u[row*k+t] * sh[(t * sh_tile) + (col - tile * sh_tile)];
//         val[c] = val[c] * sm;
//         // if(col>130 && col< 135 && c <10000)
//             // printf("from gpu %d %d %d val: %f %f \n",c, col, tile,  sh[col/tile * k ], v[col*k] ); 
        
//     }
    
// }

// //CSR format
// __global__ void comp_kernel_CSR(int const* __restrict__ row_ptr, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//  float * p_ind, int n_rows, int k)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int laneId = tId & 31;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) >> 5;
//     extern __shared__ volatile  float SD[];

//     if( c < n_rows ){
            
//         // c = rowGroupPtr[c];  float g=0,h=0;
//         unsigned int row_p = row_ptr[c], nnz_row = row_ptr[c+1] - row_ptr[c];
//         for(long i=laneId; i < nnz_row; i += 32) {
//             float sm =0 ;
//             int row = c;
//             int col = col_ind[row_p + i];
//             for (int t = 0; t < k; ++t)
//                 sm += u[row*k+t] * v[col*k+t];
//             p_ind[row_p + i] = sm * val[row_p + i];
 

//         }
//         // __syncthreads();
//         if(c>500 && c< 505  && laneId == 0)
//             printf("from gpu %d %f\n",c,  p_ind[c] );
//         // float newvj=0;
//         // g += __shfl_down(g, 16);   
//         // g += __shfl_down(g, 8);
//         // g += __shfl_down(g, 4);   
//         // g += __shfl_down(g, 2);
//         // g += __shfl_down(g, 1);
//         // __syncthreads();
       
//     } 
// }


// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, 
//     const float * __restrict__ u, const float * __restrict__ v, float * p_ind, int nnz, int k){
  
//     unsigned int tId = threadIdx.x;
//     // unsigned int laneId = tId & 31;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;//>> 5;
//     // extern __shared__ volatile  float SD[];
//     // printf("I am in \n");
//     if( c < nnz ){

//         float sm =0 ;
//         int row = row_ind[c];
//         int col = col_ind[c];
//         printf("from gpu %d %f %d %f\n", c , p_ind[c], row, v[row*k+1] );

//         for (int t = 0; t < k; ++t)
//             sm += u[row*k+t] * v[col*k+t];
//         float s = .4;
//         p_ind[c] += val[c] * s;
                    
//          // * val[c]; 

//         // // __syncthreads();
//         // if(c>5 && c< 14)
//         //     printf("from gpu %d %f\n",c,  p_ind[c] );      
//     } 



// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *tiled_ind, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;
//     int tile_st = tile_limIdx[blockIdx.x];
//     int tile_end = tile_limIdx[blockIdx.x+1];
//     __shared__ float sh[180*32];
    
//     c = tId + tile_st;
//     int cc = col_ind[c];
//     int tile  = cc/180;

//         //if( c < nnz ){
//     for (c = tId + tile_st; c < tile_end && c< nnz; c+=blockDim.x){
//         int cc = col_ind[c];
//         int tile  = cc/180;
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < 180 && (i +(tile*180)) < n_cols; i+=blockDim.x)
//                 sh[t * 180 + i] = v[ (t * n_cols) + (tile * 180 + i)];

//         }    
//         __syncthreads();
//     }

//         float sm =0 ;
//         int row = row_ind[c];
//         int col = col_ind[c];
//         for (int t = 0; t < k; ++t)
//             sm += u[row*k+t] * sh[(t * 180) + (col - tile * 180)];
//         val[c] = val[c] * sm;
//     }
// }

//no blcok cyclic 
// __global__ void comp_kernel_COO(int const* __restrict__ row_ind, int *  col_ind, float *val, const float * __restrict__ u, const float * __restrict__ v, 
//   int nnz, int n_rows, int n_cols, int k, int *tiled_ind, int *tile_limIdx)
// {
//     unsigned int tId = threadIdx.x;
//     unsigned int c = (blockIdx.x * blockDim.x + tId) ;

//     int sh_tile = 180;
//     int cc = col_ind[c];
//     int tile  = cc/sh_tile;

//     int tile_st = tile_limIdx[tile];
//     //int tile_end = tile_limIdx[blockIdx.x+1];
//     int tile_end = tile_limIdx[tile+1];
//     // c = tId + tile_st;
//     if( c < tile_end+sh_tile ){
//         //if(c < tId + tile_st + blockDim.x)
//         __shared__ float sh[180*32];
//         for (int t = 0; t < k; ++t){
//             for (int i = tId; i < sh_tile && (i +(tile*sh_tile)) < n_cols; i+=blockDim.x){
//                 sh[t * sh_tile + i] = v[ (t * n_cols) + (tile * sh_tile + i)];
//             }
//         }   
//         __syncthreads();
//         if(c == 189520)printf("gpu 1 %d %d %d %d \n", c, cc, (0 * sh_tile) + (cc - tile * sh_tile), 0*n_cols+cc);
//         if( c < tile_end ){
       
//         //if(c % blockDim.x == 0 && blockIdx.x == 0) printf("blockcyclic %d %d \n", blockIdx.x, c);
//         //if(tId == 0 ) printf("gpu" ); 
//         float sm =0 ;
//         int row = row_ind[c];
//         int col = col_ind[c];
//         // if(sh[(0 * sh_tile) + (col - tile * sh_tile)] != v[0*n_cols+col] && col< 190){
//         //     printf("gpu 1 %d %d %f %f \n", c, col, sh[(0 * sh_tile) + (col - tile * sh_tile)], v[0*n_cols+col]);
//         //     printf("gpu 2 %d %d %f %f \n", c, col, sh[(1 * sh_tile) + (col - tile * sh_tile)], v[1*n_cols+col]);
//         // }

//         for (int t = 0; t < k; ++t){
//             //sm += u[row*k+t] * v[col*k+t];
//             //sm += u[row*k+t] * sh[(col - tile*180) * k + t];
//             //sm += u[row*k+t] * v[t*n_cols+col];
//             sm += u[row*k+t] * sh[(t * sh_tile) + (col - tile * sh_tile)];
//             //if(c =55064677 && t ==6 ) printf("gpu %d %f %f \n", c, sh[(t * sh_tile) + (col - tile * sh_tile)], v[t*n_cols+col]);
//         }
//         val[c] = val[c] * sm;
//         // if(col>130 && col< 135 && c <10000)
//             // printf("from gpu %d %d %d val: %f %f \n",c, col, tile,  sh[col/tile * k ], v[col*k] ); 
//     }
//     }
    
// }



