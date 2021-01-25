#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <utility>  
#include <math.h> 
#include <omp.h>
#include <cuda.h>
#include "util.h"
#include "kernel.h"
#include <bits/stdc++.h>    
using namespace std;

// long n_rows, n_cols, nnz;

int tile_sizeX = 256; 
int tile_sizeY = 25000;
int k = 100;


inline cudaError_t checkCuda(cudaError_t result, int s){

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error in line : %s - %d\n", cudaGetErrorString(result), s);
    assert(result == cudaSuccess);
  }
  return result;
}

void sddmm_GPU( const Matrix S, const TiledMatrix tS, float* P, vector <float>W, vector <float> H){

    float *d_val, *d_W, *d_H, *d_W_t;
    int *d_row_ptr, *d_col_ind, *d_row_ind, *d_tiled_ind, *d_lastIdx, *d_active_row, *d_lastIdx_block_tile, *d_passive_row;
 
   //***********Starting GPU****************
    checkCuda(cudaMalloc((void**)&d_W, k * S.n_rows * sizeof(float)),0); 
    checkCuda(cudaMalloc((void**)&d_H, k * S.n_cols * sizeof (float)),1);
    // checkCuda(cudaMalloc((void**)&d_row_ptr, (n_rows+1) * sizeof (int)),2);   
    checkCuda(cudaMalloc((void**)&d_row_ind, tS.nnz * sizeof (int)),4);  
    checkCuda(cudaMalloc((void**)&d_col_ind, tS.nnz * sizeof (int)),4);
    checkCuda(cudaMalloc((void**)&d_val, tS.nnz * sizeof (float)),4);
    checkCuda(cudaMalloc((void**)&d_lastIdx, (tS.ntile_c+1) * sizeof (float)),4);
    checkCuda(cudaMalloc((void**)&d_active_row, tS.ntile_c * tS.max_active_row * sizeof (int)),4);
    checkCuda(cudaMalloc((void**)&d_lastIdx_block_tile, tS.ntile_c * tS.max_active_block * sizeof (int)),4);

   
    // checkCuda(cudaMemcpy(d_row_ptr,  &(row_ptr[0]), (n_rows+1) * sizeof (int), cudaMemcpyHostToDevice),4);
    checkCuda(cudaMemcpy(d_row_ind,  &(tS.rows[0]), tS.nnz * sizeof (int), cudaMemcpyHostToDevice),4);
    checkCuda(cudaMemcpy(d_col_ind, &(tS.cols[0]), tS.nnz * sizeof (int), cudaMemcpyHostToDevice),4);
    //checkCuda(cudaMemcpy(d_val, &(new_vals[0]), tS.nnz * sizeof (float), cudaMemcpyHostToDevice),4);
    cudaMemset(d_val, 0, S.nnz * sizeof (float));
    checkCuda(cudaMemcpy(d_lastIdx, &(tS.lastIdx_tile[0]), (tS.ntile_c+1) * sizeof (int), cudaMemcpyHostToDevice),4);   
    for (int i = 0; i < tS.ntile_c; ++i){
        checkCuda(cudaMemcpy(d_lastIdx_block_tile+i * tS.max_active_block, &(tS.lastIdx_block_tile[i*tS.max_active_block]), tS.max_active_block * sizeof (int), cudaMemcpyHostToDevice),4);   
        //cout <<i<<" "<< tS.lastIdx_tile[i]<<" "<<tS.lastIdx_block_tile[i*tS.max_active_block]<< endl;
    }

    int sum =0 ;
    for (int i = 0; i < tS.ntile_c; ++i){
        checkCuda(cudaMemcpy(d_active_row+sum, &(tS.active_row[i*S.n_rows]), tS.n_actv_row[i] * sizeof (int), cudaMemcpyHostToDevice),4); 
        sum += tS.n_actv_row[i];
    }
    // sum=0;
    // for (int i = 0; i < tS.ntile_c; ++i){
    //     checkCuda(cudaMemcpy(d_passive_row+sum, &(passive_row[i*S.n_rows]), S.n_rows * sizeof (int), cudaMemcpyHostToDevice),4); 
    //     sum += S.n_rows;
    // }
         
    cudaMemcpy(d_W, &(W[0]),  S.n_rows * k  * sizeof (float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_W, &(W_t[0]),  S.n_rows * k  * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, &(H[0]),  S.n_cols * k  * sizeof (float), cudaMemcpyHostToDevice);  
    //cudaMemcpy(d_H, &(H_t[0]),  S.n_cols * k  * sizeof (float), cudaMemcpyHostToDevice);  

    
    int n_tile = tS.ntile_c ;//S.n_cols/tile_sizeX + 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t stream[n_tile]; 
    for (int i = 0; i < n_tile; i++){
        cudaStreamCreate(&(stream[i]));
    }

    float mili =0, copyTime = 0 ;
  
    dim3 block(BLOCKSIZE,1,1), grid(1,1,1);
    sum = 0; int t_st =0 ;

    int k_slice = SM_CAPACITY/actv_row_size;

    checkCuda(cudaEventRecord(start), __LINE__);

    for (int tile = 0; tile < n_tile; ++tile){
        int nnz_tile = tS.lastIdx_tile[tile+1]-tS.lastIdx_tile[tile];
        //grid.x = (nnz_tile + BLOCKSIZE - 1) / BLOCKSIZE;
        int active_block_this_tile = tS.n_actv_row[tile]/actv_row_size+1;
        
        grid.x = active_block_this_tile;
        for (int t_st = 0; t_st < k ; t_st +=k_slice){
            comp_kernel_COO<<<grid,block, 0, stream[0]>>>(d_row_ind, d_col_ind, d_val, d_W, d_H, 
            S.nnz, S.n_rows, S.n_cols, k, tS.lastIdx_tile[tile], tS.lastIdx_tile[tile+1], &(d_lastIdx_block_tile[(tile)*tS.max_active_block]), 
            d_active_row+sum, tile, t_st, tS.n_actv_row[tile], actv_row_size, k_slice);
        }
        sum += tS.n_actv_row[tile];

    }       
    checkCuda(cudaEventRecord(stop), __LINE__);
    cudaEventSynchronize(stop);
    //cudaDeviceSynchronize();
    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
    cudaDeviceSynchronize();
    cout << "\nTime for SDDMM with K = " << k << " : " << mili << " ms" << endl;

    //******** correctness check
    float GPU_tot = 0, CPU_tot =0, CPU_tot_orig =0 ;
    float *p_ind_temp = new float[tS.nnz];
    checkCuda(cudaMemcpy(&(p_ind_temp[0]), d_val, tS.nnz * sizeof (float), cudaMemcpyDeviceToHost),4);;    
    for (int i = 0; i < S.nnz; ++i){
        CPU_tot +=  P[tS.tiled_ind[i]];
        CPU_tot_orig +=  P[i];
        // cout << "P " << P[tS.tiled_ind[i]] << " " << P[i] << " new,old ind: "<<tS.tiled_ind[i] <<" "<<i<< endl;
    }

    /*Correctness check*/
    /*
    cout << "\n****** Optional correctness check *******" << endl << endl;
    for (int i = 0; i <  3; ++i)
            cout << "gp idx " << i << " "  <<" GPU "<< p_ind_temp[i] << " CPU "<< P[tS.tiled_ind[i]]<<endl;  
    for (int i = S.nnz-1; i > S.nnz-3; --i)
             cout << "gp idx " << i << " "  <<" GPU "<< p_ind_temp[i] << " CPU "<< P[tS.tiled_ind[i]]<<endl;  
    long diff_tot = 0; 
    for (int i = 0; i < tS.nnz; ++i){
        //if(p_ind_temp[i] != 0)
        {
           if(abs(p_ind_temp[i]-P[tS.tiled_ind[i]]) > .000001){
                diff_tot ++;
                if(diff_tot < 5)
                    printf("CPU GPU diff %d:  %f %f %f \n", i, p_ind_temp[i], P[tS.tiled_ind[i]],p_ind_temp[i]-P[tS.tiled_ind[i]] );
            }
        }
    }
    cout << "diff values in CPU and GPU: " << diff_tot << endl;
    */
    //freeing device allocation
    cudaFree( d_row_ptr ); cudaFree( d_row_ind); cudaFree( d_col_ind); cudaFree( d_val);
    cudaFree(d_active_row); cudaFree(d_passive_row);
    cudaFree(d_lastIdx_block_tile); cudaFree(d_lastIdx);
    cudaFree( d_W ); cudaFree( d_H );
}

void sddmm_CPU_CSR(int * row_ptr, int *col_ind, float * val_ind, vector <float> W, vector <float>H, float * p_ind, const Matrix S){
       // reduction(+:rmse)
    long tot =0 ;
    #pragma omp parallel for reduction(+:tot)
    for (int r = 0; r < S.n_rows; ++r){
        tot += row_ptr[r+1] - row_ptr[r];
        float sm =0 ;
        for (int ind = row_ptr[r]; ind < row_ptr[r+1]; ++ind){
            int row = r;
            int col = col_ind[ind]; 
            int nnz = row_ptr[r+1]-row_ptr[r];

            float val = val_ind[ind];
            sm=0;
            for (int t = 0; t < k; ++t){
                sm += W[row * k + t] * H[col * k + t];
                // cout <<W[row * k + t] <<" "<<H[col * k + t]<< endl;
            }
            p_ind[ind] = sm * val_ind[ind];
            // cout << "ind " << row<<" "<<col << ":: "  <<" "<< p_ind[ind] << " = " << sm <<" * "<< val_ind[ind]<< endl;  

           
        }                
    } 
}
// vector<int> rows, vector<int> cols, vector<float> vals

void sddmm_CPU_COO(vector<int> row_ind, vector<int> col_ind, vector<float> val_ind, vector <float> W, vector <float>H, float * p_ind, const Matrix S){
       // reduction(+:rmse)
    double start_time = omp_get_wtime();
    omp_set_dynamic(0);
    omp_set_num_threads(28);
    #pragma omp parallel for //reduction(+:tot)
    for (int ind = 0; ind < S.nnz; ind++){
        float sm =0 ;
            int row = row_ind[ind];
            int col = col_ind[ind]; 
            for (int t = 0; t < k; ++t)
                sm += W[row * k + t] * H[col * k + t];
            p_ind[ind] = sm ;//* val_ind[ind];
            // cout << "ind " << row<<" "<<col << ":: "  <<" "<< p_ind[ind] << " = " << sm <<" * "<< val_ind[ind]<< endl;  
               // }                
    }
    double CPU_time = omp_get_wtime() - start_time;
    //correctness check

    // printf("\nomp time CPU : %.4f \n\n", CPU_time*1000);
}


void preprocessing(const Matrix S){

    int *row_ptr = new int[S.n_rows+1];
    float *P = new float[S.nnz]; // output Matrix
    
    TiledMatrix tiledS(S, tile_sizeX, tile_sizeY);
    int ntile_c = tiledS.ntile_c;
    int ntile_r = tiledS.ntile_r;

    /* Populate rhs dense matrices */
    vector<float> W (S.n_rows * k);
    vector<float> W_t (S.n_rows * k);
    vector<float> H (S.n_cols * k);
    vector<float> H_t (S.n_cols * k);
    
    initial(W, S.n_rows, k);
    initial(H, S.n_cols, k);
    make_HTasH(H, H_t, S.n_cols, k);
    make_HTasH(W, W_t, S.n_rows, k);
    
    int *row_holder = new int[S.n_rows];

    tiledS.nnz =0 ;
    
    /*converting col sorted matrix to row sorted */
    //unsorted_make_CSR(rows, cols, vals, S.nnz, S.n_rows, S.n_cols, row_ptr);
    
    //assuming sorted
    make_CSR(S.rows, S.cols, S.vals, S.nnz, S.n_rows, row_ptr, row_holder);

    tiledS.max_active_row = rewrite_matrix_1D(S, tiledS, row_ptr, tile_sizeX, row_holder);

    // rewrite_col_sorted_matrix(row_ptr, rows, cols, vals, tS.rows, tS.cols, new_vals, S.nnz, 
        //S.n_rows, S.n_cols, tile_sizeX, tiled_ind, lastIdx_tile, BLOCKSIZE, tS.nnz);
    
    /* CPU call */
    // sddmm_CPU_CSR(row_ptr, cols, vals, W, H, P);
    sddmm_CPU_COO(S.rows, S.cols, S.vals, W, H, P, S);
    
    /* GPU call */
    sddmm_GPU(S, tiledS, P, W, H);
    
}

int main(int argc, char* argv[]){ 
    
    Matrix S;

    ifstream fp(argv[1]);   
    k = atoi(argv[2]);
    tile_sizeY = atoi(argv[3]);  
    tile_sizeX = atoi(argv[4]); 
    actv_row_size = tile_sizeY;  
    
    string str;  
    fp >> str;
    while(!isdigit(str[0])){
        getline(fp,str);
    }

    istringstream is(str);
    is >> S.n_rows; 
    is >> S.n_cols; 
    is >> S.nnz; 
    cout << "\nMatrix info: " << " rows: " <<  S.n_rows << ", cols: " << S.n_cols << ", nnz: " << S.nnz << endl;
    long orig_nnz = S.nnz, rid = 0,cid = 0; float vid=0;

    S.rows.resize(S.nnz);
    S.cols.resize(S.nnz);
    S.vals.resize(S.nnz); 
    
    long idx=0;
    for (long o_idx = 0; o_idx < orig_nnz; ++o_idx) {
        fp >> rid >> cid >> vid;
        S.rows[idx]=rid-1;
        S.cols[idx]=cid-1;
        S.vals[idx]=vid;
        idx++;
    }
    cout << "\nTilesize: X = " << tile_sizeX << ", tilesize: Y = " << tile_sizeY << ", TB: "<< BLOCKSIZE << endl;
    S.nnz=idx;

    preprocessing(S);
}


