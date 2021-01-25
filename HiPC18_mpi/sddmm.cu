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
#include <mpi.h> 
using namespace std;
#define mpi_barrier() MPI_Barrier(MPI_COMM_WORLD);

long n_rows, n_cols, nnz;
int actv_row_size = 180;
int tile_sizeX = 256; 
int tile_sizeY = 25000;
int k=100;
int SM_CAPACITY = 8192;
int BLOCKSIZE=512;


inline cudaError_t checkCuda(cudaError_t result, int s){

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error in line : %s - %d\n", cudaGetErrorString(result), s);
    assert(result == cudaSuccess);
  }
  return result;
}

void sddmm_GPU(int * d_row_ptr, int * d_row_ind, int *d_col_ind, float * d_val_ind, float * d_W, float *d_H, 
int *d_tiled_ind, int *d_lastIdx, int *lastIdx_tile, int *d_lastIdx_block_tile ,int *d_active_row, int *d_passive_row, 
int * count_actv_row, int &max_active_block, long new_nnz, int mpi_rank, int cols_in_rank, int nnz_in_rank, int col_loc){
    int n_tile = n_cols/tile_sizeX + 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t stream[n_tile]; 
    for (int i = 0; i < n_tile; i++){
        cudaStreamCreate(&(stream[i]));
    }

    float mili =0, copyTime = 0 ;
  
    dim3 block(BLOCKSIZE,1,1), grid(1,1,1);
    int sum = 0, t_st =0 ;

    int k_slice = SM_CAPACITY/actv_row_size;
    int col_offset = col_loc ;
    mpi_barrier();

    // cout << "k_slice " << k_slice << endl;
    checkCuda(cudaEventRecord(start), __LINE__);

    int nnz_tile = nnz_in_rank; // lastIdx_tile[mpi_rank+1]-lastIdx_tile[mpi_rank];
//         if(mpi_rank == 1)
// cout << " after " << lastIdx_tile[mpi_rank] << " " << lastIdx_tile[mpi_rank+1]-lastIdx_tile[mpi_rank] << " "<< nnz_in_rank << endl;

        int active_block_this_tile = count_actv_row[mpi_rank]/actv_row_size+1;
        grid.x = active_block_this_tile;
        //  if(mpi_rank == 1)
        // cout << "ranks "<<mpi_rank <<" nnz "<< nnz_tile << " grid "<<grid.x <<" "<<count_actv_row[mpi_rank ]<< endl;
        // cout <<" ohh " <<mpi_rank <<" " << lastIdx_tile[mpi_rank] <<" " << lastIdx_tile[mpi_rank+1] <<  " " << tile_sizeX<< endl;
        //if(mpi_rank == 0)
        for (int t_st = 0; t_st < k ; t_st +=k_slice){
            comp_kernel_COO<<<grid,block, 0, stream[0]>>>(d_row_ind, d_col_ind, d_val_ind, d_W, d_H, 
            nnz, n_rows, n_cols, k, lastIdx_tile[mpi_rank], lastIdx_tile[mpi_rank+1], &(d_lastIdx_block_tile[(mpi_rank)*max_active_block]), 
            d_active_row, mpi_rank, t_st, count_actv_row[mpi_rank], actv_row_size, k_slice, col_offset);
        }


    
    //mpi_barrier();      
    checkCuda(cudaEventRecord(stop), __LINE__);

    cudaEventSynchronize(stop);
    // printf("\nTotal seconds: %.3f for rank = %d\n\n", seconds() - t0, mpi_rank);
    //cudaDeviceSynchronize();
    checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
    cudaDeviceSynchronize();
    cout << mili << " ";
    mpi_barrier();

}
void sddmm_CPU_CSR(int * row_ptr, int *col_ind, float * val_ind, float * W, float *H, float * p_ind){
       // reduction(+:rmse)
    long tot =0 ;
    #pragma omp parallel for reduction(+:tot)
    for (int r = 0; r < n_rows; ++r){
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

void sddmm_CPU_COO(int * row_ind, int *col_ind, float * val_ind, float * W, float *H, float * p_ind){
       // reduction(+:rmse)
    double start_time = omp_get_wtime();
    omp_set_dynamic(0);
    omp_set_num_threads(28);
    #pragma omp parallel for //reduction(+:tot)
    for (int ind = 0; ind < nnz; ind++){
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
//cout << "CPU " << row_ind[1] <<" "<<col_ind[1]  << endl;
    // printf("\nomp time CPU : %.4f \n\n", CPU_time*1000);
}


void init(int *rows, int *cols, float* vals){

    //**************************MPI init*************************
    MPI_Init(NULL, NULL);
    // Find out rank, size
    int mpi_rank, n_proc;;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    int root_p = sqrt(n_proc+1);
    int row_color = mpi_rank / root_p;
    int col_color = mpi_rank % root_p;
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, mpi_rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col_color, mpi_rank, &col_comm);

    int row_rank, row_size, col_rank, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    //printf("WORLD RANK/SIZE: %d/%d \t ROW RANK/SIZE: %d/%d %d/%d\n", mpi_rank, n_proc, row_rank, row_size,  col_rank, col_size);

    


    int n_bin=10;
    int *count = new int[n_bin];
    int *row_ptr = new int[n_rows+1];
    int *col_ptr = new int[n_cols+1];
    float *p_ind = new float[nnz];
   
    float *W = new float[n_rows*k];
    float *W_t = new float[n_rows*k];
    float *H = new float[n_cols*k];
    float *H_t = new float[n_cols*k];
    int n_tile_c = n_proc;
    int n_tile_r = n_rows/tile_sizeY + 1;
    int max_active_block = (n_rows/actv_row_size+1);
    int *count_actv_row = new int[n_tile_c];
    int *lastIdx_tile = new int[n_tile_c+1];
    int *lastIdx_block_tile = new int[(n_tile_c+1) * (n_rows/actv_row_size+1)];
    float *d_val, *d_W, *d_H, *d_W_t;
    int *d_row_ptr, *d_col_ind, *d_row_ind, *d_tiled_ind, *d_lastIdx, *d_active_row, *d_lastIdx_block_tile, *d_passive_row;

    int n_tileX = n_proc;// n_cols/tile_sizeX+1; 
    int n_tileY = n_rows/tile_sizeY+1; 
    long new_nnz =0 ;
    initial(W, n_rows, k);
    initial(H, n_cols, k);
    make_HTasH(H, H_t, n_cols, k);
    make_HTasH(W, W_t, n_rows, k);
    
    int *new_rows = new int[nnz];
    int *new_cols = new int[nnz];
    float *new_vals = new float[nnz];
    int *row_holder = new int[n_rows];
    int *tiled_ind = new int [nnz];
    int *active_row = new int[n_tileX * n_rows];
    int *passive_row = new int[n_tileX * n_rows];

    // int *new_rows = new int[nnz];
    // int *new_cols = new int[nnz];
    // float *new_vals = new float[nnz];
    //converting col sorted matrix to row sorted
    //unsorted_make_CSR(rows, cols, vals, nnz, n_rows, n_cols, row_ptr);
    //assuming sorted
    int *cols_in_asym_rank = new int [n_proc];
    int *nnz_in_asym_rank = new int [n_proc];
    make_CSR(rows, cols, vals, nnz, n_rows, row_ptr, row_holder);
    make_CSC(rows, cols, vals, nnz, n_rows, n_cols, col_ptr, mpi_rank, n_proc, cols_in_asym_rank, nnz_in_asym_rank);


   int cols_in_rank =  cols_in_asym_rank[mpi_rank];

    //in MPI set tile sizeX as cols in rank
    tile_sizeX = cols_in_rank;
    int col_loc = cols_in_rank * mpi_rank; 

    //comp_bin(n_bin, count, n_rows, row_ptr, nnz);
    int max_active_row=0;
    mpi_barrier();
    max_active_row = rewrite_matrix_1D(row_ptr, rows, cols, vals, new_rows, new_cols, new_vals, nnz, n_rows, n_cols, 
        tile_sizeX, tiled_ind, lastIdx_tile, active_row, passive_row, count_actv_row, lastIdx_block_tile, actv_row_size, 
         new_nnz, row_holder, actv_row_size, cols_in_asym_rank, nnz_in_asym_rank, mpi_rank, n_proc);

    mpi_barrier();
    int nnz_in_rank = nnz_in_asym_rank[mpi_rank];//lastIdx_tile[mpi_rank+1] - lastIdx_tile[mpi_rank];

    // if(mpi_rank == 1)
    //     cout <<" out " << lastIdx_tile[mpi_rank] <<" "<<
    //     lastIdx_tile[mpi_rank+1]<< endl;

    int nnz_loc = lastIdx_tile[mpi_rank];
    int nnz_loc_end = lastIdx_tile[mpi_rank+1];
    mpi_barrier();
    // cout << "sizes: "<<mpi_rank <<" " << tile_sizeX <<" nnz in rank "<< nnz_in_rank << " " <<nnz_loc<<
    // " " << cols_in_rank <<endl;
      mpi_barrier();
    // <<" " << nnz_loc <<" " <<nnz_loc_end <<endl;

           // rewrite_col_sorted_matrix(row_ptr, rows, cols, vals, new_rows, new_cols, new_vals, nnz, n_rows, n_cols, 
    //     tile_sizeX, tiled_ind, lastIdx_tile, BLOCKSIZE, new_nnz);
    double t0 = seconds();
    // sddmm_CPU_CSR(row_ptr, cols, vals, W, H, p_ind);
    sddmm_CPU_COO(rows, cols, vals, W, H, p_ind);

   //***********Starting GPU****************
    //if(mpi_rank == 0)
   // checkCuda(cudaMalloc((void**)&d_W, k*n_rows*sizeof(float)),0); 
    //********Allocate GPU memory********
    checkCuda(cudaMalloc((void**)&d_W, k*n_rows*sizeof(float)),0); 
    checkCuda(cudaMalloc((void**)&d_H, k*n_cols*sizeof(float)),1); 
    checkCuda(cudaMalloc((void**)&d_row_ind, nnz_in_rank*sizeof(int)),4);  
    checkCuda(cudaMalloc((void**)&d_col_ind, nnz_in_rank*sizeof(int)),4);
    checkCuda(cudaMalloc((void**)&d_val, nnz_in_rank*sizeof(float)),4);
    checkCuda(cudaMalloc((void**)&d_lastIdx, (n_tile_c+1)*sizeof(float)),4);
    checkCuda(cudaMalloc((void**)&d_active_row, n_tileX*max_active_row*sizeof(int)),4);
    checkCuda(cudaMalloc((void**)&d_lastIdx_block_tile, n_tileX*max_active_block*sizeof(int)),4);


    //******** Copy GPU memory********
    checkCuda(cudaMemcpy(d_row_ind,  &(new_rows[nnz_loc]), nnz_in_rank*sizeof(int), cudaMemcpyHostToDevice),4);
    checkCuda(cudaMemcpy(d_col_ind, &(new_cols[nnz_loc]), nnz_in_rank*sizeof(int), cudaMemcpyHostToDevice),4);
    //checkCuda(cudaMemcpy(d_val, &(new_vals[0]), new_nnz*sizeof(float), cudaMemcpyHostToDevice),4);

    cudaMemset(d_val, 0,  nnz_in_rank*sizeof(float));
    checkCuda(cudaMemcpy(d_lastIdx, &(lastIdx_tile[0]), (n_tile_c+1)*sizeof(int), cudaMemcpyHostToDevice),4);   
    for (int i = 0; i < n_tileX; ++i){
        checkCuda(cudaMemcpy(d_lastIdx_block_tile+i*max_active_block, &(lastIdx_block_tile[i*max_active_block]), max_active_block*sizeof(int), cudaMemcpyHostToDevice),4);   
        //cout <<i<<" "<< lastIdx_tile[i]<<" "<<lastIdx_block_tile[i*max_active_block]<< endl;
         mpi_barrier();
    }
    checkCuda(cudaMemcpy(d_active_row, &(active_row[mpi_rank*n_rows]), count_actv_row[mpi_rank]*sizeof(int), cudaMemcpyHostToDevice),4); 
    
    // int sum =0 ;
    // for (int i = 0; i < n_tileX; ++i){
    //     checkCuda(cudaMemcpy(d_active_row+sum, &(active_row[i*n_rows]), count_actv_row[i]*sizeof(int), cudaMemcpyHostToDevice),4); 
    //     mpi_barrier();
    //     sum += count_actv_row[i];
    // }
  
    cudaMemcpy(d_W, &(W[0]),  n_rows * k *sizeof(float), cudaMemcpyHostToDevice);

    //cudaMemcpy(d_W, &(W_t[0]),  n_rows * k *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, &(H[0]),  n_cols * k *sizeof(float), cudaMemcpyHostToDevice);  
    //cudaMemcpy(d_H, &(H[col_loc*k]),  cols_in_rank * k *sizeof(float), cudaMemcpyHostToDevice);  
    //cudaMemcpy(d_H, &(H_t[0]),  n_cols * k *sizeof(float), cudaMemcpyHostToDevice);  
    mpi_barrier();
            // if(mpi_rank == 1)
    // cout << " before " << lastIdx_tile[mpi_rank] << " " << lastIdx_tile[mpi_rank+1];
    sddmm_GPU(d_row_ptr, d_row_ind, d_col_ind, d_val, d_W, d_H, d_tiled_ind, d_lastIdx, lastIdx_tile, d_lastIdx_block_tile, d_active_row, d_passive_row ,count_actv_row, 
        max_active_block, new_nnz, mpi_rank, cols_in_rank, nnz_in_rank, col_loc );

    //******** correctness check
    mpi_barrier();

    float GPU_tot = 0, CPU_tot =0, CPU_tot_orig =0 ;
    float *p_ind_temp = new float[new_nnz];
    // cout << "nz loc " << nnz_loc <<" " << nnz_in_rank << endl; 
    mpi_barrier();
    checkCuda(cudaMemcpy(&(p_ind_temp[nnz_loc]), d_val, nnz_in_rank *sizeof(float), cudaMemcpyDeviceToHost),4);;    

    mpi_barrier();
    for (int i = 0; i < nnz; ++i){
        CPU_tot +=  p_ind[tiled_ind[i]];
        CPU_tot_orig +=  p_ind[i];
        // cout << "p_ind " << p_ind[tiled_ind[i]] << " " << p_ind[i] << " new,old ind: "<<tiled_ind[i] <<" "<<i<< endl;
    }
   
   // if(mpi_rank == 0)
   //  for (int i = nnz_loc; i <  nnz_loc+2; ++i)
   //          cout << "rank and idx: " <<mpi_rank <<" " << i << " "  <<" GPU "<< p_ind_temp[i] << " CPU "<< p_ind[tiled_ind[i]]<<endl;  
   // //  //mpi_barrier();

   //  for (int i = nnz_loc+nnz_in_rank -1; i > nnz_loc+nnz_in_rank -3  ; --i)
   //           cout << "rank and idx: " <<mpi_rank <<" " << i << " "  <<" GPU "<< p_ind_temp[i] << " CPU "<< p_ind[tiled_ind[i]]<<endl;  
    mpi_barrier();
//}
    long diff_tot = 0; 
    for (int i = nnz_loc; i < nnz_in_rank+nnz_loc; ++i){
        //if(p_ind_temp[i] != 0)
        {
           if(abs(p_ind_temp[i]-p_ind[tiled_ind[i]]) > .0001){
                diff_tot ++;
                //if(diff_tot < 5)
                //if(mpi_rank == 1)
            }
            //if(mpi_rank == 1)
                //printf("CPU GPU diff %d:  %f %f %f \n", i, p_ind_temp[i], p_ind[tiled_ind[i]],p_ind_temp[i]-p_ind[tiled_ind[i]] );

        }
    }
    mpi_barrier();
    if(diff_tot > 0)
    cout << "diff values in CPU and GPU in machine: " << mpi_rank  <<" " << diff_tot<< endl;
    MPI_Finalize();
    cout << endl;
    //freeing device allocation
    cudaFree( d_row_ptr );
    cudaFree( d_row_ind);
    cudaFree( d_col_ind);
    cudaFree( d_val);
    cudaFree(d_active_row);
    cudaFree(d_passive_row);
    cudaFree(d_lastIdx_block_tile);
    cudaFree(d_lastIdx);
    cudaFree( d_W );
    cudaFree( d_H );
    delete(rows); delete(cols);
    delete(vals);
    
}

int main(int argc, char* argv[]){ 
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
    is >> n_rows; 
    is >> n_cols; 
    is >> nnz; 
    //fp >> n_rows >> n_cols >> nnz;
    long orig_nnz=nnz, rid=0,cid=0; float vid=0;
  
    int *rows = new int[nnz];
    int *cols = new int[nnz];  
    float *vals = new float[nnz]; 
    long idx=0;
    for (long o_idx = 0; o_idx < orig_nnz; ++o_idx) {
        fp >> rid >> cid >> vid;
        rows[idx]=rid-1;
        cols[idx]=cid-1;
        vals[idx]=vid;
        idx++;
    }
    //cout << "From main: "<<n_rows << " "<<n_cols <<" "<< nnz << " tile-size: " << tile_sizeX<< " k: "<<k << " TB: "<< BLOCKSIZE<< endl;
    nnz=idx;

    init(rows, cols, vals);

}


