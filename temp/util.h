#include <vector>
#include <algorithm>
#include <iterator>
#include <utility>  
#include <bits/stdc++.h>  
#include <time.h>
#include <sys/time.h>
using namespace std;
int dense_th = 10000;
inline double seconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
void make_HTasH(float *H, float *H_t, int n_cols, int k){
    for(long r = 0; r < n_cols; ++r){
        for(long t = 0; t < k; ++t)
            H_t[t*n_cols+r] = H[r*k+t]; //-1;
    }
}

void initial(float *X, long n, int k){
    srand48(0L);
    for(long r = 0; r < n; ++r){
        for(long t = 0; t < k; ++t)
            X[r*k+t] = 0.1*drand48(); //-1;
    }
}
void unsorted_make_CSR(int *rows, int *cols, float * vals, long nnz, long n_rows, long n_cols, int *row_ptr){
    ofstream of;
    of.open ("nytimes_tans.txt");
    
    vector < tuple <int,int,float> > items;
    std::tuple <int,int,float> ap;
    for (long idx = 0; idx < nnz; ++idx) { 
    //  cout << "orig " << rows[idx] <<" " << cols[idx] <<" "<< vals[idx] << endl;
    // //transpose   
        ap=std::make_tuple(cols[idx],rows[idx],vals[idx]);
        //regular
        //ap=std::make_tuple(rows[idx],cols[idx],vals[idx]);
        items.push_back(ap);
    }
    sort(items.begin(), items.end());

    //if CSR
    // int *nnz_row = new int[n_rows];
    long idx=0, tot=0;
    row_ptr[0]=0;
    for (int r= 0; r <n_rows; ++r){
        // nnz_row[r] =0;
        while(get<0>(items[idx])==r && idx < nnz){
            // nnz_row[r]++;
            rows[idx] = get<0>(items[idx])+1;
            cols[idx] = get<1>(items[idx])+1;
            vals[idx] = get<2>(items[idx]);
            idx++;
            
        }
        row_ptr[r+1]=idx;
        // tot += nnz_row[r];
    }
    of << n_cols <<" " << n_rows <<" " <<nnz << endl;
    for (long idx = 0; idx < nnz; ++idx) 
        of << rows[idx] <<" " << cols[idx] <<" "<< vals[idx] << endl;
    of.close();    
}
// void make_tile(smat_t &R, mat_int &tiled_bin, const int TS)

void make_CSR(int *rows, int *cols, float * vals, long nnz, long n_rows, int *row_ptr){
    //assuming sorted
    
    //if CSR
    // int *nnz_row = new int[n_rows];
    long idx=0, tot=0;
    row_ptr[0]=0;
    for (int r= 0; r <n_rows; ++r){
        // nnz_row[r] =0;
        while(rows[idx]==r && idx < nnz){
            // nnz_row[r]++;
            idx++;           
        }
        row_ptr[r+1]=idx;
        // tot += nnz_row[r];
    } 
}
//load balanced tile

int rewrite_matrix_1D(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, 
    int *new_rows, int *new_cols, float * new_vals, long nnz, long n_rows, long n_cols,
    int TS, int *tiled_ind, int * lastIdx_tile, int *active_row, int *lastIdx_block_tile, int *count,  
    int &actv_row_size, long &new_nnz, int *no_block_tile, int sh_tile_r, int *tile_block_rowSt, int block_th ){

    long new_idx = 0, idx =0;
    int max_block_inAtile =(n_rows*actv_row_size/block_th+1);

    int n_tile = n_cols/TS + 1, tile_no=0;
    int *row_lim = new int[ n_rows];
    lastIdx_tile[0] = 0; 
    int max_active_row = 0, prev_row =0, prev_idx =0;
    int last_idx =0 ;
    int g_block_count = 0, min = 999999999;
    unsigned char c[4];
    int row =0;
    int col =0;
    unsigned int final_int =0, final_row, final_col;
    // #pragma omp parallel for 
    for(int tile_lim = TS; tile_lim <= (n_cols+TS-1); tile_lim+=TS){ 
        int block_nnz =0, prev_row_lim;
        int block_count =0 ;
        int n_row_tile =0 ;
        int changed_row =0;
        tile_no = tile_lim/TS;  
        count[tile_no-1] = 0;
        tile_block_rowSt[(tile_no-1)* max_block_inAtile + 0] =0;
        for(int r = 0; r <n_rows ; ++r){ 
            if(tile_lim == TS){
                idx = row_ptr[r]; row_lim[r] = idx;}
            else  idx = row_lim[r];          
            int old_idx = new_idx;

            while(col_ind[idx] < tile_lim && idx < row_ptr[r+1] && row_ptr[r+1]-row_ptr[r]>dense_th){
                prev_idx = new_idx;
                tiled_ind[new_idx] = idx;
                new_rows[new_idx] = row_ind[idx];
                new_cols[new_idx] = col_ind[idx];

                // ******* bit mask start *******
                if(sh_tile_r < 256){
                    row = row_ind[idx];
                    col = col_ind[idx]%sh_tile_r;
                    c[0] = (col>>0) & 0xff;
                    c[1] = (row>>16) & 0xFF;
                    c[2] = (row>>8) & 0xFF;
                    c[3] = (row>>0) & 0xff;
                    final_int = ((c[1]) << 24) | ((c[2]) << 16) | c[3] << 8 | c[0];
                    new_rows[new_idx] = final_int;
                }
                // ******* bit mask finish ******
                new_vals[new_idx] = val_ind[idx];
                new_idx++;
                idx++;
                block_nnz++;
                if(block_nnz == block_th){
                    block_nnz = 0;
                    lastIdx_block_tile[(tile_no-1)* max_block_inAtile + block_count]  = new_idx;                     
                    tile_block_rowSt[(tile_no-1)* max_block_inAtile + block_count] = r ;  
                    block_count++;                                           
                    n_row_tile = 0;                  
                }
            }   
            if(idx != row_lim[r]){
                active_row[(tile_no-1) * n_rows + count[tile_no-1]++]=r;             
            }  
            row_lim[r] = idx;
            n_row_tile++;            

            if(n_row_tile >= sh_tile_r) { 
                n_row_tile = 0;
                lastIdx_block_tile[(tile_no-1)* max_block_inAtile + block_count] = new_idx; 
                block_nnz = 0;
                block_count++; 
            }  
            if( r == n_rows-1 && n_row_tile > 0 && n_row_tile < sh_tile_r && block_nnz <=block_th) {
               lastIdx_block_tile[(tile_no-1)* max_block_inAtile + block_count] = new_idx;
               block_count++;
            }        
        }
        g_block_count += block_count ;//- 1;
        no_block_tile[tile_no-1] = g_block_count;
        if(count[tile_no-1] > max_active_row) max_active_row = count[tile_no-1];
        if(count[tile_no-1] < min) min = count[tile_no-1];
        lastIdx_tile[tile_no] = new_idx;       
    }

    //  for (int i = 1; i <= 3; ++i){
    //     for (int j = 1; j <  no_block_tile[i] -no_block_tile[i-1]   ; ++j)
    //         cout << i <<" here " << tile_block_rowSt[i* max_block_inAtile + j] 
    //         << " " << lastIdx_block_tile[i* max_block_inAtile + j-1]<< endl;
    //     cout << "oye!" << endl;
    // }
    
    cout << "min-max act row " << min << " - "<< max_active_row<< " "<< nnz<< " "<< max_block_inAtile << endl;
    new_nnz = nnz;
    delete(row_lim);
    return max_active_row;
}

//wo load blanaced tile
// int rewrite_matrix_1D(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, 
//     int *new_rows, int *new_cols, float * new_vals, long nnz, long n_rows, long n_cols,
//     int TS, int *tiled_ind, int * lastIdx_tile, int *active_row, int *lastIdx_block_tile, int *count,  
//     int &actv_row_size, long &new_nnz, int *no_block_tile, int sh_tile_r, int *tile_block_rowSt ){

//     long new_idx = 0, idx =0;
//     int max_block_inAtile = n_rows/actv_row_size+1;
//     int n_tile = n_cols/TS + 1, tile_no=0;
//     int *row_lim = new int[ n_rows];
//     lastIdx_tile[0] = 0; 
//     int max_active_row = 0;
//     int last_idx =0 ;
//     int g_block_count = 0, min = 999999999;
//     unsigned char c[4];
//     int row =0;
//     int col =0;
//     unsigned int final_int =0, final_row, final_col;
//     // #pragma omp parallel for 
//     for(int tile_lim = TS; tile_lim <= (n_cols+TS-1); tile_lim+=TS){ 
//         int block_nnz =0;
//         int block_count =0 ;
//         int n_row_tile =0 ;
//         tile_no = tile_lim/TS;  
//         count[tile_no-1] = 0;
//        // cout << " tile " <<tile_no<< endl;

//         for(int r = 0; r <n_rows ; ++r){ 
//             if(tile_lim == TS){
//                 idx = row_ptr[r]; row_lim[r] = idx;}
//             else 
//                 idx = row_lim[r];
                        
//             while(col_ind[idx] < tile_lim && idx < row_ptr[r+1] && row_ptr[r+1]-row_ptr[r]>dense_th){
//                 tiled_ind[new_idx] = idx;
//                 new_rows[new_idx] = row_ind[idx];
//                 //new_rows[new_idx] = count[tile_no-1];
//                 new_cols[new_idx] = col_ind[idx];

//                 // ******* bit mask start *******
//                 //if active
//                 //row = count[tile_no-1];//row_ind[idx];
//                 //if not active
//                 if(sh_tile_r < 256){
//                     row = row_ind[idx];
//                     col = col_ind[idx]%sh_tile_r;
//                     c[0] = (col>>0) & 0xff;
//                     c[1] = (row>>16) & 0xFF;
//                     c[2] = (row>>8) & 0xFF;
//                     c[3] = (row>>0) & 0xff;
//                     final_int = ((c[1]) << 24) | ((c[2]) << 16) | c[3] << 8 | c[0];
//                     new_rows[new_idx] = final_int;
//                 }
//                 // ******* bit mask finish ******

//                 new_vals[new_idx] = val_ind[idx];
//                 new_idx++;
//                 idx++;
//                 block_nnz++;
//             }   
//             if(idx != row_lim[r]){
//                active_row[(tile_no-1) * n_rows + count[tile_no-1]++]=r;  
//                //if active
//                //n_row_tile++;              
//             }  
//             //if not active
//             n_row_tile++;            
//             row_lim[r] = idx;
 
//             if(n_row_tile >= sh_tile_r) { 
//                 n_row_tile = 0;
//                 block_nnz = 0;
//                 lastIdx_block_tile[(tile_no-1)* max_block_inAtile + block_count] = new_idx; 
           
//                 block_count++;  
//             }   
//             if(r == n_rows-1 && n_row_tile > 0 && n_row_tile < sh_tile_r){
//                 lastIdx_block_tile[(tile_no-1)* max_block_inAtile + block_count] = new_idx;
//                 block_count++;
//             } 
//         }
//         g_block_count += block_count ;//- 1;
//         no_block_tile[tile_no-1] = g_block_count;
//         if(count[tile_no-1] > max_active_row) max_active_row = count[tile_no-1];
//         if(count[tile_no-1] < min) min = count[tile_no-1];
//         lastIdx_tile[tile_no] = new_idx; 
//         //cout << tile_no <<" " << n_rows-count[tile_no-1] << endl;
        
//     }
//     cout << "min-max act row " << min << " - "<< max_active_row<< endl;
//     new_nnz = nnz;
//     delete(row_lim);
//     return max_active_row;
// }

void write_mat(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, 
    int *new_rows, int *new_cols, float * new_vals, long nnz, long n_rows, long n_cols,
    int TS, int TS_Y, int *tiled_ind, int * lastIdx_tile,  int block, long &new_nnz){
    
    ofstream myfile;
    myfile.open ("dense_pubmed250k.txt");
    myfile << "%%260 8192175 109266398 \n";
    myfile << "%%MatrixMarket matrix coordinate real general\n";
    
    int tot_act =0, d_row_counter=0;
    long new_idx = 0, idx =0;

    int *dense_row = new int[n_rows];
    int *active_col = new int[n_cols];
    for (int c = 0; c < n_cols; ++c)
        active_col[c] = 0;

    lastIdx_tile[0] = 0; 
    unsigned int final_int =0, final_row, final_col;
    long d_row =0;
    for(int holder = 0; holder <n_rows ; ++holder){ 
        if(row_ptr[holder+1]-row_ptr[holder]>dense_th)   { 
            d_row++;
            while(idx < row_ptr[holder+1] ){
                myfile <<  d_row <<" "<< col_ind[idx]+1
                <<" " <<  val_ind[idx] << endl;
                new_idx++;
                idx++;
            }
        } 
        else{
             while(idx < row_ptr[holder+1] )
                idx++;
        }  
    }
    myfile.close();
}


void make_2DBlocks(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, long nnz, long n_rows, long n_cols){
    int *new_row_ind = new int[nnz];
    int *new_col_ind = new int[nnz];
    float *new_val_ind = new float[nnz];
    int block_dimX = 2;
    int block_dimY = 2;
    int n_blockY = n_cols/block_dimX + 1;
    int n_blockX = n_rows/block_dimY + 1;
    int n_block = (n_rows/block_dimY + 1) * (n_cols/block_dimX + 1);
    int nnz_row =0 ;
    int *new_ind = new int[nnz];
    int *list = new int[n_block];
    long idx = 0; int dimx =0 , dimy=0, block_no=0;
    
    //initialization
    for (int i = 0; i < n_block; ++i)
        list[i] = 0;

    // #pragma omp parallel for 
    for(int r = 0; r <n_rows; ++r){ 
        int block_noY = r/block_dimY;
        for (long idx = row_ptr[r]; idx < row_ptr[r+1]; ++idx){
            int block_noX = col_ind[idx]/block_dimX;// - 1;  
            block_no = block_noY * n_blockX +  block_noX;
            cout << "processing " << r << " " << col_ind[idx] << " ::: "
            << block_noY << " "<<block_noX <<" "<< block_no << endl;
            list[block_no]++;
            // new_ind[n_rows * i + count[i]++] = idx;  

            // list[bucket_no]++ = idx;                
            //while((idx-tiled_bin[tile_no-1][c]) < TS && idx < R.col_ptr[c+1]){ //CHANGED for nnz tiles
        }   
    }
    for (int i = 0; i < n_block; ++i)
        cout << " adf " << i<< " "<<list[i] << endl;
}

void rewrite_matrix_2D(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, 
    int *new_rows, int *new_cols, float * new_vals, long nnz, long n_rows, long n_cols,
    int TS, int *tiled_ind, int * lastIdx_tile){
    int TS_r=2;
    long new_idx = 0, idx =0;
    int n_tile_c = n_cols/TS + 1, n_tile_r = n_rows/TS_r + 1, tile_no=0;
    int tot_tile = n_tile_c * n_tile_r;
    int *row_lim = new int[(n_tile_c+1) * n_rows];
    lastIdx_tile[0] = 0; 
    for (int i = 0; i < nnz; ++i)
        cout << "orig " << i <<" : " <<row_ind[i] <<" "<<col_ind[i] << endl;    

    // #pragma omp parallel for 
    for(int tile_lim = TS; tile_lim <= (n_cols+TS-1); tile_lim+=TS){ 
         int tile_no_c =  tile_lim/TS;
        for(int tile_lim_r = 0; tile_lim_r < n_rows+TS_r-1; tile_lim_r+=TS_r){  
            tile_no = tile_no_c * n_tile_r + tile_lim_r/TS_r; 
            for(int r = tile_lim_r; r <tile_lim_r+TS_r && r<n_rows ; ++r){ 
                if(tile_lim == TS){
                    idx = row_ptr[r]; row_lim[r] = idx;}
                else 
                    idx = row_lim[(tile_no-1) * n_rows +r];                     
                while(col_ind[idx] < tile_lim && idx < row_ptr[r+1]){
                    cout << " inside " <<r<<":" << new_idx <<" "<< idx << endl;                        
                    tiled_ind[new_idx] = idx;
                    // new_rows[new_idx] = row_ind[idx];
                    // new_cols[new_idx] = col_ind[idx];
                    // new_vals[new_idx] = val_ind[idx];
                    new_idx++;
                    idx++;
                }            
                row_lim[tile_no_c * n_rows +r] = idx;
            }  
            // lastIdx_tile[tile_no] = new_idx; 
        }
    }
    // for (int i = 0; i <10; ++i)
       //       cout  << i <<" : "<<row_ind[i] <<" " << col_ind[i] << " new: " << tiled_ind[i] 
//        <<" , "<< new_rows[i] <<" "<< new_cols[i]<< endl;
    delete(row_lim);
}
void rewrite_col_sorted_matrix(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, 
    int *new_rows, int *new_cols, float * new_vals, long nnz, long n_rows, long n_cols,
    int TS, int *tiled_ind, int * lastIdx_tile,  int block, long &new_nnz){

    long new_idx = 0, idx =0;
    int n_tile = n_cols/TS + 1, tile_no=0;
    lastIdx_tile[0] = 0; 

    // #pragma omp parallel for 
    int c =0;
    for(int tile_lim = TS; tile_lim <= (n_cols+TS-1); tile_lim+=TS){ 
        tile_no = tile_lim/TS;  
        //being lazy ..can skip the part
        for(int c = 0; c <tile_lim && c<n_cols; ++c){ 
            while(col_ind[idx] == c){           
                tiled_ind[new_idx] = idx;
                new_rows[new_idx] = row_ind[idx];
                new_cols[new_idx] = col_ind[idx];
                new_vals[new_idx] = val_ind[idx];
                new_idx++;
                idx++;
            }
        }                    
        lastIdx_tile[tile_no] = new_idx; 
        if(tile_no < 5)
        cout << "lastIdx_tile " << tile_no << " " <<lastIdx_tile[tile_no] << endl;
    }
    new_nnz = nnz;
}


//perfect
// void rewrite_matrix_1D(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, 
//  int *new_rows, int *new_cols, float * new_vals, long nnz, long n_rows, long n_cols,
//  int TS, int *tiled_ind, int * lastIdx_tile,  int block, long &new_nnz){

//  long new_idx = 0, idx =0;
//  int n_tile = n_cols/TS + 1, tile_no=0;
//     int *row_lim = new int[ n_rows];
//     int *count = new int[ n_tile];
//     int *active_row = new int[n_tile * n_rows];
//     lastIdx_tile[0] = 0; 

//     // #pragma omp parallel for 
//     for(int tile_lim = TS; tile_lim <= (n_cols+TS-1); tile_lim+=TS){ 
//      tile_no = tile_lim/TS;  

//      for(int r = 0; r <n_rows; ++r){ 
//          if(tile_lim == TS){
//              idx = row_ptr[r]; row_lim[r] = idx;}
//          else 
//                 idx = row_lim[r];
                        
//              while(col_ind[idx] < tile_lim && idx < row_ptr[r+1]){
//                  // if(col_ind[idx] == 1)
//           //           cout << " inside " <<col_ind[idx]<<" " << idx <<" "<< new_idx <<" "<<tile_no << endl;                        
//                 tiled_ind[new_idx] = idx;
//                 new_rows[new_idx] = row_ind[idx];
//                 new_cols[new_idx] = col_ind[idx];
//                 new_vals[new_idx] = val_ind[idx];
//                 new_idx++;
//                 idx++;
//             }   
//             if(idx != row_lim[r])  
//                 active_row[tile_no * n_rows + count[tile_no]++]=r;       
//             row_lim[r] = idx;
            
//         }
//         lastIdx_tile[tile_no] = new_idx; 
//         // if(tile_no < 5){
//         //     cout << tile_no << " tile : " << count[tile_no] << endl;
//         //     for (int i = 0; i < count[tile_no]; ++i)
//         //         cout << active_row[tile_no*n_rows+i] << endl;

//         // }
//     }
//     for (int i = lastIdx_tile[2]; i < lastIdx_tile[3]; ++i)
//     {
//         if(new_cols[i]<180 )
//             cout << "lastIdx_tile " <<i<< " " <<row_ind[i]<< " "<<col_ind[i]<<" " <<new_cols[i] << endl;
//     }

//     new_nnz = nnz;

//     // for (int i = 0; i < nnz; ++i)
//     //   cout << "old nnz " <<i<<": " <<row_ind[i] <<" "<<col_ind[i] <<" "<<val_ind[i]<< endl;

//     // for (int i = 0; i < nnz; ++i)
//     //   cout << "new nnz " << i<<": " << new_rows[i] <<" "<<new_cols[i] <<" "<<new_vals[i] << endl;
//     // for (int i = 0; i <10; ++i)
//     //       cout  << i <<" : "<<row_ind[i] <<" " << col_ind[i] << " new: " << tiled_ind[i] 
//     //    <<" , "<< new_rows[i] <<" "<< new_cols[i]<< endl;
//      delete(row_lim);
// }
// fixing last tile
// long rewrite_matrix_1D(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, 
//  int *new_rows, int *new_cols, float * new_vals, long nnz, long n_rows, long n_cols,
//  int TS, int *tiled_ind, int * lastIdx_tile, int blocksize, long &new_nnz){

//  long new_idx = 0, idx =0;
//  int n_tile = n_cols/TS + 1, tile_no=0;
//     int *row_lim = new int[ n_rows];
//     lastIdx_tile[0] = 0; 
//     long tot =0 ;

//     // #pragma omp parallel for 
//     for(int tile_lim = TS; tile_lim <= (n_cols+TS-1); tile_lim+=TS){ 
//      tile_no = tile_lim/TS; 
//         long tile_nnz = 0; 
//      for(int r = 0; r <n_rows; ++r){ 
//          if(tile_lim == TS){
//              idx = row_ptr[r]; row_lim[r] = idx;}
//          else {idx = row_lim[r];
//              // cout << " real " << r <<" "<<idx << endl;
//          } 

//              while(col_ind[idx] < tile_lim && idx < row_ptr[r+1]){
//                  // cout << " inside " <<r<<" " << new_idx <<" "<< idx << endl;                        
//                 tiled_ind[new_idx] = idx;
//                 new_rows[new_idx] = row_ind[idx];
//                 new_cols[new_idx] = col_ind[idx];
//                 new_vals[new_idx] = val_ind[idx];
//                 new_idx++;
//                 idx++;
//                 tile_nnz++;
                
//             }            
//             row_lim[r] = idx;
//         }  
//         // tot += tile_nnz;
//         // cout <<tile_nnz << " "<< tot << endl;

//         //lastIdx_tile[tile_no] = new_idx; 
//         int nnz_tile = new_idx - lastIdx_tile[tile_no-1];
//         int remainder_block = blocksize - nnz_tile % blocksize;
//         if(nnz_tile % blocksize == 0)
//          remainder_block = 0;

//         while(remainder_block >0){
//          tiled_ind[new_idx] = idx-1;
//             new_rows[new_idx] = new_rows[new_idx-1];
//             new_cols[new_idx] = new_cols[new_idx-1];
//             new_vals[new_idx] = 0;
//             // cout <<"fill up " <<new_idx <<" "<< idx <<" "<< row_ind[idx] << endl;
//             new_idx++;
//             remainder_block--;
//         }
//         lastIdx_tile[tile_no] = new_idx; 
//     }
//     cout << lastIdx_tile[0] << " "<<lastIdx_tile[1] << " "<<lastIdx_tile[2] << " " << endl;
//     new_nnz = new_idx;

//   //   for (int i = 0; i < nnz; ++i)
//      // cout << "before "<<i <<": "<< row_ind[i] <<" "<< col_ind[i] << " " << val_ind[i] << endl;
//   //   for (int i = 0; i < new_nnz; ++i)
//   //         cout << "after "<<i <<": "<< new_rows[i] <<" "<< new_cols[i] << " " << new_vals[i] << endl;

//   //   //for (int i = 16000; i <1650; ++i)
//     //       cout  << i <<" : " << lastIdx_tile[i+1]-lastIdx_tile[i] << endl;
//     delete(row_lim);

// }


void make_2Dtile(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, long nnz, long n_rows, long n_cols,
    int TS, int *row_lim){
    int *tiled_matrix = new int [TS * n_rows];
    // #pragma omp parallel for 
    for(int r = 0; r <n_rows; ++r){ 
        long idx = row_ptr[r]; 
        row_lim[r] = idx;
        for(int tile = TS; tile <= (n_cols+TS-1); tile+=TS){            
            while(col_ind[idx] < tile && idx < row_ptr[r+1]){               
                // cout << "processing: " << r <<" "<<col_ind[idx] << " "<<tile<<" "<<idx << endl;
                idx++;
            }
            int tile_no = tile/TS-1;  
            row_lim[tile_no * n_rows +r] = idx;
        }   
    }
    // for (int ii = 0; ii < 4; ++ii)
       //  for (int i = 0; i < 4; ++i)
       //    {
       //       cout << ii << "i: "<< i<<" row lim " <<row_lim[ii * n_rows +i]<< endl;
       //    }  
       //    cout << endl;


}
void comp_bin(int n_bin, int *count, int n_rows, int* row_ptr, int nnz_max){   
    int tot=0;
    int *LB = new int[n_bin];
    int *UB = new int[n_bin];
    int *rowGroupPtr=new int[n_bin*n_rows];
    for (int i = 0; i < n_bin; i++){
        count[i]=0;
        UB[i] = (1 << i) * 2 + 1;
        LB[i] = UB[i] >> 1;
    }
    LB[0] = 0;
    UB[n_bin - 1] = nnz_max + 1;
    //for 6 bin 32,64, 128,256,512, more
    
    // LB[4] = 0; ////crucial...only last 6 bins are valid
    UB[n_bin - 1] = nnz_max + 1;
    omp_set_num_threads(n_bin);  // create as many CPU threads as there are # of bins   
    #pragma omp parallel
    {
    unsigned int cpu_thread_id = omp_get_thread_num();
    int i = cpu_thread_id;
    for (int r = 0; r < n_rows; r++){
        int nnz_row = row_ptr[r+1] - row_ptr[r];
        if (nnz_row > LB[i] && nnz_row < UB[i])
            rowGroupPtr[n_rows * i + count[i]++] = r;   
    }
    }
    for (int i = 0; i < n_bin; ++i)
    {
        cout << i <<" bin limit: "<<UB[i] <<" " << count[i] << endl;
    }
    long max = 0;
    for (int r = 0; r < n_rows; r++){
        int nnz_row = row_ptr[r+1] - row_ptr[r];

            if(nnz_row > max)
                max = nnz_row;
            
    }
    cout << "max" << " "<<max <<endl; 
        
}