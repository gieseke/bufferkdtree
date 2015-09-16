#if USE_DOUBLE > 0
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FLOAT_TYPE double
#define FLOAT_TYPE4 double4
#define MAX_FLOAT_TYPE     1.7976931348623158e+308
#define MIN_FLOAT_TYPE     -1.7976931348623158e+308
#else
#define FLOAT_TYPE float
#define FLOAT_TYPE4 float4
#define MAX_FLOAT_TYPE     3.402823466e+38f
#define MIN_FLOAT_TYPE     -3.402823466e+38f
#endif

#define VECSIZE 4

__kernel void do_bruteforce_all_leaves_nearest_neighbors(
    int total_number_test_indices_removed,
    int n_test_indices_chunk,
    int test_indices_offset,
    int num_train_patterns,
    int all_brute,
    __global FLOAT_TYPE* train_patterns_sorted, 
    __global FLOAT_TYPE* test_patterns_subset,
    __global int *fr_indices,
    __global int *to_indices,
    __global FLOAT_TYPE * dist_mins_global_tmp,
    __global int * idx_mins_global_tmp,
    int offset
    ){
    
    // global thread id
    uint tid = get_global_id(0);
    if(tid >= n_test_indices_chunk){
        return;
    }

    tid += test_indices_offset;

    uint j;

    FLOAT_TYPE4 dist_tmp_vec, tmp_vec;
    FLOAT_TYPE dist_tmp, tmp;
    uint tmp_idx, train_idx;
    FLOAT_TYPE dist_mins_private[K_NN]; 
    uint idx_mins_private[K_NN]; 

    uint dim_mul_vecsize = VECSIZE * (DIM / VECSIZE);
    FLOAT_TYPE4 test_patt_vec[DIM / VECSIZE];
    FLOAT_TYPE test_patt_scalar[DIM - VECSIZE * (DIM / VECSIZE)];

    // from and to indices (coalesced access)
    uint fr_idx = fr_indices[tid];
    uint to_idx = to_indices[tid];

    // initialize private copies (distances and indices)    
    if (!all_brute){
        for (j=0; j<K_NN; j++){
            dist_mins_private[j] = dist_mins_global_tmp[j*total_number_test_indices_removed + tid]; //MAX_FLOAT_TYPE;
            idx_mins_private[j] = idx_mins_global_tmp[j*total_number_test_indices_removed + tid]; //0;
        }
    } else { // for final brute-force: use fresh distances!
        for (j=0; j<K_NN; j++){
            dist_mins_private[j] = MAX_FLOAT_TYPE;
            idx_mins_private[j] = 0;
        }
    }

    // generate private copy of test pattern (coalesced access)
    for (j=0; j<dim_mul_vecsize; j+=VECSIZE){
        test_patt_vec[j/VECSIZE] = (FLOAT_TYPE4)(
                                test_patterns_subset[(j+0)*total_number_test_indices_removed + tid],
                                test_patterns_subset[(j+1)*total_number_test_indices_removed + tid], 
                                test_patterns_subset[(j+2)*total_number_test_indices_removed + tid],
                                test_patterns_subset[(j+3)*total_number_test_indices_removed + tid]
                                );
    }
    for(j=0; j<DIM-dim_mul_vecsize; j++){
        test_patt_scalar[j] = test_patterns_subset[(dim_mul_vecsize+j)*total_number_test_indices_removed + tid];
    }
    
    // compute distance between all training patterns in the leaf and the test pattern    
    for (train_idx=fr_idx; train_idx<to_idx; train_idx++){

        int train_idx_offset = train_idx - offset;

        // compute distance
#if USE_DOUBLE > 0
        dist_tmp = 0.0;
        dist_tmp_vec = (FLOAT_TYPE4) (0.0,0.0,0.0,0.0);
#else
        dist_tmp = 0.0f;
        dist_tmp_vec = (FLOAT_TYPE4) (0.0f,0.0f,0.0f,0.0f);
#endif

        for (j=0; j<dim_mul_vecsize; j+=VECSIZE){
            tmp_vec = (FLOAT_TYPE4) (train_patterns_sorted[train_idx_offset*DIM + j+0], 
                                     train_patterns_sorted[train_idx_offset*DIM + j+1], 
                                     train_patterns_sorted[train_idx_offset*DIM + j+2], 
                                     train_patterns_sorted[train_idx_offset*DIM + j+3]);
            tmp_vec = tmp_vec - test_patt_vec[j/VECSIZE];
            dist_tmp_vec += tmp_vec*tmp_vec;
        }

        for (j=dim_mul_vecsize; j<DIM; j++){
            tmp = (train_patterns_sorted[train_idx_offset*DIM + j] - test_patt_scalar[j-dim_mul_vecsize]);
            dist_tmp += tmp*tmp;
        }
        dist_tmp += dist_tmp_vec.s0 + dist_tmp_vec.s1 + dist_tmp_vec.s2 + dist_tmp_vec.s3;

        // insert distance and index
        j = K_NN - 1;	   
        
        // if distances have to be updated ...
        if(dist_mins_private[j] > dist_tmp){

            dist_mins_private[j] = dist_tmp;
            idx_mins_private[j] = train_idx;
            
            for(;j>0;j--){

	            if(dist_mins_private[j-1] > dist_mins_private[j]){

		            //swap distances
		            tmp=dist_mins_private[j-1];
		            dist_mins_private[j-1]=dist_mins_private[j];
		            dist_mins_private[j]=tmp;

		            //swap indices
		            tmp_idx=idx_mins_private[j-1];
		            idx_mins_private[j-1]=idx_mins_private[j];
		            idx_mins_private[j]=tmp_idx;

	            } //else break; 
            }
        }

    }    
    
    // write distances and indices to global buffers (coalesced access)
    for (j=0; j<K_NN; j++){
        dist_mins_global_tmp[j*total_number_test_indices_removed + tid] = dist_mins_private[j]; 
        idx_mins_global_tmp[j*total_number_test_indices_removed + tid] = idx_mins_private[j];
    }

}

