#if USE_DOUBLE > 0
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FLOAT_TYPE double
#define MAX_FLOAT_TYPE     1.7976931348623158e+308
#define MIN_FLOAT_TYPE     -1.7976931348623158e+308
#else
#define FLOAT_TYPE float
#define MAX_FLOAT_TYPE     3.402823466e+38
#define MIN_FLOAT_TYPE     -3.402823466e+38
#endif

__kernel void do_retrieve_distances(
    int total_number_test_indices_removed, 
    __global int* test_indices_removed_from_all_buffers,     
    __global FLOAT_TYPE *dist_mins_global, 
    __global int *idx_mins_global,
    __global FLOAT_TYPE *dist_mins_global_tmp, 
    __global int *idx_mins_global_tmp
    ){

    int tid = get_global_id(0);
    if(tid >= total_number_test_indices_removed){
        return;
    }

    int j;
    int test_idx_KNN = test_indices_removed_from_all_buffers[tid] * K_NN;

    for (j=0; j<K_NN; j++){
        dist_mins_global_tmp[j*total_number_test_indices_removed + tid] = dist_mins_global[test_idx_KNN +j];
        idx_mins_global_tmp[j*total_number_test_indices_removed + tid] = idx_mins_global[test_idx_KNN +j];
    }    

}   

__kernel void do_update_distances(
    int total_number_test_indices_removed, 
    int all_brute,
    __global int* test_indices_removed_from_all_buffers,     
    __global FLOAT_TYPE * dist_mins_global, 
    __global int * idx_mins_global,
    __global FLOAT_TYPE * dist_mins_global_tmp, 
    __global int * idx_mins_global_tmp
    ){
    
    int tid = get_global_id(0);
    if(tid >= total_number_test_indices_removed){
        return;
    }

    int j;

    int test_idx_KNN = test_indices_removed_from_all_buffers[tid] * K_NN;

//    if (test_indices_removed_from_all_buffers[tid] == 0){
//        dist_mins_global[test_idx_KNN +j] = dist_mins_global_tmp[0*total_number_test_indices_removed + tid];
//        return;
//    }


    for (j=0; j<K_NN; j++){
        dist_mins_global[test_idx_KNN +j] = dist_mins_global_tmp[j*total_number_test_indices_removed + tid];
        idx_mins_global[test_idx_KNN +j] = idx_mins_global_tmp[j*total_number_test_indices_removed + tid];
    } 
    
}

__kernel void do_compute_final_distances_indices(
    int number_test_patterns,
    __global FLOAT_TYPE * dist_mins_global, 
    __global int * idx_mins_global
    ){

    int tid=get_global_id(0);
    if(tid >= number_test_patterns*K_NN){
        return;
    }

    dist_mins_global[tid] = sqrt(dist_mins_global[tid]);
    //idx_mins_global[tid] = train_indices_sorted[idx_mins_global[tid]];

}

