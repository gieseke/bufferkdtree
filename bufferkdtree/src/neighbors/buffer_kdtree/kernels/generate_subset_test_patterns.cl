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

// KERNEL IS USED        
__kernel void do_generate_subset_test_patterns(
    int total_number_test_indices_removed,
    __global int* test_indices_removed_from_all_buffers,
    __global FLOAT_TYPE * test_patterns,    
    __global FLOAT_TYPE * test_patterns_subset_tmp
    ){
    
    // get global thread id
    int tid=get_global_id(0);
    if(tid>=total_number_test_indices_removed){return;}
    // get corresponding test index
    int test_idx = test_indices_removed_from_all_buffers[tid];
    // copy element to position tid in tmp array
    int j;
    for (j=0; j<DIM; j++){
        // non-coalesced access for original test patterns (arbitrary access via test_idx not avoidable), 
        // coalesced access for test_patterns_subset_tmp
        test_patterns_subset_tmp[j*total_number_test_indices_removed + tid] = test_patterns[test_idx*DIM+j];
    }
    
  
    

}


