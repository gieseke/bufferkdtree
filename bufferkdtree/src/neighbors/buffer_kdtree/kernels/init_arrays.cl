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

__kernel void do_init_distances(
    int num_elts, 
    __global FLOAT_TYPE * dist_mins_global, 
    __global int * idx_mins_global
    ){

    int tid=get_global_id(0);
    if(tid>=num_elts){
        return;
    }

    dist_mins_global[tid] = MAX_FLOAT_TYPE;
    idx_mins_global[tid] = 0;        

}

__kernel void do_init_allstacks(
    int num_elts,
    __global int * all_stacks
    ){

    int tid = get_global_id(0);
    if(tid >= num_elts){
        return;
    }    

    all_stacks[tid] = 0;
}

__kernel void do_init_depths_idx(
    int num_elts,
    __global int * all_depths,
    __global int * all_idxs    
    ){

    int tid = get_global_id(0);
    if(tid >= num_elts){
        return;
    }    

    all_depths[tid] = 0;
    all_idxs[tid] = 0;

}

__kernel void init_array_value(
    int n,
    __global int * array,
    int value
    ){

    int tid=get_global_id(0);
    if(tid>=n){
        return;
    }    

    array[tid] = value;

}

