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

#define INIT_STACK(); __global int *stack = all_stacks + test_idx*TREE_DEPTH;
#define COPY_STACK_BACK();

typedef struct tree_node {

	int axis;
	FLOAT_TYPE splitting_value;

} TREE_NODE;

// KERNEL IS USED
__kernel void find_leaf_idx_batch(
					__global int* all_next_indices,
					int num_indices,    
					__global FLOAT_TYPE* test_patterns,					
					__global int* ret_vals, // we need write access here   
					__global int* all_depths, // we need write access here
					__global int* all_idxs, // we need write access here
					__global int* all_stacks, // we need write access here					
					int kd_tree_depth,
					__global FLOAT_TYPE* d_min, // we need write access here
					__global int* idx_min, // we need write access here
					__global TREE_NODE* nodes,
					__global FLOAT_TYPE* leaves
					)

{
    // kernel index
    int tid = get_global_id(0);    
    if(tid>=num_indices){return;}        
    // get local workgroup id    
    int lid = get_local_id(0);    
    // counter variable
    unsigned int k;    
    // get the test index
    unsigned int test_idx = all_next_indices[tid];
 
    // private copy seems to be slower...
    //FLOAT_TYPE test_patt[DIM];       
    //for(k=DIM;k--;){test_patt[k] = test_patterns[test_idx*DIM + k];}
    __global FLOAT_TYPE *test_patt = test_patterns+test_idx*DIM;        

    // private copy of index (global one is updated at the end)
    unsigned int idx_local = *(all_idxs+test_idx);
    // private copy of depth (global one is updated at the end)    
    int depth_local = *(all_depths+test_idx);

    INIT_STACK();
    int axis = 0;
    unsigned int status = 0;

    
    for (k=MAX_VISITED;k--;){
        if (depth_local == TREE_DEPTH){
            ret_vals[tid] = idx_local - NUM_NODES; // leaf_idx
            idx_local = (idx_local-1)*0.5;
            depth_local = depth_local-1;
            break;
        }
        status = stack[depth_local];

        //axis = depth_local % DIM;
        axis = nodes[idx_local].axis;

        if (status == 2){
            // if left child was visited first, we have to check the right side of the node
            FLOAT_TYPE tmp = test_patt[axis] - nodes[idx_local].splitting_value;
            if(tmp*tmp <= d_min[test_idx*NUM_NEIGHBORS + NUM_NEIGHBORS-1] ){
                // the test instance is nearer to the median than to the nearest neighbors
                // found so far: go right!
                idx_local = 2*(idx_local)+2;
                stack[depth_local] = 3;
                depth_local = depth_local+1;
            }else{ // if the median is far away, we can go up (without checking the right side)
                idx_local = (idx_local-1)*0.5;
                stack[depth_local] = 0;
                depth_local = depth_local-1;
                if(depth_local<0) {ret_vals[tid]=-1;break;}
            }
        } else if (status == 1){
            FLOAT_TYPE tmp = test_patt[axis] - nodes[idx_local].splitting_value;
            // if right child was visited first, we have to check the left side of the node
            if( tmp*tmp <= d_min[test_idx*NUM_NEIGHBORS + NUM_NEIGHBORS-1]){                                        
                // the test instance is nearer to the median than to the nearest neighbors
                // found so far: go left!
                idx_local = 2*(idx_local)+1;
                stack[depth_local] = 3;
                depth_local = depth_local+1;
            }else{ // if the median is far away, we can go up (without checking the left side)
                idx_local = (idx_local-1)*0.5;
                stack[depth_local] = 0;
                depth_local = depth_local-1;      
                if(depth_local<0) {ret_vals[tid]=-1;break;}
            }
        } else if (status==3) { 
            // both children have been visited...time to go up!
            idx_local = (idx_local-1)*0.5;
            stack[depth_local] = 0;
            depth_local = depth_local-1;      
            if(depth_local<0) {ret_vals[tid]=-1;break;}
        } else { // status == 0
            // if first visit
            if (test_patt[axis] - nodes[idx_local].splitting_value >= 0){
                // go right (down)
                stack[depth_local] = 1;
                idx_local = 2*(idx_local)+2;
                depth_local = depth_local+1;
            } else {
                // go left (down)
                stack[depth_local] = 2;
                idx_local = 2*(idx_local)+1;
                depth_local = depth_local+1;
            }
        }        
    } // end while 
    // copy values back
    *(all_idxs+test_idx) = idx_local;
    *(all_depths+test_idx) = depth_local;
    COPY_STACK_BACK();
}


