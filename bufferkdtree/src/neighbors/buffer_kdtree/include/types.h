/* 
 * types.h
 */

#ifndef NEIGHBORS_BUFFER_KD_TREE_TYPES_H_
#define NEIGHBORS_BUFFER_KD_TREE_TYPES_H_

#include "../../../include/timing.h"
#include "../../../include/float.h"
#include "../../../include/opencl.h"

#include <string.h>

// helper macros
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define SPLITTING_TYPE_CYCLIC 0
#define SPLITTING_TYPE_LONGEST_BOX 1

// 1024*1024*1024 or 1E9
#define MEM_GB 1.073741824E9

#define DO_ALL_BRUTE_NO 0
#define DO_ALL_BRUTE_YES 1

#define TRAIN_CHUNK_0 0
#define TRAIN_CHUNK_1 1

#ifndef MAX_CHUNK_BRUTE_KERNEL
#define MAX_CHUNK_BRUTE_KERNEL 1048576
#endif

// workgroup sizes
#ifndef WORKGROUP_SIZE_BRUTE
#define WORKGROUP_SIZE_BRUTE 256
#endif

#ifndef WORKGROUP_SIZE_LEAVES
#define WORKGROUP_SIZE_LEAVES 32
#endif

#ifndef WORKGROUP_SIZE_UPDATE
#define WORKGROUP_SIZE_UPDATE 16
#endif

#ifndef WORKGROUP_SIZE_COPY_INIT
#define WORKGROUP_SIZE_COPY_INIT 32
#endif

#ifndef WORKGROUP_SIZE_COMBINE
#define WORKGROUP_SIZE_COMBINE 64
#endif

#ifndef WORKGROUP_SIZE_TEST_SUBSET
#define WORKGROUP_SIZE_TEST_SUBSET 32
#endif

#ifndef WORKGROUP_SIZE_COPY_DISTS_INDICES
#define WORKGROUP_SIZE_COPY_DISTS_INDICES 32
#endif

#define LEAF_WIDTH 2

// if changed -> modify init_extern, swig interface, and OpenCL kernels
#define INT_TYPE int
#define UINT_TYPE unsigned int

// struct for storing a single node
typedef struct tree_node {

	int axis;
	FLOAT_TYPE splitting_value;

} TREE_NODE;

typedef struct {
	INT_TYPE size;
	INT_TYPE start;
	INT_TYPE end;
	INT_TYPE *items;
} circular_buffer;

// struct for input parameters
typedef struct tree_parameters {

	INT_TYPE n_neighbors;
	INT_TYPE tree_depth;
	INT_TYPE num_threads;
	INT_TYPE splitting_type;
	char *kernels_source_directory;
	INT_TYPE verbosity_level;
	INT_TYPE platform_id;
	INT_TYPE device_id;

	// do brute force as soon as only bf_remaining_threshold elements are in test queue
	UINT_TYPE bf_remaining_threshold;

	// the number of desired training chunks
	INT_TYPE n_train_chunks;

	FLOAT_TYPE allowed_train_mem_percent_chunk;
	FLOAT_TYPE allowed_test_mem_percent;

} TREE_PARAMETERS;

// struct for input parameters
typedef struct tree_record {

	// array containing pairs of patterns and (original) indices
	void *XtrainI;

	// in-place sorted training patterns
	FLOAT_TYPE *Xtrain_sorted;

	// in-place sorted training indices
	INT_TYPE *Itrain_sorted;

	// dimension of patterns
	INT_TYPE dXtrain;

	// number of training patters
	INT_TYPE nXtrain;

	// test patterns
	FLOAT_TYPE *Xtest;

	// number of test patterns
	UINT_TYPE nXtest;

	// top tree stored as array nodes (median values w.r.t. axes)
	TREE_NODE *nodes;

	// number of (internal) nodes
	INT_TYPE n_nodes;

	// leaves of the kdtree (left,right for each leaf)
	FLOAT_TYPE *leaves;

	// number of leaves
	INT_TYPE n_leaves;

	// max number of visits in tree (needed for kernels)
	INT_TYPE max_visited;

	// size of circular buffers for query indices (i.e., how many indices we can store at each leaf)
	INT_TYPE leaves_initial_buffer_sizes;

	// threshold for the buffers: we have to have many indices in the leaf in
	// order to make the processing efficient. When should we empty a buffer?
	INT_TYPE leaves_buffer_sizes_threshold;

	// estimate: how many elements can be put into the buffers in each round?
	UINT_TYPE approx_number_of_avail_buffer_slots;

	// flag if one buffer is almost full
	UINT_TYPE buffer_full_warning;

	// test buffers
	circular_buffer **buffers;

	// queue "reinsert"
	circular_buffer queue_reinsert;

	// queue "insert", implemented as counter current_test_index
	INT_TYPE current_test_index;

	// all stacks
	INT_TYPE *all_stacks;

	// all depths
	INT_TYPE *all_depths;

	// all idxs
	INT_TYPE *all_idxs;

	// global distances to be returned by a query
	FLOAT_TYPE *dist_mins_global;

	// global indices to be returned by a query
	INT_TYPE *idx_mins_global;

	// return values of find_leaf_batch
	INT_TYPE *leaf_indices_batch_ret_vals;

	// counter for leaf calls
	INT_TYPE find_leaf_idx_calls;

	// counter for empty buffer calls
	INT_TYPE empty_all_buffers_calls;

	// global opencl variables
	cl_platform_id gpu_platform;
	cl_device_id gpu_device;
	cl_context gpu_context;
	cl_command_queue gpu_command_queue;
	cl_command_queue gpu_command_queue_chunk_0;
	cl_command_queue gpu_command_queue_chunk_1;
	DEVICE_INFOS device_infos;

	// kernels
	cl_kernel brute_nn_kernel;
	cl_kernel update_dist_kernel;
	cl_kernel retrieve_dist_kernel;
	cl_kernel find_leaves_kernel;
	cl_kernel generate_test_subset_kernel;
	cl_kernel init_dists_kernel;
	cl_kernel init_stacks_kernel;
	cl_kernel init_depths_idxs_kernel;
	cl_kernel compute_final_dists_idxs_kernel;

	// train buffers
	INT_TYPE current_chunk_id;
	cl_mem device_train_patterns_chunk_0; // n_train_chunks * dXtrain * sizeof(FLOAT_TYPE)
	cl_mem device_train_patterns_chunk_1; // n_train_chunks * dXtrain * sizeof(FLOAT_TYPE)
	FLOAT_TYPE *host_pinned_train_patterns_chunk_0;
	FLOAT_TYPE *host_pinned_train_patterns_chunk_1;
	INT_TYPE n_patts_per_chunk;
	cl_mem device_nodes; // n_nodes * sizeof(FLOAT_TYPE)
	cl_mem device_leave_bounds; // n_leaves * 2 * sizeof(FLOAT_TYPE)

	// test buffers
	cl_mem device_test_patterns; // nXtest * dXtest * sizeof(FLOAT_TYPE)
	cl_mem device_d_mins; // nXtest * n_neighbors * sizeof(FLOAT_TYPE)
	cl_mem device_idx_mins; // nXtest * n_neighbors * sizeof(INT_TYPE)
	cl_mem device_all_stacks; // nXtest * tree_depth * sizeof(INT_TYPE)
	cl_mem device_all_depths; // nXtest * sizeof(INT_TYPE)
	cl_mem device_all_idxs; // nXtest * sizeof(INT_TYPE)
	cl_mem device_idx_mins_tmp; // nXtest * n_neighbors * sizeof(FLOAT_TYPE)
	cl_mem device_dist_mins_tmp; // nXtest * n_neighbors * sizeof(INT_TYPE)
	cl_mem device_test_patterns_subset_tmp; // nXtest * dXtest * sizeof(FLOAT_TYPE)

	// tmp
	cl_mem device_test_indices_removed_from_all_buffers; // n_test_indices * sizeof(INT_TYPE)
	cl_mem device_all_next_indices; // tree_record->approx_number_of_avail_buffer_slots * sizeof(INT_TYPE)
	cl_mem device_ret_vals; // tree_record->approx_number_of_avail_buffer_slots * sizeof(INT_TYPE)
	cl_mem device_fr_indices; // nXtest * sizeof(INT_TYPE)
	cl_mem device_to_indices; // nXtest * sizeof(INT_TYPE)

	// Memory Consumption (roughly)
	// train: 2* n_train_chunks * dXtrain * sizeof(FLOAT_TYPE)
	//        + n_nodes * sizeof(FLOAT_TYPE)
	//        + n_leaves * 2 * sizeof(FLOAT_TYPE)
	// test:  nXtest * sizeof(FLOAT_TYPE) * (2 * dXtest + 2 * n_neighbors)
	//        nXtest * sizeof(INT_TYPE) * (2 * n_neighbors + tree_depth + 2)

	// tree_record->approx_number_of_avail_buffer_slots = 5 * tree_record->leaves_initial_buffer_sizes < 5 * 2^20 = 5E6
	// tmp: nXtest * 3 * sizeof(INT_TYPE) + 5E6 * sizeof(INT_TYPE)

	int device_query_buffers_allocated;

	TIMER timers[30];
	INT_TYPE counters[10];

} TREE_RECORD;

#ifndef USE_GPU
#define USE_GPU 0
#endif


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#endif /* NEIGHBORS_BUFFER_KD_TREE_TYPES_H_ */
