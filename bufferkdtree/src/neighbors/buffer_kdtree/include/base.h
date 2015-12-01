/* 
 * bufferkdtree.h
 */

#ifndef NEIGHBORS_BUFFER_KD_TREE_BASE_H_
#define NEIGHBORS_BUFFER_KD_TREE_BASE_H_

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
#include <sched.h>
#include <omp.h>

#include "types.h"
#include "util.h"
#include "kdtree.h"
#include "cpu.h"
#include "gpu_opencl.h"

#if USE_GPU > 0
#define PROCESS_ALL_BUFFERS process_all_buffers_gpu
#define FIND_LEAF_IDX_BATCH find_leaf_idx_batch_gpu
#define INIT_OPENCL_DEVICES(tree_record, params); init_opencl_devices(tree_record, params);
#define ALLOCATE_MEMORY_OPENCL_DEVICES(tree_record, params); allocate_memory_opencl_devices(tree_record, params);
#define FREE_OPENCL_DEVICES(tree_record, params); free_opencl_devices(tree_record, params);
#define GET_DISTANCES_AND_INDICES get_distances_and_indices_gpu
#define WRITE_SORTED_TRAINING_PATTERNS write_sorted_training_patterns_gpu
#define INIT_ARRAYS(tree_record, params);
#else
#define PROCESS_ALL_BUFFERS process_all_buffers_cpu
#define FIND_LEAF_IDX_BATCH find_leaf_idx_batch_cpu
#define INIT_OPENCL_DEVICES(tree_record, params);
#define ALLOCATE_MEMORY_OPENCL_DEVICES(tree_record, params);
#define FREE_OPENCL_DEVICES(tree_record, params);
#define GET_DISTANCES_AND_INDICES get_distances_and_indices_cpu
#define WRITE_SORTED_TRAINING_PATTERNS write_sorted_training_patterns_cpu
#define INIT_ARRAYS(tree_record, params); init_arrays_cpu(tree_record, params);
#endif

#define PRINT(params) if ((params->verbosity_level) > 0) printf

/* --------------------------------------------------------------------------------
 * Interface (extern): Initialize components
 * --------------------------------------------------------------------------------
 */
void init_extern(int n_neighbors, int tree_depth, int num_threads, int num_nXtrain_chunks, int platform_id,
		int device_id, double allowed_train_mem_percent_chunk, double allowed_test_mem_percent,
		int splitting_type, char *kernels_source_directory,
		int verbosity_level, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Builds a buffer k-d-tree.
 * -------------------------------------------------------------------------------- 
 */
void build_bufferkdtree(FLOAT_TYPE * Xtrain, INT_TYPE nXtrain, INT_TYPE dXtrain,
		TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Testing: Perform the test queries (using the buffer kd-tree).
 * -------------------------------------------------------------------------------- 
 */
void neighbors_extern(FLOAT_TYPE * Xtest, INT_TYPE nXtest, INT_TYPE dXtest,
		FLOAT_TYPE * distances, INT_TYPE ndistances, INT_TYPE ddistances,
		INT_TYPE *indices, INT_TYPE nindices, INT_TYPE dindices,
		TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Frees additional resources on host system.
 * -------------------------------------------------------------------------------- 
 */
void extern_free_resources(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Frees opencl test buffers
 * --------------------------------------------------------------------------------
 */
void extern_free_query_buffers(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Get maximum number of test queries that can be processed.
 * --------------------------------------------------------------------------------
 */
long get_max_nXtest_extern(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Checks if platform and device are valid
 * --------------------------------------------------------------------------------
 */
int extern_check_platform_device(int platform_id, int device_id);

#endif  /* NEIGHBORS_BUFFER_KD_TREE_BASE_H_ */
