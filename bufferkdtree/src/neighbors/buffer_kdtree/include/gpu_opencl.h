/* 
 * gpu.h
 */

#ifndef NEIGHBORS_BUFFER_KD_TREE_GPU_OPENCL_H_
#define NEIGHBORS_BUFFER_KD_TREE_GPU_OPENCL_H_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>

#include "base.h"
#include "types.h"
#include "util.h"

#include "../../../include/opencl.h"

/* -------------------------------------------------------------------------------- 
 * Initializes all devices at the beginning of the  querying process.
 * -------------------------------------------------------------------------------- 
 */
void init_opencl_devices(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * After having performed all queries: Free memory etc.
 * -------------------------------------------------------------------------------- 
 */
void free_opencl_devices(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Free buffers used during training.
 * --------------------------------------------------------------------------------
 */
void free_train_buffers_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Free buffers needed for querying phase.
 * --------------------------------------------------------------------------------
 */
void free_query_buffers_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);


/* --------------------------------------------------------------------------------
 * Allocates memory for testing phase.
 * --------------------------------------------------------------------------------
 */
void allocate_memory_opencl_devices(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

INT_TYPE retrieve_indices_from_buffers_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params,
		INT_TYPE all_brute, INT_TYPE *tindices_removed, INT_TYPE *fr_indices,
		INT_TYPE *to_indices);

/* -------------------------------------------------------------------------------- 
 * Processes all buffers on the GPU. It is important that only INDICES are moved
 * from the CPU to the GPU (only CPU->GPU; GPU->CPU is not necessary). Further, the
 * global distances and the indices are updated ON THE GPU.
 * -------------------------------------------------------------------------------- 
 */
void process_all_buffers_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

void process_buffers_brute_force_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params, INT_TYPE all_brute);

void process_buffers_brute_force_in_chunks_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params,
		INT_TYPE all_brute, INT_TYPE *tindices_removed, INT_TYPE n_tindices_removed,
		INT_TYPE *fr_indices, INT_TYPE *to_indices);

/* -------------------------------------------------------------------------------- 
 * Apply brute-force approach for all leaves
 * -------------------------------------------------------------------------------- 
 */
void do_brute_force_all_leaves_FIRST_gpu(INT_TYPE *test_indices,
		INT_TYPE n_test_indices,
		INT_TYPE *fr_indices,
		INT_TYPE *to_indices,
		TREE_RECORD *tree_record,
		TREE_PARAMETERS *params,
		INT_TYPE n_device_train_patterns,
		cl_mem device_train_patterns,
		INT_TYPE chunk_offset,
		INT_TYPE all_brute,
		INT_TYPE current_chunk);

void do_brute_force_all_leaves_SECOND_gpu(INT_TYPE *test_indices,
	INT_TYPE n_test_indices,
	INT_TYPE *fr_indices,
	INT_TYPE *to_indices, TREE_RECORD *tree_record, TREE_PARAMETERS *params,
	INT_TYPE n_device_train_patterns, cl_mem device_train_patterns,
	INT_TYPE chunk_offset,
	INT_TYPE all_brute, INT_TYPE current_chunk);

/* -------------------------------------------------------------------------------- 
 * Finds the next leaf indices for all test patterns indixed by all_next_indices.
 * -------------------------------------------------------------------------------- 
 */
void find_leaf_idx_batch_gpu(INT_TYPE *all_next_indices, INT_TYPE num_all_next_indices,
		INT_TYPE *ret_vals, TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Copies the arrays dist_min_global and idx_min_global from GPU to CPU
 * -------------------------------------------------------------------------------- 
 */
void get_distances_and_indices_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Writes the training patterns in a specific ordering (needed for coalesced data
 * access on the GPU etc.
 * -------------------------------------------------------------------------------- 
 */
void write_sorted_training_patterns_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

void init_train_patterns_buffers(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params, INT_TYPE chunk,
		INT_TYPE n_indices);

void copy_train_patterns_to_device(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params, INT_TYPE chunk,
		INT_TYPE start_idx, INT_TYPE end_idx);

void free_train_patterns_device(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params, INT_TYPE chunk);

int training_chunks_inactive(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

#endif /* NEIGHBORS_BUFFER_KD_TREE_GPU_OPENCL_H_ */
