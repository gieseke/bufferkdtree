/*
 * gpu_opencl.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 *               2013-2016 Cosmin Oancea <cosmin.oancea@di.ku.dk> 
 *               2013 Justin Heinermann <justin.heinermann@uni-oldenburg.de>
 * License: GPL v2
 *
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

/**
 * Initializes all devices at the beginning of the  querying process.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void init_opencl_devices(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params);
		
/** 
 * After having performed all queries: Free memory etc.
 * 
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * 
 */
void free_opencl_devices(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params);

/**
 * Free buffers used during training.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void free_train_buffers_gpu(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

/**
 * Free buffers needed for querying phase.
 * 
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * 
 */
void free_query_buffers_gpu(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params);

/**
 * Allocates memory for testing phase.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void allocate_memory_opencl_devices(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

/**
 * Processes all buffers on the GPU. It is important that only INDICES are moved
 * from the CPU to the GPU (only CPU->GPU; GPU->CPU is not necessary). Further, the
 * global distances and the indices are updated ON THE GPU.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void process_all_buffers_gpu(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

/**
 * Helper function that processed all buffers via different
 * methods, specified via the flag 'all_brute'.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param all_brute
 *
 */
void process_buffers_brute_force_gpu(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params,
		INT_TYPE all_brute);

/**
 * Retrieves all indices from all buffers
 * 
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param all_brute Flag specififying if all indices should be processed completely via brute force
 * @param *tindices_removed Array containing the test indices afterwards
 * @param *fr_indices Array of "from" indices
 * @param *to_indices Array of "to" indices
 */		
INT_TYPE retrieve_indices_from_buffers_gpu(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params,
		INT_TYPE all_brute,
		INT_TYPE *tindices_removed,
		INT_TYPE *fr_indices,
		INT_TYPE *to_indices);

/**
 * Processes all indices stored in all buffers
 * 
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param all_brute Flag specififying if all indices should be processed completely via brute force
 * @param *tindices_removed Array containing the test indices afterwards
 * @param n_tindices_removed Number of indices
 * @param *fr_indices Array of "from" indices
 * @param *to_indices Array of "to" indices
 * 
 */
void process_buffers_brute_force_in_chunks_gpu(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params, 
		INT_TYPE all_brute,
		INT_TYPE *tindices_removed, 
		INT_TYPE n_tindices_removed, 
		INT_TYPE *fr_indices, 
		INT_TYPE *to_indices);

/**
 * Apply brute-force approach for all leaves (first stage)
 *
 * @param *test_indices Array of test indices to be processed
 * @param n_test_indices Number of test indices
 * @param *fr_indices Array of "from" indices, one for each test index
 * @param *to_indices Array of "to" indices, one for each test index
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param n_device_train_patterns Number of training patterns on device
 * @param device_train_patterns Pointer to training patterns
 * @param chunk_offset Offset (for indices) for the given chunk
 * @param all_brute Flag specififying if all indices should be processed completely via brute force
 * @param current_chunk The current chunk (id)
 *
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

/**
 * Second stage of the brute-force processing
 * 
 * @param *test_indices Array of test indices to be processed
 * @param n_test_indices Number of test indices
 * @param *fr_indices Array of "from" indices, one for each test index
 * @param *to_indices Array of "to" indices, one for each test index
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param n_device_train_patterns Number of training patterns on device
 * @param device_train_patterns Pointer to training patterns
 * @param chunk_offset Offset (for indices) for the given chunk
 * @param all_brute Flag specififying if all indices should be processed completely via brute force
 * @param current_chunk The current chunk (id)
 *
 */
void do_brute_force_all_leaves_SECOND_gpu(INT_TYPE *test_indices,
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

/**
 * Finds the next leaf indices for all test patterns indexed by all_next_indices.
 *
 * @param *all_next_indices Array containing all indices that need to be processed next
 * @param num_all_next_indices Number of indices
 * @param *ret_vals Array containing the next leaf ids for each index
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void find_leaf_idx_batch_gpu(INT_TYPE *all_next_indices,
		INT_TYPE num_all_next_indices,
		INT_TYPE *ret_vals,
		TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

/**
 * Copies the arrays dist_min_global and idx_min_global from GPU to CPU
 * Updates the distances and indices (w.r.t the original indices)
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void get_distances_and_indices_gpu(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params);

/**
 * Writes the training patterns in a specific ordering
 * (needed for coalesced data access on the GPU etc.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void write_sorted_training_patterns_gpu(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params);

/**
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param chunk Specifies the chunk 0 or 1
 * @param n_indices Number of indices for which space shall be allocated
 *
 */
void init_train_patterns_buffers(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params,
		INT_TYPE chunk, 
		INT_TYPE n_indices);

/**
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param chunk Specifies the chunk 0 or 1
 * @param start_idx The start index
 * @param end_idx The end index
 */
void copy_train_patterns_to_device(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params, 
		INT_TYPE chunk,
		INT_TYPE start_idx, 
		INT_TYPE end_idx);

/**
 * Releases training patterns on device
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param chunk Specifies the chunk 0 or 1
 */
void free_train_patterns_device(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params, 
		INT_TYPE chunk);

/**
 * Helper method that checks of training chunks are used or not
 * 
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
int training_chunks_inactive(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

#endif /* NEIGHBORS_BUFFER_KD_TREE_GPU_OPENCL_H_ */
