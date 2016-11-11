/*
 * cpu.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
 */

#ifndef NEIGHBORS_BUFFER_KD_TREE_CPU_H_
#define NEIGHBORS_BUFFER_KD_TREE_CPU_H_

#include <stdlib.h>
#include <string.h>

#include "base.h"
#include "types.h"
#include "util.h"

/**
 * Initializes all arrays.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void init_arrays_cpu(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params);

/**
 * Brute-force nearest neigbhor search in a leaf of the tree (determined by fr_idx,
 * to_idx, and XI).
 *
 * @param fr_idx The "from" index w.r.t. the training indices
 * @param to_idx The "to" index w.r.t. the training indices
 * @param *test_patterns Array of test patterns
 * @param ntest_patterns Number of test patterns
 * @param *d_min Array of distance values that can be updated
 * @param *idx_min Array of indices that can be updated
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 */
void brute_force_leaf_cpu(INT_TYPE fr_idx,
		INT_TYPE to_idx,
		FLOAT_TYPE * test_patterns,
		INT_TYPE ntest_patterns,
		FLOAT_TYPE * d_min,
		INT_TYPE *idx_min,
		TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

/**
 * Processes all buffers on the CPU.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 */
void process_all_buffers_cpu(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params);

/**
 * Performs a brute-force in the leaves.
 *
 * @param *test_indices_removed_from_all_buffers Array of indices that were removed from all buffers
 * @param total_number_test_indices_removed Number of removed indices
 * @param *fr_indices Array of "from" indices (see paper)
 * @param *to_indices Array of "to" indices (see paper)
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 */
void do_bruteforce_all_leaves_cpu(INT_TYPE *test_indices_removed_from_all_buffers,
		INT_TYPE total_number_test_indices_removed,
		INT_TYPE *fr_indices,
		INT_TYPE *to_indices,
		TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

/**
 * If only few elements are left in the queues after the buffers have been emptied, 
 * we do a simple brute force step to compute the nearest neighbors.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 */
void process_queue_via_brute_force_cpu(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

/**
 * Finds the next leaf indices for all test patterns indixed by all_next_indices.
 *
 * @param all_next_indices Array containing indices to be processed
 * @param num_all_next_indices Number of indices
 * @param *ret_vals For each index, the next leaf buffer that needs to be processed
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void find_leaf_idx_batch_cpu(INT_TYPE *all_next_indices,
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
 */
void get_distances_and_indices_cpu(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);
		
/**
 * Writes the training patterns in a specific ordering.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 */
void write_sorted_training_patterns_cpu(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params);

#endif /* NEIGHBORS_BUFFER_KD_TREE_CPU_H_ */
