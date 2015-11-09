/* 
 * cpu.h
 */

#ifndef NEIGHBORS_BUFFER_KD_TREE_CPU_H_
#define NEIGHBORS_BUFFER_KD_TREE_CPU_H_

#include <stdlib.h>
#include <string.h>

#include "base.h"
#include "types.h"
#include "util.h"

/* -------------------------------------------------------------------------------- 
 * Initializes all arrays.
 * -------------------------------------------------------------------------------- 
 */
void init_arrays_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Brute-force nearest neigbhor search in a leaf of the tree (determined by fr_idx,
 * to_idx, and XI).
 * -------------------------------------------------------------------------------- 
 */
void brute_force_leaf_cpu(INT_TYPE fr_idx, INT_TYPE to_idx, FLOAT_TYPE * test_patterns,
		INT_TYPE ntest_patterns, FLOAT_TYPE * d_min, INT_TYPE *idx_min, TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Processes all buffers on the CPU.
 * -------------------------------------------------------------------------------- 
 */
void process_all_buffers_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Performs a brute-force in the leaves.
 * -------------------------------------------------------------------------------- 
 */
void do_bruteforce_all_leaves_cpu(INT_TYPE *test_indices_removed_from_all_buffers,
		INT_TYPE total_number_test_indices_removed, INT_TYPE *fr_indices, INT_TYPE *to_indices,
		TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Updates the distances and indices (w.r.t the original indices)
 * -------------------------------------------------------------------------------- 
 */
void get_distances_and_indices_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * If only few elements are left in the queues after the buffers have been emptied, 
 * we do a simple brute force step to compute the nearest neighbors.
 * -------------------------------------------------------------------------------- 
 */
void process_queue_via_brute_force_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Finds the next leaf indices for all test patterns indixed by all_next_indices.
 * -------------------------------------------------------------------------------- 
 */
void find_leaf_idx_batch_cpu(INT_TYPE *all_next_indices, INT_TYPE num_all_next_indices,
		INT_TYPE *ret_vals, TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Writes the training patterns in a specific ordering.
 * -------------------------------------------------------------------------------- 
 */
void write_sorted_training_patterns_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

#endif /* NEIGHBORS_BUFFER_KD_TREE_CPU_H_ */
