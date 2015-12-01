/* 
 * kdtree.h
 */

#ifndef NEIGHBORS_BUFFER_KD_TREE_KDTREE_H_
#define NEIGHBORS_BUFFER_KD_TREE_KDTREE_H_

#include "base.h"
#include "types.h"
#include "../../../include/util.h"

/* -------------------------------------------------------------------------------- 
 * Builds the kd-tree (recursive construction).
 * -------------------------------------------------------------------------------- 
 */
void kd_tree_build_tree(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Finds the splitting axis.
 * --------------------------------------------------------------------------------
 */
void kd_tree_find_best_split(int depth, int left, int right,
		TREE_RECORD *tree_record, TREE_PARAMETERS *params,
		int *axis, int *pivot_idx, FLOAT_TYPE *splitting_value);

/* -------------------------------------------------------------------------------- 
 * Helper method to build up the kd-tree in a recursive manner.
 * -------------------------------------------------------------------------------- 
 */
void kd_tree_build_recursive(TREE_RECORD *tree_record, TREE_PARAMETERS *params,
		INT_TYPE left, INT_TYPE right, INT_TYPE idx, INT_TYPE depth);

/* -------------------------------------------------------------------------------- 
 * Parse patterns and store the original indices (this array of both the FLOAT_TYPEs
 * and the indices) is sorted in-place during the construction of the kd-tree.
 * -------------------------------------------------------------------------------- 
 */
void kd_tree_generate_training_patterns_indices(void *XI, FLOAT_TYPE * X, INT_TYPE n,
		INT_TYPE dim);

/* -------------------------------------------------------------------------------- 
 * Sorts the training patterns in range left to right (inclusive) with respect to
 * "axis".
 * -------------------------------------------------------------------------------- 
 */
INT_TYPE kd_tree_split_training_patterns_via_pivot(void *XI, INT_TYPE left, INT_TYPE right,
		INT_TYPE axis, INT_TYPE dim);



#endif /* NEIGHBORS_BUFFER_KD_TREE_KDTREE_H_ */
