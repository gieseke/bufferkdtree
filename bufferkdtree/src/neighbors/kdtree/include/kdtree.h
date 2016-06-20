/*
 * kdtree.h
 */

#ifndef NEIGHBORS_KDTREE_INCLUDE_KDTREE_H_
#define NEIGHBORS_KDTREE_INCLUDE_KDTREE_H_

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "global.h"
#include "util.h"

#include "../../../include/util.h"


/* --------------------------------------------------------------------------------
 * Initializes space for the kd tree (e.g., nodes and leaves)
 * --------------------------------------------------------------------------------
 */
void kd_tree_init_tree_record(KD_TREE_RECORD *record, int kd_tree_depth,
		FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain);

/* --------------------------------------------------------------------------------
 * Frees space for the kd tree (e.g., nodes and leaves)
 * --------------------------------------------------------------------------------
 */
void kd_tree_free_tree_record(KD_TREE_RECORD *record);

/* --------------------------------------------------------------------------------
 * Finds the splitting axis.
 * --------------------------------------------------------------------------------
 */
int kd_tree_find_best_split(int depth, int left, int right, KD_TREE_RECORD *kdtree_record, KD_TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Queries the kd-tree to obtain the nearest neighbors, sequential version.
 * --------------------------------------------------------------------------------
 */
void kd_tree_query_tree_sequential(FLOAT_TYPE *test_pattern, FLOAT_TYPE *d_min,
		int *idx_min, int K, KD_TREE_RECORD *record);

/* --------------------------------------------------------------------------------
 * Computes the nearest neighbors in a leaf for a given test query.
 * --------------------------------------------------------------------------------
 */
void kd_tree_brute_force_leaf(void *XI, int dim, int fr_idx, int to_idx,
		FLOAT_TYPE *test_pattern, FLOAT_TYPE *d_min, int *idx_min, int K);

/* --------------------------------------------------------------------------------
 * Builds the kd-tree (recursive construction)
 * --------------------------------------------------------------------------------
 */
void kd_tree_build_tree(KD_TREE_RECORD *kdtree_record, KD_TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Helper method to build up the kd-tree in a recursive manner.
 * --------------------------------------------------------------------------------
 */
void kd_tree_build_recursive(KD_TREE_RECORD *kdtree_record, KD_TREE_PARAMETERS *params,
		int left, int right, int idx, int depth);

/* --------------------------------------------------------------------------------
 * Parse patterns and store the original indices (this array of both the FLOAT_TYPEs
 * and the indices) is sorted in-place during the construction of the kd-tree.
 * --------------------------------------------------------------------------------
 */
void kd_tree_generate_training_patterns_indices(KD_TREE_RECORD *kdtree_record);

/* --------------------------------------------------------------------------------
 * Sorts the training patterns in range left to right (inclusive) with respect to
 * "axis".
 * --------------------------------------------------------------------------------
 */
int kd_tree_split_training_patterns_via_pivot(void *XI, int left, int right,
		int axis, int dim);

/* --------------------------------------------------------------------------------
 * Inserts the value pattern_dist with index pattern_idx in the list nearest_dist
 * of FLOAT_TYPEs and the list nearest_idx of indices. Both lists contain at most K
 * elements.
 * --------------------------------------------------------------------------------
 */
void kd_tree_insert(FLOAT_TYPE pattern_dist, int pattern_idx,
		FLOAT_TYPE *nearest_dist, int *nearest_idx, int K);

#endif /* NEIGHBORS_KDTREE_INCLUDE_KDTREE_H_ */
