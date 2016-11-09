/*
 * kdtree.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
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


/** 
 * Initializes space for the kd tree (e.g., nodes and leaves)
 * 
 * @param *record The kd tree struct instance
 * @param kd_tree_depth The desired tree depth
 * @param *Xtrain Pointer to array containing the patterns (as FLOAT_TYPE)
 * @param nXtrain Number of patterns
 * @param nXtrain Dimensionality of each pattern
 */
void kd_tree_init_tree_record(KD_TREE_RECORD *record, 
		int kd_tree_depth,
		FLOAT_TYPE *Xtrain, 
		int nXtrain, 
		int dXtrain);

/**
 * Frees space for the kd tree (e.g., nodes and leaves)
 *
 *@param *record A kd tree record instance
 */
void kd_tree_free_tree_record(KD_TREE_RECORD *record);

/**
 * Builds the kd-tree (recursive construction)
 * 
 * @param *record A kd tree record instance
 * @param *params Associated kd tree parameters
 */
void kd_tree_build_tree(KD_TREE_RECORD *kdtree_record, 
		KD_TREE_PARAMETERS *params);

/**
 * Function to recursively build a kd tree
 * 
 * @param *record A kd tree record instance
 * @param *params Associated kd tree parameters
 * @param left left bound for training patterns
 * @param right right bound for training patterns
 * @param idx Node index
 * @param depth Current tree depth
 */
void kd_tree_build_recursive(KD_TREE_RECORD *kdtree_record, 
		KD_TREE_PARAMETERS *params,
		int left, 
		int right, 
		int idx, 
		int depth);

/**
 * Queries the kd-tree to obtain the nearest neigbhors, sequential version.
 * 
 *@param *test_pattern The test pattern for which the nearest neighbors shall be computed
 *@param *d_min Float array that shall contain the distances
 *@param *idx_min Integer array to store the nearest neighbor indices
 *@param K Number of nearest neigbhors that shall be found
 *@param *record A kd tree record instance
 */
void kd_tree_query_tree_sequential(FLOAT_TYPE *test_pattern, 
		FLOAT_TYPE *d_min,
		int *idx_min, 
		int K, 
		KD_TREE_RECORD *record);

/**
 * Brute-force nearest neigbhor search in a leaf of the tree (determined by fr_idx,
 * to_idx, and XI).
 *
 *@param *XI Pointer to array containing the patterns and the associated original training indices
 *@param dim Dimensionality of the patterns
 *@param fr_idx The from index in the array (left bound)
 *@param to_idx The to index in the array (right bound)
 *@param *test_pattern Pointer to test pattern
 *@param *d_min Pointer to array that shall contain the distances afterwards
 *@param *idx_min Pointer to array that shall contain the indices afterwards
 *@param K Number of nearest neighbors
 */
void kd_tree_brute_force_leaf(void *XI,
		int dim,
		int fr_idx,
		int to_idx,
		FLOAT_TYPE *test_pattern,
		FLOAT_TYPE *d_min,
		int *idx_min,
		int K);
								
/**
 * Finds the splitting axis.
 * 
 *@param depth The current tree depth
 *@param left The left bound
 *@param right The right bound
 *@param *kdtree_record Record storing the tree instances
 *@param *params Struct containing the parameters
 */
int kd_tree_find_best_split(int depth, 
		int left, 
		int right, 
		KD_TREE_RECORD *kdtree_record, 
		KD_TREE_PARAMETERS *params);

/**
 * Parse patterns and store the original indices (this array of both the FLOAT_TYPEs
 * and the indices) is sorted in-place during the construction of the kd-tree.
 *
 *@param *kdtree_record Record storing the tree instances
 */
void kd_tree_generate_training_patterns_indices(KD_TREE_RECORD *kdtree_record);

/**
 * Sorts the training patterns in range left to right (inclusive) with respect to
 * "axis".
 * 
 *@param *XI Pointer to array containing the patterns and the associated original training indices
 *@param left Left bound
 *@param right Right bound
 *@param axis The current axis
 *@param dim The dimensionality of the patterns
 */
int kd_tree_split_training_patterns_via_pivot(void *XI, 
		int left, 
		int right,
		int axis, 
		int dim);

/**
 * Inserts the value pattern_dist with index pattern_idx in the list nearest_dist
 * of FLOAT_TYPEs and the list nearest_idx of indices. Both lists contain at most K
 * elements.
 * 
 *@param pattern_dist Distance to the (current) pattern
 *@param pattern_idx The index of the (current) pattern
 *@param *nearest_dist The array of floats to be updated
 *@param *nearest_idx The array of indices to be updated
 *@param K The number of nearest neighbors
 */
void kd_tree_insert(FLOAT_TYPE pattern_dist, 
		int pattern_idx,
		FLOAT_TYPE *nearest_dist, 
		int *nearest_idx, 
		int K);

#endif /* NEIGHBORS_KDTREE_INCLUDE_KDTREE_H_ */
