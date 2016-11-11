/*
 * kdtree.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
 */

#ifndef NEIGHBORS_BUFFER_KD_TREE_KDTREE_H_
#define NEIGHBORS_BUFFER_KD_TREE_KDTREE_H_

#include "base.h"
#include "types.h"
#include "../../../include/util.h"

/**
 * Builds the kd-tree (recursive construction)
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void kd_tree_build_tree(TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params);

/**
 * Finds the optimal splitting axis.
 * 
 * @param depth The current depth of the splitting process
 * @param left The left bound of values to be tested
 * @param right The right bound of values to be tested
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param *axis The current axis
 * @param *pivot_idx The output pivot index
 * @param *splitting_value The output splitting value
 *  
 */
void kd_tree_find_best_split(int depth, 
		int left, 
		int right,
		TREE_RECORD *tree_record, 
		TREE_PARAMETERS *params,
		int *axis, 
		int *pivot_idx, 
		FLOAT_TYPE *splitting_value);

/**
 * Helper method to build up the kd-tree in a recursive manner.
 *
 * @param *tree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 * @param left The left bound of values to be tested
 * @param right The right bound of values to be tested
 * @param idx The current node index
 * @param depth The current tree depth
 */
void kd_tree_build_recursive(TREE_RECORD *tree_record,
		TREE_PARAMETERS *params,
		INT_TYPE left,
		INT_TYPE right,
		INT_TYPE idx,
		INT_TYPE depth);

/** 
 * Parse patterns and store the original indices (this array of both the FLOAT_TYPEs
 * and the indices) is sorted in-place during the construction of the kd-tree.
 *  
 * @param *XI Pointer to array containing patterns and indices
 * @param *X Pointer to array containing patterns
 * @param n Number of elements in arrays
 * @param dim Dimensionality of patterns
 */
void kd_tree_generate_training_patterns_indices(void *XI, 
		FLOAT_TYPE * X, 
		INT_TYPE n,
		INT_TYPE dim);

/**
 * Sorts the training patterns in range left to right (inclusive) with respect to
 * "axis".
 * 
 * @param *XI Pointer to array containing patterns and indices
 * @param left The left bound of points to be tested
 * @param right The right bound of points to be tested
 * @param axis The current axis
 * @param dim Dimensionality of patterns
 * 
 */
INT_TYPE kd_tree_split_training_patterns_via_pivot(void *XI, 
		INT_TYPE left, 
		INT_TYPE right,
		INT_TYPE axis, 
		INT_TYPE dim);

#endif /* NEIGHBORS_BUFFER_KD_TREE_KDTREE_H_ */