/*
 * util.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
 */

#ifndef NEIGHBORS_KDTREE_INCLUDE_UTIL_H_
#define NEIGHBORS_KDTREE_INCLUDE_UTIL_H_

#include "global.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/**
 * Sets default parameters.
 *
 *@param *params Pointer to struct containg the parameters to be updated 
 */
void set_default_parameters(KD_TREE_PARAMETERS *params);

/**
 * Sanity check for parameters
 * 
 *@param *params Pointer to struct containg the parameters to be checked
 */
void check_parameters(KD_TREE_PARAMETERS *params);

/**
 * Partitions a given array based on a pivot element and a given axis
 * 
 *@param *array The array that shall be processed
 *@param count The number of elements in the array
 *@param axis The axis that shall be used
 *@param size_per_elt Number of bytes a single element occupies
 *@param pivot_value The pivot element (FLOAT_TYPE)
 */
void partition_array_via_pivot(void *array, 
		int count, 
		int axis,
		int size_per_elt, 
		FLOAT_TYPE pivot_value);
		
/**
 * Swaps two elements (used by kd_tree_split_training_patterns_via_pivot)
 *
 *@param *p1 Pointer to first element
 *@param *p2 Pointer to second element
 *@param size_elt Number of bytes per element
 */
void swap_elements(void *p1, 
		void *p2, 
		int size_elt);

/**
 * Copies an element (used by kd_tree_split_training_patterns_via_pivot)
 *
 *@param *dest Pointer to the destination
 *@param *src Pointer to the source element
 *@param size_elt Number of bytes per element
 */
void copy_element(void *dest,
		const void *src,
		int size_elt);

/**
 * Computes the square value a*a for a given a.
 * 
 *@param a The value to be squared
 *
 */
FLOAT_TYPE squared(FLOAT_TYPE a);

/**
 * Computes the distance between point a and b in R^dim
 *
 *@param *a Pointer to first point
 *@param *b Pointer to second point
 *@param dim Dimensionality of points
 */
FLOAT_TYPE kd_tree_dist(const FLOAT_TYPE *a, 
		const FLOAT_TYPE *b, 
		int dim);

#endif /* NEIGHBORS_KDTREE_INCLUDE_UTIL_H_ */
