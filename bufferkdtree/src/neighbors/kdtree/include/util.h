/*
 * util.h
 *
 *  Created on: 31.10.2014
 *      Author: Fabian Gieseke
 */

#ifndef NEIGHBORS_KDTREE_INCLUDE_UTIL_H_
#define NEIGHBORS_KDTREE_INCLUDE_UTIL_H_

#include "global.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* --------------------------------------------------------------------------------
 * Sets default parameters.
 * --------------------------------------------------------------------------------
 */
void set_default_parameters(KD_TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Checks default parameters.
 * --------------------------------------------------------------------------------
 */
void check_parameters(KD_TREE_PARAMETERS *params);


/* --------------------------------------------------------------------------------
 * Swaps two elements (used by kd_tree_split_training_patterns_via_pivot)
 * --------------------------------------------------------------------------------
 */
void swap_elements(void *p1, void *p2, int size_elt);

/* --------------------------------------------------------------------------------
 * Copies an element (used by kd_tree_split_training_patterns_via_pivot)
 * --------------------------------------------------------------------------------
 */
void copy_element(void *dest, const void *src, int size_elt);

void partition_array_via_pivot(void *array, int count, int axis, int size_per_elt, FLOAT_TYPE pivot_value);

/* --------------------------------------------------------------------------------
 * Helper function.
 * --------------------------------------------------------------------------------
 */
FLOAT_TYPE squared(FLOAT_TYPE a);

/* --------------------------------------------------------------------------------
 * Computes the distance between point a and b in R^dim
 * --------------------------------------------------------------------------------
 */
FLOAT_TYPE kd_tree_dist(const FLOAT_TYPE *a, const FLOAT_TYPE *b, int dim);

#endif /* NEIGHBORS_KDTREE_INCLUDE_UTIL_H_ */
