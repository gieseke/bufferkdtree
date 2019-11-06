/*
 * base.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
 */

#ifndef NEIGHBORS_KDTREE_INCLUDE_BASE_H_
#define NEIGHBORS_KDTREE_INCLUDE_BASE_H_

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#include "global.h"
#include "util.h"
#include "kdtree.h"

/**
 * Initializes the *params struct with the parameters provided.
 *
 * @param n_neighbors The number of nearest neighbors to be found
 * @param tree_depth The tree depth of the tree to be built
 * @param max_leaves The maximum number of leaf visits for each query.
 * @param num_threads The number of threads that should be used
 * @param splitting_type The splitting type that can be used during the construction of the tree
 * @param verbosity_level The verbosity level (0==no output, 1==more output, 2==...)
 * @param *params Pointer to struct that is used to store all parameters
 *
 */
void init_extern(int n_neighbors,
		int tree_depth,
		int max_leaves,
		int num_threads,
		int splitting_type,
		int verbosity_level,
		KD_TREE_PARAMETERS *params);

/**
 * Builds a k-d-tree
 *
 * @param *Xtrain Pointer to array of type "FLOAT_TYPE" (either "float" or "double")
 * @param nXtrain Number of rows in *X (i.e., points/patterns)
 * @param dXtrain Number of columns in *X (one column per point/pattern)
 * @param *kdtree_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 */
void fit_extern(FLOAT_TYPE *Xtrain,
		int nXtrain,
		int dXtrain,
		KD_TREE_RECORD *kdtree_record,
		KD_TREE_PARAMETERS *params);

/**
 * Interface (extern): Computes the k nearest neighbors for a given set of test points
 * stored in *Xtest and stores the results in two arrays *distances and *indices.
 *
 * @param *Xtest Pointer to the set of query/test points (stored as FLOAT_TYPE)
 * @param nXtest The number of query points
 * @param dXtest The dimension of each query point
 * @param *distances The distances array (FLOAT_TYPE) used to store the computed distances
 * @param ndistances The number of query points
 * @param ddistances The number of distance values for each query point
 * @param *indices Pointer to arrray storing the indices of the k nearest neighbors for each query point
 * @param nindices The number of query points
 * @param dindices The number of indices comptued for each query point
 * @param *kdtree_record Pointer to struct storing all relevant information for model
 * @param *params Pointer to struct containing all relevant parameters
 *
 */
void neighbors_extern(FLOAT_TYPE * Xtest,
		int nXtest,
		int dXtest,
		FLOAT_TYPE * distances,
		int ndistances,
		int ddistances,
		int *indices,
		int nindices,
		int dindices,
		KD_TREE_RECORD *kdtree_record,
		KD_TREE_PARAMETERS *params);

/**
 * Frees resources
 *
 */
void free_resources_extern(void);

#endif /* NEIGHBORS_KDTREE_INCLUDE_BASE_H_ */
