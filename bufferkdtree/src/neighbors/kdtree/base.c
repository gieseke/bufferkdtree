/*
 * base.c
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
 */
#include "include/base.h"

/**
 * Initializes the *params struct with the parameters provided.
 *
 * @param n_neighbors The number of nearest neighbors to be found
 * @param tree_depth The tree depth of the tree to be built
 * @param num_threads The number of threads that should be used
 * @param splitting_type The splitting type that can be used during the construction of the tree
 * @param verbosity_level The verbosity level (0==no output, 1==more output, 2==...)
 * @param *params Pointer to struct that is used to store all parameters
 *
 */
void init_extern(int n_neighbors,
		int tree_depth,
		int num_threads,
		int splitting_type,
		int verbosity_level,
		KD_TREE_PARAMETERS *params) {

	set_default_parameters(params);

	params->n_neighbors = n_neighbors;
	params->tree_depth = tree_depth;
	params->num_threads = num_threads;
	params->verbosity_level = verbosity_level;
	params->splitting_type = splitting_type;

	check_parameters(params);

	omp_set_num_threads(params->num_threads);

}

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
		KD_TREE_PARAMETERS *params) {

	kd_tree_init_tree_record(kdtree_record, params->tree_depth, Xtrain, nXtrain, dXtrain);
	kd_tree_generate_training_patterns_indices(kdtree_record);
	kd_tree_build_tree(kdtree_record, params);

}

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
		KD_TREE_PARAMETERS *params) {

	int i, j;
	int K = params->n_neighbors;

	// simply parallelize over queries
#pragma omp parallel for
	for (i = 0; i < nXtest; i++) {
		FLOAT_TYPE *tpattern = Xtest + i * dXtest;
		kd_tree_query_tree_sequential(tpattern, distances + i * K,
				indices + i * K, K, kdtree_record);
	}

	char *XI = kdtree_record->XI;
	int size_elt = dXtest * sizeof(FLOAT_TYPE) + sizeof(int);

	// create array containing the original indices
	// (cannot be parallelized due to interleaved access)
	for (i = 0; i < nXtest; i++) {
		for (j = 0; j < K; j++) {

			int index_in_tree = indices[i * K + j];
			indices[i * K + j] = *((int *) (XI + index_in_tree * size_elt + dXtest * sizeof(FLOAT_TYPE)));
			distances[i * K + j] = sqrt(distances[i * K + j]);

		}
	}

}

/**
 * Frees resources
 *
 */
void free_resources_extern(void) {

}
