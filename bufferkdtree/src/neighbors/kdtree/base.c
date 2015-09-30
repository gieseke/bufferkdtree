/*
 * base.c
 *
 *  Created on: 31.10.2014
 *      Author: Fabian Gieseke
 */

#include "include/base.h"

/* --------------------------------------------------------------------------------
 * Interface (extern): Initialize all components
 * --------------------------------------------------------------------------------
 */
void init_extern(int n_neighbors, int tree_depth, int num_threads, int splitting_type,
		int verbosity_level, KD_TREE_PARAMETERS *params) {

	set_default_parameters(params);

	params->n_neighbors = n_neighbors;
	params->tree_depth = tree_depth;
	params->num_threads = num_threads;
	params->verbosity_level = verbosity_level;
	params->splitting_type = splitting_type;

	check_parameters(params);

	omp_set_num_threads(params->num_threads);

}

/* --------------------------------------------------------------------------------
 * Fits the nearest neighbor model (build kd-tree)
 * --------------------------------------------------------------------------------
 */
void fit_extern(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain,
		KD_TREE_RECORD *kdtree_record, KD_TREE_PARAMETERS *params) {

	kd_tree_init_tree_record(kdtree_record, params->tree_depth, Xtrain, nXtrain, dXtrain);
	kd_tree_generate_training_patterns_indices(kdtree_record);
	kd_tree_build_tree(kdtree_record, params);

}

/* --------------------------------------------------------------------------------
 * Computes nearest neighbors
 * --------------------------------------------------------------------------------
 */
void neighbors_extern(FLOAT_TYPE * Xtest, int nXtest, int dXtest,
		FLOAT_TYPE * distances, int ndistances, int ddistances,
		int *indices, int nindices, int dindices,
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

/* --------------------------------------------------------------------------------
 * Frees resources
 * --------------------------------------------------------------------------------
 */
void free_resources_extern(void) {

}
