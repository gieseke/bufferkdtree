/*
 * cpu.c
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
 */

#include "include/cpu.h"
#include "include/base.h"
#include "include/util.h"
#include "include/global.h"


/**
 * Intializes components if needed.
 *
 * @param *brute_record Pointer to record storing the model
 * @param *params Pointer to struct storing the parameters
 */
void init_cpu(BRUTE_RECORD *brute_record,
		BRUTE_PARAMETERS *params){

}

/**
 * Fits a model given the training data (and parameters)
 *
 * @param *Xtrain Training patterns
 * @param nXtrain Number of training patterns
 * @param dXtrain Dimensionality of training patterns
 * @param *brute_record Pointer to record storing the model
 * @param *params Pointer to struct storing the parameters
 */
void fit_cpu(FLOAT_TYPE *Xtrain,
		int nXtrain,
		int dXtrain,
		BRUTE_RECORD *brute_record,
		BRUTE_PARAMETERS *params) {

	brute_record->Xtrain = Xtrain;
	brute_record->nXtrain = nXtrain;
	brute_record->dXtrain = dXtrain;

}

/**
 * Does some clean up (before exiting the program).
 *
 * @param *brute_record Pointer to record storing the model
 * @param *params Pointer to struct storing the parameters
 */
void free_resources_cpu(BRUTE_RECORD *brute_record,
		BRUTE_PARAMETERS *params) {

	// nothing to do

}

/**
 * Computes the neighbors (for test patterns)
 *
 * @param *Xtest Pointer to the set of query/test points (stored as FLOAT_TYPE)
 * @param nXtest The number of query points
 * @param dXtest The dimension of each query point
 * @param *d_mins The distances array (FLOAT_TYPE) used to store the computed distances
 * @param *idx_mins Pointer to arrray storing the indices of the k nearest neighbors for each query point
 * @param *brute_record Pointer to struct storing all relevant information for model
 * @param *params Pointer to struct containing all relevant parameters
 *
 */
void neighbors_cpu(FLOAT_TYPE *Xtest,
		int nXtest,
		int dXtest,
		FLOAT_TYPE *d_mins,
		int *idx_mins,
		BRUTE_RECORD *brute_record,
		BRUTE_PARAMETERS *params) {

	int i;

	int K = params->n_neighbors;

	for (i = 0; i < nXtest * K; i++) {
		idx_mins[i] = 0;
		d_mins[i] = MAX_FLOAT_TYPE;
	}

	// number of threads (for parallel CPU version)
	omp_set_num_threads(params->num_threads);

	// parallel querying (one thread per test instance): by parallelizing over this loop,
	// the training patterns will be accessed by all threads (caching, share same elements)
#pragma omp parallel for
	for (i = 0; i < nXtest; i++) {
		compute_neighbors_single_instance_cpu(brute_record->Xtrain, \
				brute_record->nXtrain, brute_record->dXtrain, \
				Xtest + i * brute_record->dXtrain, d_mins + i * K, \
				idx_mins + i * K, K);
	}


}

/**
 * Computes nearest neighbors for a single test instance.
 *
 * @param *Xtrain Training patterns
 * @param nXtrain Number of training patterns
 * @param dim Dimensionality of training patterns
 * @param *test_pattern Test pattern
 * @param *d_min Pointer to array that shall be used to store the distances
 * @param *idx_min Pointer ot array that shall be used to store the indices
 * @param K The number of neighbors
 */
void compute_neighbors_single_instance_cpu(FLOAT_TYPE *Xtrain,
		int nXtrain,
		int dim,
		FLOAT_TYPE *test_pattern,
		FLOAT_TYPE *d_min,
		int *idx_min,
		int K) {

	int i;
	for (i = 0; i < nXtrain; i++) {
		FLOAT_TYPE d = squared_dist_cpu(Xtrain + i * dim, test_pattern, dim);
		insert_cpu(d, i, d_min, idx_min, K);
	}

}

/**
 * Computes the distance between point a and b in R^dim
 *
 * @param *a Pointer to first point
 * @param *b Pointer to second point
 * @param dim Dimensionality of the points
 */
inline FLOAT_TYPE squared_dist_cpu(FLOAT_TYPE *a,
		FLOAT_TYPE *b,
		int dim) {

	int j;
	FLOAT_TYPE d = 0.0;
	FLOAT_TYPE tmp;
	for (j = 0; j < dim; j++) {
		tmp = a[j] - b[j];
		d += tmp * tmp;
	}
	return d;

}

/**
 * Inserts the value pattern_dist with index pattern_idx in the list nearest_dist
 * of FLOAT_TYPEs and the list nearest_idx of indices. Both lists contain at most K
 * elements.
 *
 * @param pattern_dist Distance to the current pattern
 * @param pattern_idx Associated index
 * @param *nearest_dist Array of distances to be updated
 * @param *nearest_idx Arra of indices to be updated
 * @param K Number of nearest neighbors
 */
void insert_cpu(FLOAT_TYPE pattern_dist,
		int pattern_idx,
		FLOAT_TYPE *nearest_dist,
		int *nearest_idx,
		int K) {

	int j = K - 1;

	if (nearest_dist[j] <= pattern_dist)
		//rightmost is smaller
		return;

	nearest_dist[j] = pattern_dist;
	nearest_idx[j] = pattern_idx;

	FLOAT_TYPE tmp_d = 0.0;
	int tmp_idx = 0;

	// This is usually very fast since most candidates are not closer
	// (thus, we are exiting here directly, better than logarithmic search)
	for (; j > 0; j--) {
		if (nearest_dist[j] < nearest_dist[j - 1]) {
			//swap dist
			tmp_d = nearest_dist[j];
			nearest_dist[j] = nearest_dist[j - 1];
			nearest_dist[j - 1] = tmp_d;
			//swap idx
			tmp_idx = nearest_idx[j];
			nearest_idx[j] = nearest_idx[j - 1];
			nearest_idx[j - 1] = tmp_idx;
		} else
			break;
	}
}
