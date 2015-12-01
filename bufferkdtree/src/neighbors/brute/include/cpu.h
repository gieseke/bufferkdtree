/* 
 * cpu.h
 */

#ifndef BRUTE_INCLUDE_CPU_H_
#define BRUTE_INCLUDE_CPU_H_

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
#include <sched.h>
#include <omp.h>

#include "util.h"
#include "global.h"

/* -------------------------------------------------------------------------------- 
 * Intializes components if needed.
 * --------------------------------------------------------------------------------
 */
void init_cpu(BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Fits a model given the training data (and parameters)
 * -------------------------------------------------------------------------------- 
 */
void fit_cpu(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain, \
		BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Does some clean up (before exiting the program).
 * --------------------------------------------------------------------------------
 */
void free_resources_cpu(BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Computes the neighbors (for test patterns)
 * -------------------------------------------------------------------------------- 
 */
void neighbors_cpu(FLOAT_TYPE *Xtest, int nXtest, int dXtest, \
		FLOAT_TYPE *d_mins, int *idx_mins, \
		BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Computes nearest neighbors for a single test instance.
 * -------------------------------------------------------------------------------- 
 */
void compute_neighbors_single_instance_cpu(FLOAT_TYPE *Xtrain, int nXtrain, \
		int dim, FLOAT_TYPE *test_pattern, FLOAT_TYPE *d_min, \
		int *idx_min, int K);

/* -------------------------------------------------------------------------------- 
 * Inserts the value pattern_dist with index pattern_idx in the list nearest_dist
 * of FLOAT_TYPEs and the list nearest_idx of indices. Both lists contain at most K
 * elements.
 * -------------------------------------------------------------------------------- 
 */
void insert_cpu(FLOAT_TYPE pattern_dist, int pattern_idx,
		FLOAT_TYPE *nearest_dist, int *nearest_idx, int K);

/* -------------------------------------------------------------------------------- 
 * Computes the distance between point a and b in R^dim
 * -------------------------------------------------------------------------------- 
 */
inline FLOAT_TYPE squared_dist_cpu(FLOAT_TYPE *a, FLOAT_TYPE *b, int dim);

#endif /* BRUTE_INCLUDE_CPU_H_ */
