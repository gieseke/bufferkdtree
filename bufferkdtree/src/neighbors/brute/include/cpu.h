/*
 * cpu.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
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

/**
 * Intializes components if needed.
 * 
 *@param *brute_record Pointer to record storing the model
 *@param *params Pointer to struct storing the parameters
 */
void init_cpu(BRUTE_RECORD *brute_record, 
		BRUTE_PARAMETERS *params);

/**
 * Fits a model given the training data (and parameters)
 *
 *@param *Xtrain Training patterns
 *@param nXtrain Number of training patterns
 *@param dXtrain Dimensionality of training patterns
 *@param *brute_record Pointer to record storing the model
 *@param *params Pointer to struct storing the parameters
 */
void fit_cpu(FLOAT_TYPE *Xtrain,
		int nXtrain,
		int dXtrain,
		BRUTE_RECORD *brute_record,
		BRUTE_PARAMETERS *params);

/**
 * Does some clean up (before exiting the program).
 *
 *@param *brute_record Pointer to record storing the model
 *@param *params Pointer to struct storing the parameters
 */
void free_resources_cpu(BRUTE_RECORD *brute_record,
		BRUTE_PARAMETERS *params);

/**
 * Computes the neighbors (for test patterns)
 *
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
		BRUTE_PARAMETERS *params);

/**
 * Computes nearest neighbors for a single test instance.
 *
 *@param *Xtrain Training patterns
 *@param nXtrain Number of training patterns
 *@param dim Dimensionality of training patterns
 *@param *test_pattern Test pattern
 *@param *d_min Pointer to array that shall be used to store the distances
 *@param *idx_min Pointer ot array that shall be used to store the indices
 *@param K The number of neighbors
 */
void compute_neighbors_single_instance_cpu(FLOAT_TYPE *Xtrain,
		int nXtrain,
		int dim,
		FLOAT_TYPE *test_pattern,
		FLOAT_TYPE *d_min,
		int *idx_min,
		int K);

/** 
 * Computes the distance between point a and b in R^dim
 *
 *@param *a Pointer to first point
 *@param *b Pointer to second point
 *@param dim Dimensionality of the points
 */
inline FLOAT_TYPE squared_dist_cpu(FLOAT_TYPE *a, 
		FLOAT_TYPE *b, 
		int dim);
		
/**
 * Inserts the value pattern_dist with index pattern_idx in the list nearest_dist
 * of FLOAT_TYPEs and the list nearest_idx of indices. Both lists contain at most K
 * elements.
 *
 *@param pattern_dist Distance to the current pattern
 *@param pattern_idx Associated index
 *@param *nearest_dist Array of distances to be updated
 *@param *nearest_idx Arra of indices to be updated
 *@param K Number of nearest neighbors
 */
void insert_cpu(FLOAT_TYPE pattern_dist,
		int pattern_idx,
		FLOAT_TYPE *nearest_dist,
		int *nearest_idx,
		int K);

#endif /* BRUTE_INCLUDE_CPU_H_ */