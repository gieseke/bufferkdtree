/* 
 * brute.h
 */
#ifndef BRUTE_INCLUDE_BASE_H_
#define BRUTE_INCLUDE_BASE_H_

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

#include "../../../include/timing.h"

/* -------------------------------------------------------------------------------- 
 * Interface (extern): Initialize components
 * --------------------------------------------------------------------------------
 */
void init_extern(int n_neighbors, int num_threads, int platform_id, \
		int device_id, int verbosity_level);

/* --------------------------------------------------------------------------------
 * Interface (extern): fit model
 * -------------------------------------------------------------------------------- 
 */
void fit_extern(FLOAT_TYPE *X, int nX, int dX);

/* -------------------------------------------------------------------------------- 
 * Interface (extern): compute k nearest neighbors
 * -------------------------------------------------------------------------------- 
 */
void neighbors_extern(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
		FLOAT_TYPE* distances, int ndistances, int ddistances,
		int* indices, int nindices, int dindices);

/* --------------------------------------------------------------------------------
 * Frees some resources (e.g., on the GPU)
 * --------------------------------------------------------------------------------
 */
void free_resources_extern(void);

/* --------------------------------------------------------------------------------
 * Prints parameters that are used.
 * --------------------------------------------------------------------------------
 */
void print_parameters_extern(void);

#endif /* BRUTE_INCLUDE_BASE_H_ */
