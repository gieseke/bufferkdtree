/* 
 * gpu_opencl.h
 */
#ifndef BRUTE_INCLUDE_GPU_OPENCL_H_
#define BRUTE_INCLUDE_GPU_OPENCL_H_

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
#include "../../../include/opencl.h"
#include "../../../include/util.h"
#include "../../../include/float.h"

// helper macros
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// default workgroup sizes
#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 256
#endif

/* --------------------------------------------------------------------------------
 * Intializes components if needed.
 * --------------------------------------------------------------------------------
 */
void init_gpu(BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Fits a model given the training data (and parameters)
 * -------------------------------------------------------------------------------- 
 */
void fit_gpu(FLOAT_TYPE *X, int nX, int dX, \
		BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Does some clean up (before exiting the program).
 * --------------------------------------------------------------------------------
 */
void free_resources_gpu(BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Computes the predictions (for test patterns)
 * --------------------------------------------------------------------------------
 */
void neighbors_gpu(FLOAT_TYPE *Xtest, int nXtest, int dXtest, FLOAT_TYPE *d_mins,
		int *idx_mins, BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params);

#endif /* BRUTE_INCLUDE_GPU_OPENCL_H_ */
