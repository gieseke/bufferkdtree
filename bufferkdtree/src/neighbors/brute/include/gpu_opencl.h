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

// kernel names
#define KERNEL_SOURCES_BRUTE STR(ABSOLUTE_PATH)"/kernels/opencl/brute.cl"

// default workgroup sizes
#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 256
#endif

// OpenCL
cl_platform_id gpu_platform;
cl_device_id gpu_device;
cl_context gpu_context;
cl_command_queue gpu_command_queue;

// OpenCL kernels
cl_kernel gpu_brute_nearest_neighbors_kernel;
cl_kernel gpu_brute_transpose_kernel;

// OpenCL buffers
int gpu_nXtrain, gpu_dXtrain;
cl_mem gpu_device_Xtrain;
cl_mem gpu_device_d_mins_trans;
cl_mem gpu_device_idx_mins_trans;

/* --------------------------------------------------------------------------------
 * Intializes components if needed.
 * --------------------------------------------------------------------------------
 */
void init_gpu(void);

/* -------------------------------------------------------------------------------- 
 * Fits a model given the training data (and parameters)
 * -------------------------------------------------------------------------------- 
 */
void fit_gpu(FLOAT_TYPE *X, int nX, int dX, Parameters *params);

/* --------------------------------------------------------------------------------
 * Does some clean up (before exiting the program).
 * --------------------------------------------------------------------------------
 */
void free_resources_gpu(void);

/* --------------------------------------------------------------------------------
 * Computes the predictions (for test patterns)
 * --------------------------------------------------------------------------------
 */
void neighbors_gpu(FLOAT_TYPE *Xtest, int nXtest, int dXtest, FLOAT_TYPE *d_mins,
		int *idx_mins, Parameters *params);

#endif /* BRUTE_INCLUDE_GPU_OPENCL_H_ */
