/* 
 * global.h
 */
#ifndef BRUTE_INCLUDE_GLOBAL_H_
#define BRUTE_INCLUDE_GLOBAL_H_

#define INT_TYPE int
#define UINT_TYPE unsigned int

#include "../../../include/opencl.h"
#include "../../../include/timing.h"
#include "../../../include/float.h"

#ifndef USE_GPU
#define USE_GPU 0
#endif

// use simple float if not specified otherwise
#ifndef USE_DOUBLE
#define USE_DOUBLE 0
#endif

// floating point precision (single/double)
#if USE_DOUBLE > 0
#define FLOAT_TYPE double
#define MAX_FLOAT_TYPE     1.7976931348623158e+308
#define MIN_FLOAT_TYPE     -1.7976931348623158e+308
#define TRANSPOSE_ARRAY transpose_array_double
#else
#define FLOAT_TYPE float
#define MAX_FLOAT_TYPE     3.402823466e+38
#define MIN_FLOAT_TYPE     -3.402823466e+38
#define TRANSPOSE_ARRAY transpose_array_float
#endif

// CPU and GPU functions are mapped via the following macros
#if USE_GPU > 0
#define INIT init_gpu
#define FIT fit_gpu
#define NEIGHBORS neighbors_gpu
#define FREE_RESOURCES free_resources_gpu
#else
#define INIT init_cpu
#define FIT fit_cpu
#define NEIGHBORS neighbors_cpu
#define FREE_RESOURCES free_resources_cpu
#endif

// struct for input parameters
typedef struct brute_parameters {

	INT_TYPE n_neighbors;
	INT_TYPE num_threads;
	char *kernels_source_directory;
	INT_TYPE verbosity_level;
	INT_TYPE platform_id;
	INT_TYPE device_id;

} BRUTE_PARAMETERS;


// record struct for patterns and CL stuff
typedef struct brute_record {

	// training patterns
	FLOAT_TYPE *Xtrain;

	// dimension of patterns
	INT_TYPE dXtrain;

	// number of training patters
	INT_TYPE nXtrain;

	// test patterns
	FLOAT_TYPE *Xtest;

	// number of test patterns
	UINT_TYPE nXtest;

	TIMER timers[25];
	INT_TYPE counters[10];

	// OpenCL stuff
	cl_platform_id gpu_platform;
	cl_device_id gpu_device;
	cl_context gpu_context;
	cl_command_queue gpu_command_queue;

	// OpenCL kernels
	cl_kernel gpu_brute_nearest_neighbors_kernel;
	cl_kernel gpu_brute_transpose_kernel;

	// OpenCL buffers
	cl_mem gpu_device_Xtrain;
	cl_mem gpu_device_d_mins_trans;
	cl_mem gpu_device_idx_mins_trans;


} BRUTE_RECORD;

#endif /* BRUTE_INCLUDE_GLOBAL_H_ */
