/* 
 * global.h
 */
#ifndef BRUTE_INCLUDE_GLOBAL_H_
#define BRUTE_INCLUDE_GLOBAL_H_

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
#define PRINT_PATTERN print_pattern_double
#define SAVE_PREDICTIONS save_predictions_double
#define READ_INPUT_PATTERNS read_input_patterns_double
#define PLOT_PRECISION_INFOS plot_precision_infos_double
#else
#define FLOAT_TYPE float
#define MAX_FLOAT_TYPE     3.402823466e+38
#define MIN_FLOAT_TYPE     -3.402823466e+38
#define TRANSPOSE_ARRAY transpose_array_float
#define PRINT_PATTERN print_pattern_float
#define SAVE_PREDICTIONS save_predictions_float
#define READ_INPUT_PATTERNS read_input_patterns_float
#define PLOT_PRECISION_INFOS plot_precision_infos_float
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

// print function
#define PRINT if (params_global.verbosity_level > 0) printf

// struct for input parameters
struct parameters {
	int n_neighbors;
	int num_threads;
	int platform_id;
	char *kernels_source_directory;
	int device_id;
	int verbosity_level;
};

typedef struct parameters Parameters;

// parameters
extern Parameters params_global;

// declare timers
DECLARE_TIMER(1);
DECLARE_TIMER(2);
DECLARE_TIMER(3);

#endif /* BRUTE_INCLUDE_GLOBAL_H_ */
