/* 
 * brute.c
 */
#include "include/base.h"
#include "include/util.h"
#include "include/cpu.h"
#include "include/gpu_opencl.h"
#include "include/global.h"

// global parameters
Parameters params_global;

// timers
DEFINE_TIMER(1);
DEFINE_TIMER(2);
DEFINE_TIMER(3);

/* -------------------------------------------------------------------------------- 
 * Interface (extern): Initialize components
 * --------------------------------------------------------------------------------
 */
void init_extern(int n_neighbors, int num_threads, int platform_id, \
		int device_id, int verbosity_level) {

	START_TIMER(1);

	set_default_parameters(&params_global);
	params_global.n_neighbors = n_neighbors;
	params_global.num_threads = num_threads;
	params_global.platform_id = platform_id;
	params_global.device_id = device_id;
	params_global.verbosity_level = verbosity_level;
	check_parameters(&params_global);

	INIT();

	STOP_TIMER(1);

	PRINT("Initializing time (extern): \t\t%2.10f\n", (FLOAT_TYPE) GET_TIME(1));

}

void print_parameters_extern(void) {
	printf("=================================== Parameter Settings ===================================\n");
	printf("Number of nearest neighbors (K_NN): %i\n", params_global.n_neighbors);
	printf("Number of threads for CPU (num_threads): %i\n", params_global.num_threads);
	printf("Level of verbosity (verbosity_level): %i\n", params_global.verbosity_level);
	printf("Double precision? (USE_DOUBLE): %i\n", USE_DOUBLE);
	printf("Using GPU device (USE_GPU): %i\n", USE_GPU);
	printf("Floating point precision: %s\n", PLOT_PRECISION_INFOS());
	printf("==========================================================================================\n");
}

/* --------------------------------------------------------------------------------
 * Interface (extern): fit model
 * -------------------------------------------------------------------------------- 
 */
void fit_extern(FLOAT_TYPE *X, int nX, int dX) {

	PRINT("Fitting model ...\n");

	START_TIMER(2);
	FIT(X, nX, dX, &params_global);
	STOP_TIMER(2);

	PRINT("Fitting time (extern): \t\t\t\t\t\t\t\t%2.10f\n", (FLOAT_TYPE) GET_TIME(2));

}

/* -------------------------------------------------------------------------------- 
 * Interface (extern): compute k nearest neighbors
 * -------------------------------------------------------------------------------- 
 */
void neighbors_extern(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
		FLOAT_TYPE* distances, int ndistances, int ddistances,
		int* indices, int nindices, int dindices) {

	PRINT("Computing nearest neighbors ...\n");

	START_TIMER(3);
	NEIGHBORS(Xtest, nXtest, dXtest, distances, indices, &params_global);
	STOP_TIMER(3);

	PRINT("Total computation time (extern): \t\t\t\t\t\t%2.10f\n", (FLOAT_TYPE) GET_TIME(3));

}

/* --------------------------------------------------------------------------------
 * Frees some resources (e.g., on the GPU)
 * --------------------------------------------------------------------------------
 */
void free_resources_extern(void) {

	FREE_RESOURCES();

}

