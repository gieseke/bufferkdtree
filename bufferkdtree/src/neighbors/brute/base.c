/* 
 * brute.c
 */
#include "include/base.h"
#include "include/util.h"
#include "include/cpu.h"
#include "include/gpu_opencl.h"
#include "include/global.h"

/* -------------------------------------------------------------------------------- 
 * Interface (extern): Initialize components
 * --------------------------------------------------------------------------------
 */
void init_extern(int n_neighbors, int num_threads, int platform_id, \
		int device_id, char *kernels_source_directory, int verbosity_level, \
		BRUTE_PARAMETERS *params) {

	set_default_parameters(params);

	params->n_neighbors = n_neighbors;
	params->num_threads = num_threads;
	params->platform_id = platform_id;
	params->device_id = device_id;
	params->kernels_source_directory = (char*) malloc((strlen(kernels_source_directory) + 10) * sizeof(char));
	strcpy(params->kernels_source_directory, kernels_source_directory);
	params->verbosity_level = verbosity_level;

	check_parameters(params);

}

/* --------------------------------------------------------------------------------
 * Interface (extern): fit model
 * -------------------------------------------------------------------------------- 
 */
void fit_extern(FLOAT_TYPE *X, int nX, int dX, BRUTE_RECORD *brute_record,
		BRUTE_PARAMETERS *params) {

	int i;
	for (i = 0; i < 25; i++) {
		INIT_MY_TIMER(brute_record->timers + i);
	}

	PRINT(params)("Fitting model ...\n");
	START_MY_TIMER(brute_record->timers + 1);

	brute_record->dXtrain = dX;
	brute_record->nXtrain = nX;

	INIT(brute_record, params);
	FIT(X, nX, dX, brute_record, params);

	STOP_MY_TIMER(brute_record->timers + 1);
	PRINT(params)("Fitting time (extern): \t\t\t\t\t\t\t\t%2.10f\n", \
			(FLOAT_TYPE) GET_MY_TIMER(brute_record->timers + 1));

}

/* -------------------------------------------------------------------------------- 
 * Interface (extern): compute k nearest neighbors
 * -------------------------------------------------------------------------------- 
 */
void neighbors_extern(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
		FLOAT_TYPE* distances, int ndistances, int ddistances,
		int* indices, int nindices, int dindices,
		BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params) {

	PRINT(params)("Computing nearest neighbors ...\n");
	START_MY_TIMER(brute_record->timers + 2);
	NEIGHBORS(Xtest, nXtest, dXtest, distances, indices, brute_record, params);
	STOP_MY_TIMER(brute_record->timers + 2);

	PRINT(params)("Total computation time (extern): \t\t\t\t\t\t%2.10f\n", \
			(FLOAT_TYPE) GET_MY_TIMER(brute_record->timers + 2));

}

/* --------------------------------------------------------------------------------
 * Frees some resources (e.g., on the GPU)
 * --------------------------------------------------------------------------------
 */
void free_resources_extern(BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params) {

	FREE_RESOURCES(brute_record, params);

	free(params->kernels_source_directory);

}

/* --------------------------------------------------------------------------------
 * Checks if platform and device are valid
 * --------------------------------------------------------------------------------
 */
int extern_check_platform_device(int platform_id, int device_id){

	return get_device_infos(platform_id, device_id, NULL);

}
