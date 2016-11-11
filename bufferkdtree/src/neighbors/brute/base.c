/*
 * brute.c
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
 */
#include "include/base.h"
#include "include/util.h"
#include "include/cpu.h"
#include "include/gpu_opencl.h"
#include "include/global.h"

/**
 * Initializes the *params struct with the parameters provided.
 *
 * @param n_neighbors The number of nearest neighbors to be found
 * @param num_threads The number of threads that should be used
 * @param platform_id The OpenCL platform that should be used
 * @param device_id The id of the OpenCL device that should be used
 * @param *kernels_source_directory Pointer to string that contains the path to the OpenCL kernels
 * @param verbosity_level The verbosity level (0==no output, 1==more output, 2==...)
 * @param *params Pointer to struct that is used to store all parameters
 *
 */
void init_extern(int n_neighbors,
		int num_threads,
		int platform_id, \
		int device_id,
		char *kernels_source_directory,
		int verbosity_level, \
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

/**
 * Interface (extern): Fits the model (e.g., intializes OPenCL device etc.)
 *
 * @param *X Pointer to array of type "FLOAT_TYPE" (either "float" or "double")
 * @param nX Number of rows in *X (i.e., points/patterns)
 * @param dX Number of columns in *X (one column per point/pattern)
 * @param *brute_record Pointer to struct instance storing the model
 * @param *params Pointer to struct instance storing all model parameters
 *
 */
void fit_extern(FLOAT_TYPE *X,
		int nX,
		int dX,
		BRUTE_RECORD *brute_record,
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

/**
 * Interface (extern): Computes the k nearest neighbors for a given set of test points
 * stored in *Xtest and stores the results in two arrays *distances and *indices.
 *
 * @param *Xtest Pointer to the set of query/test points (stored as FLOAT_TYPE)
 * @param nXtest The number of query points
 * @param dXtest The dimension of each query point
 * @param *distances The distances array (FLOAT_TYPE) used to store the computed distances
 * @param ndistances The number of query points
 * @param ddistances The number of distance values for each query point
 * @param *indices Pointer to arrray storing the indices of the k nearest neighbors for each query point
 * @param nindices The number of query points
 * @param dindices The number of indices comptued for each query point
 * @param *brute_record Pointer to struct storing all relevant information for model
 * @param *params Pointer to struct containing all relevant parameters
 *
 */
void neighbors_extern(FLOAT_TYPE *Xtest,
		int nXtest,
		int dXtest,
		FLOAT_TYPE* distances,
		int ndistances,
		int ddistances,
		int* indices,
		int nindices,
		int dindices,
		BRUTE_RECORD *brute_record,
		BRUTE_PARAMETERS *params) {

	PRINT(params)("Computing nearest neighbors ...\n");
	START_MY_TIMER(brute_record->timers + 2);
	NEIGHBORS(Xtest, nXtest, dXtest, distances, indices, brute_record, params);
	STOP_MY_TIMER(brute_record->timers + 2);

	PRINT(params)("Total computation time (extern): \t\t\t\t\t\t%2.10f\n", \
			(FLOAT_TYPE) GET_MY_TIMER(brute_record->timers + 2));

}

/**
 * Frees resources (e.g., on the GPU)
 *
 * @param *brute_record Pointer to struct storing all relevant information for model
 * @param *params Pointer to struct containing all relevant parameters
 *
 */
void free_resources_extern(BRUTE_RECORD *brute_record,
		BRUTE_PARAMETERS *params) {

	FREE_RESOURCES(brute_record, params);

	free(params->kernels_source_directory);

}

/**
 * Checks if platform and device are valid
 *
 * @param platform_id The OpenCL platform id to be used
 * @param device_id The OpenCL device id to be used
 *
 */
int extern_check_platform_device(int platform_id,
		int device_id){

	return get_device_infos(platform_id, device_id, NULL);

}
