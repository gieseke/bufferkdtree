/* 
 * gpu_opencl.c
 */

#include "include/gpu_opencl.h"
#include "include/base.h"
#include "include/util.h"
#include "include/global.h"

/* --------------------------------------------------------------------------------
 * Initializes components if needed.
 * --------------------------------------------------------------------------------
 */
void init_gpu(BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params) {

	PRINT(params)("Initializing OpenCL (platform_id=%i, device_id=%i)\n", \
			params->platform_id, params->device_id);
	init_opencl(params->platform_id, &(brute_record->gpu_platform), \
			params->device_id, &(brute_record->gpu_device), \
			&(brute_record->gpu_context), \
			&(brute_record->gpu_command_queue), \
			params->verbosity_level);

}

/* --------------------------------------------------------------------------------
 * Fits a model given the training data (and parameters)
 * --------------------------------------------------------------------------------
 */
void fit_gpu(FLOAT_TYPE *X, int nX, int dX, BRUTE_RECORD *brute_record, \
		BRUTE_PARAMETERS *params) {

	cl_int err;

	PRINT(params)("Compiling kernels ...\n");

	// kernel constants
	int K = params->n_neighbors;
	char constants[MAX_KERNEL_CONSTANTS_LENGTH];
	sprintf(constants,"#define USE_DOUBLE %d\n#define DIM %d\n#define K_NN %d\n", USE_DOUBLE, dX, K);

	// brute-force kernel
	char brute_kernel_fname[] = "brute.cl";
    char *kernel_final_path = malloc(strlen(params->kernels_source_directory) + strlen(brute_kernel_fname) + 1);
    strcpy(kernel_final_path, params->kernels_source_directory);
    strcat(kernel_final_path, brute_kernel_fname);

	brute_record->gpu_brute_nearest_neighbors_kernel = make_kernel_from_file(brute_record->gpu_context,
			brute_record->gpu_device, constants, kernel_final_path, "nearest_neighbors");
	brute_record->gpu_brute_transpose_kernel = make_kernel_from_file(brute_record->gpu_context,
			brute_record->gpu_device, constants, kernel_final_path, "transpose_simple");

	free(kernel_final_path);
	PRINT(params)("Compilation done!\n");

	// initialize buffers for training patterns
	brute_record->gpu_device_Xtrain = clCreateBuffer(brute_record->gpu_context, \
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nX * dX * sizeof(FLOAT_TYPE), X, &err);
	check_cl_error(err, __FILE__, __LINE__);

}

/* --------------------------------------------------------------------------------
 * Does some clean up (before exiting the program).
 * --------------------------------------------------------------------------------
 */
void free_resources_gpu(BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params) {

	cl_int err;

	// release memory buffers
	err = clReleaseMemObject(brute_record->gpu_device_Xtrain);
	check_cl_error(err, __FILE__, __LINE__);

	// free kernels
	err = clReleaseKernel(brute_record->gpu_brute_nearest_neighbors_kernel);
	err |= clReleaseKernel(brute_record->gpu_brute_transpose_kernel);
	check_cl_error(err, __FILE__, __LINE__);

	// release OpenCL
	PRINT(params)("Freeing GPU resources ...\n");
	clReleaseCommandQueue(brute_record->gpu_command_queue);
	clReleaseContext(brute_record->gpu_context);
	//clReleaseDevice(gpu_device);


}

/* --------------------------------------------------------------------------------
 * Computes the predictions (for test patterns)
 * --------------------------------------------------------------------------------
 */
void neighbors_gpu(FLOAT_TYPE *Xtest, int nXtest, int dXtest, FLOAT_TYPE *d_mins, \
		int *idx_mins, BRUTE_RECORD *brute_record, BRUTE_PARAMETERS *params) {

	cl_int err;
	cl_event event;

	int K = params->n_neighbors;

	// transposed array
	FLOAT_TYPE *Xtest_transposed = (FLOAT_TYPE*) malloc(nXtest * dXtest * sizeof(FLOAT_TYPE));
	TRANSPOSE_ARRAY(Xtest, nXtest, dXtest, Xtest_transposed);

	cl_mem gpu_device_Xtest = clCreateBuffer(brute_record->gpu_context, \
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, \
			nXtest * dXtest * sizeof(FLOAT_TYPE), Xtest_transposed, &err);
	check_cl_error(err, __FILE__, __LINE__);
	free(Xtest_transposed);

	// buffer for distances
	brute_record->gpu_device_d_mins_trans = clCreateBuffer(brute_record->gpu_context, CL_MEM_READ_WRITE,
			nXtest * K * sizeof(FLOAT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// buffer for indices
	brute_record->gpu_device_idx_mins_trans = clCreateBuffer(brute_record->gpu_context, CL_MEM_READ_WRITE,
			nXtest * K * sizeof(int), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// apply nearest neighbor kernel

	// set kernel parameters
	err = clSetKernelArg(brute_record->gpu_brute_nearest_neighbors_kernel, 0, sizeof(cl_mem),
			&(brute_record->gpu_device_Xtrain));
	err |= clSetKernelArg(brute_record->gpu_brute_nearest_neighbors_kernel, 1, sizeof(cl_mem),
			&(gpu_device_Xtest));
	err |= clSetKernelArg(brute_record->gpu_brute_nearest_neighbors_kernel, 2, sizeof(cl_mem),
			&(brute_record->gpu_device_d_mins_trans));
	err |= clSetKernelArg(brute_record->gpu_brute_nearest_neighbors_kernel, 3, sizeof(cl_mem),
			&(brute_record->gpu_device_idx_mins_trans));
	err |= clSetKernelArg(brute_record->gpu_brute_nearest_neighbors_kernel, 4, sizeof(int),
			&(brute_record->nXtrain));
	err |= clSetKernelArg(brute_record->gpu_brute_nearest_neighbors_kernel, 5, sizeof(int),
			&nXtest);
	check_cl_error(err, __FILE__, __LINE__);

	int max_chunk_size = 65536;
	if (max_chunk_size > nXtest){
		max_chunk_size = nXtest;
	}

	// global and local sizes
	size_t local_size_brute_nearest_neighbors = WORKGROUP_SIZE;

	int chunk_start = 0;
	while (chunk_start < nXtest) {
		int chunk_end = chunk_start + max_chunk_size;
		if (chunk_end > nXtest) {
			chunk_end = nXtest;
		}
		int n_indices = chunk_end - chunk_start;

		size_t global_size_brute_nearest_neighbors = ceil(n_indices / (float) WORKGROUP_SIZE) * WORKGROUP_SIZE;

		err = clSetKernelArg(brute_record->gpu_brute_nearest_neighbors_kernel, 6, sizeof(int), &chunk_start);
		check_cl_error(err, __FILE__, __LINE__);

		// execute kernel
		err = clEnqueueNDRangeKernel(brute_record->gpu_command_queue,
				brute_record->gpu_brute_nearest_neighbors_kernel, 1, NULL,
				&global_size_brute_nearest_neighbors,
				&local_size_brute_nearest_neighbors, 0, NULL, &event);
		check_cl_error(err, __FILE__, __LINE__);

		err = clWaitForEvents(1, &event);
		check_cl_error(err, __FILE__, __LINE__);
		clReleaseEvent(event);

		chunk_start += max_chunk_size;

	}

	// copy results back to host system
	int *idx_mins_trans = (int*) malloc(nXtest *  K * sizeof(int));
	err = clEnqueueReadBuffer(brute_record->gpu_command_queue, brute_record->gpu_device_idx_mins_trans, CL_TRUE, 0,
			nXtest *  K * sizeof(int), idx_mins_trans, 0, NULL, &event);
    check_cl_error(err, __FILE__, __LINE__);
	err = clWaitForEvents(1, &event);
	check_cl_error(err, __FILE__, __LINE__);
	clReleaseEvent(event);

	FLOAT_TYPE *d_mins_trans = (FLOAT_TYPE*) malloc(nXtest *  K * sizeof(FLOAT_TYPE));
	err = clEnqueueReadBuffer(brute_record->gpu_command_queue, brute_record->gpu_device_d_mins_trans, CL_TRUE, 0,
			nXtest *  K * sizeof(FLOAT_TYPE), d_mins_trans, 0, NULL, &event);
    check_cl_error(err, __FILE__, __LINE__);
	err = clWaitForEvents(1, &event);
	check_cl_error(err, __FILE__, __LINE__);
	clReleaseEvent(event);

	// release buffers
	err = clReleaseMemObject(gpu_device_Xtest);
	err |= clReleaseMemObject(brute_record->gpu_device_d_mins_trans);
	err |= clReleaseMemObject(brute_record->gpu_device_idx_mins_trans);
	check_cl_error(err, __FILE__, __LINE__);

	// transpose arrays
	transpose_array_int(idx_mins_trans, K, nXtest, idx_mins);
	TRANSPOSE_ARRAY(d_mins_trans, K, nXtest, d_mins);

	free(idx_mins_trans);
	free(d_mins_trans);

}

