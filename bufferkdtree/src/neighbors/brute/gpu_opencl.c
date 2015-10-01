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
void init_gpu(void) {

	PRINT("Initializing OpenCL (platform_id=%i, device_id=%i)\n", params_global.platform_id, params_global.device_id);
	init_opencl(params_global.platform_id, &gpu_platform, params_global.device_id, &gpu_device,
			&gpu_context, &gpu_command_queue, params_global.verbosity_level);

}

/* -------------------------------------------------------------------------------- 
 * Fits a model given the training data (and parameters)
 * -------------------------------------------------------------------------------- 
 */
void fit_gpu(FLOAT_TYPE *X, int nX, int dX, Parameters *params) {

	cl_int err;

	PRINT("Compiling kernels ...\n");

	// kernel constants
	int K = params->n_neighbors;
	char constants[MAX_KERNEL_CONSTANTS_LENGTH];
	sprintf(constants,"#define USE_DOUBLE %d\n#define DIM %d\n#define K_NN %d\n", USE_DOUBLE, dX, K);

	// brute-force kernel
	char brute_kernel_fname[] = "brute.cl";
    char *kernel_final_path = malloc(strlen(params->kernels_source_directory) + strlen(brute_kernel_fname) + 1);
    strcpy(kernel_final_path, params->kernels_source_directory);
    strcat(kernel_final_path, brute_kernel_fname);

	gpu_brute_nearest_neighbors_kernel = make_kernel_from_file(gpu_context,
			gpu_device, constants, kernel_final_path, "nearest_neighbors");
	gpu_brute_transpose_kernel = make_kernel_from_file(gpu_context,
			gpu_device, constants, kernel_final_path, "transpose_simple");

	free(kernel_final_path);
	PRINT("Compilation done!\n");

	// initialize buffers for training patterns
	gpu_nXtrain = nX;
	gpu_dXtrain = dX;
	gpu_device_Xtrain = clCreateBuffer(gpu_context,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nX * dX * sizeof(FLOAT_TYPE), X, &err);
	check_cl_error(err, __FILE__, __LINE__);

}

/* --------------------------------------------------------------------------------
 * Does some clean up (before exiting the program).
 * --------------------------------------------------------------------------------
 */
void free_resources_gpu(void) {

	cl_int err;

	// release memory buffers
	err = clReleaseMemObject(gpu_device_Xtrain);
	check_cl_error(err, __FILE__, __LINE__);

	// free kernels
	err = clReleaseKernel(gpu_brute_nearest_neighbors_kernel);
	err |= clReleaseKernel(gpu_brute_transpose_kernel);
	check_cl_error(err, __FILE__, __LINE__);

	// release OpenCL
	PRINT("Freeing GPU resources ...\n");
	clReleaseCommandQueue(gpu_command_queue);
	clReleaseContext(gpu_context);
	//clReleaseDevice(gpu_device);


}

/* --------------------------------------------------------------------------------
 * Computes the predictions (for test patterns)
 * --------------------------------------------------------------------------------
 */
void neighbors_gpu(FLOAT_TYPE *Xtest, int nXtest, int dXtest, FLOAT_TYPE *d_mins, int *idx_mins, Parameters *params) {

	cl_int err;
	cl_event event;

	int K = params->n_neighbors;

	// buffer for test patterns
	cl_mem gpu_device_Xtest;

	// transposed array
	FLOAT_TYPE *Xtest_transposed = (FLOAT_TYPE*) malloc(
			nXtest * dXtest * sizeof(FLOAT_TYPE));
	TRANSPOSE_ARRAY(Xtest, nXtest, dXtest, Xtest_transposed);

	gpu_device_Xtest = clCreateBuffer(gpu_context,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			nXtest * dXtest * sizeof(FLOAT_TYPE), Xtest_transposed, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// free memory for transposed
	free(Xtest_transposed);

	// buffer for distances
	gpu_device_d_mins_trans = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE,
			nXtest * K * sizeof(FLOAT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// buffer for indices
	gpu_device_idx_mins_trans = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE,
			nXtest * K * sizeof(int), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// apply nearest neighbor kernel

	// global and local sizes
	size_t global_size_brute_nearest_neighbors = ceil(
			nXtest / (float) WORKGROUP_SIZE) * WORKGROUP_SIZE;
	size_t local_size_brute_nearest_neighbors = WORKGROUP_SIZE;

	// set kernel parameters
	err = clSetKernelArg(gpu_brute_nearest_neighbors_kernel, 0, sizeof(cl_mem),
			&gpu_device_Xtrain);
	err |= clSetKernelArg(gpu_brute_nearest_neighbors_kernel, 1, sizeof(cl_mem),
			&gpu_device_Xtest);
	err |= clSetKernelArg(gpu_brute_nearest_neighbors_kernel, 2, sizeof(cl_mem),
			&gpu_device_d_mins_trans);
	err |= clSetKernelArg(gpu_brute_nearest_neighbors_kernel, 3, sizeof(cl_mem),
			&gpu_device_idx_mins_trans);
	err |= clSetKernelArg(gpu_brute_nearest_neighbors_kernel, 4, sizeof(int),
			&gpu_nXtrain);
	err |= clSetKernelArg(gpu_brute_nearest_neighbors_kernel, 5, sizeof(int),
			&nXtest);
	check_cl_error(err, __FILE__, __LINE__);

	// execute kernel
	err = clEnqueueNDRangeKernel(gpu_command_queue,
			gpu_brute_nearest_neighbors_kernel, 1, NULL,
			&global_size_brute_nearest_neighbors,
			&local_size_brute_nearest_neighbors, 0, NULL, &event);
	err |= clWaitForEvents(1, &event);

	check_cl_error(err, __FILE__, __LINE__);
	clReleaseEvent(event);

	// copy results back to host system
	int *idx_mins_trans = (int*) malloc(nXtest *  K * sizeof(int));
	err = clEnqueueReadBuffer(gpu_command_queue, gpu_device_idx_mins_trans, CL_TRUE, 0,
			nXtest *  K * sizeof(int), idx_mins_trans, 0, NULL, &event);
    check_cl_error(err, __FILE__, __LINE__);
	err = clWaitForEvents(1, &event);
	check_cl_error(err, __FILE__, __LINE__);
	clReleaseEvent(event);


	FLOAT_TYPE *d_mins_trans = (FLOAT_TYPE*) malloc(nXtest *  K * sizeof(FLOAT_TYPE));
	err = clEnqueueReadBuffer(gpu_command_queue, gpu_device_d_mins_trans, CL_TRUE, 0,
			nXtest *  K * sizeof(FLOAT_TYPE), d_mins_trans, 0, NULL, &event);
    check_cl_error(err, __FILE__, __LINE__);
	err = clWaitForEvents(1, &event);
	check_cl_error(err, __FILE__, __LINE__);
	clReleaseEvent(event);

	// release buffers
	err = clReleaseMemObject(gpu_device_Xtest);
	err |= clReleaseMemObject(gpu_device_d_mins_trans);
	err |= clReleaseMemObject(gpu_device_idx_mins_trans);

	check_cl_error(err, __FILE__, __LINE__);

	// transpose arrays
	transpose_array_int(idx_mins_trans, K, nXtest, idx_mins);
	TRANSPOSE_ARRAY(d_mins_trans, K, nXtest, d_mins);

	free(idx_mins_trans);
	free(d_mins_trans);

}

