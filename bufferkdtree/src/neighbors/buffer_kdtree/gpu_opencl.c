/* 
 * gpu.c
 */

#include "include/gpu_opencl.h"

/* --------------------------------------------------------------------------------
 * Initializes all devices at the beginning of the  querying process.
 * --------------------------------------------------------------------------------
 */
void init_opencl_devices(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	cl_int err;
	cl_event event_copy_nodes, event_copy_leaves;

	tree_record->counters[0] = 0;
	tree_record->counters[1] = 0;

	PRINT(params)("-----------------------------------------------------\n");
	PRINT(params)("Initializing OpenCL (platform_id=%i, device_id=%i)\n", params->platform_id, params->device_id);
	init_opencl(params->platform_id, &(tree_record->gpu_platform), params->device_id, &(tree_record->gpu_device),
			&(tree_record->gpu_context), &(tree_record->gpu_command_queue), params->verbosity_level);
	init_command_queue(&(tree_record->gpu_command_queue_chunk_0), &(tree_record->gpu_device), &(tree_record->gpu_context));
	init_command_queue(&(tree_record->gpu_command_queue_chunk_1), &(tree_record->gpu_device), &(tree_record->gpu_context));
	PRINT(params)("-----------------------------------------------------\n");

	// nodes
	tree_record->device_nodes = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_ONLY,
			tree_record->n_nodes * sizeof(TREE_NODE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);
	err = clEnqueueWriteBuffer(tree_record->gpu_command_queue, tree_record->device_nodes, CL_FALSE, 0,
			tree_record->n_nodes * sizeof(TREE_NODE), tree_record->nodes, 0, NULL, &event_copy_nodes);
	check_cl_error(err, __FILE__, __LINE__);

	// leaves
	tree_record->device_leave_bounds = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_ONLY,
			tree_record->n_leaves * 2 * sizeof(FLOAT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);
	err = clEnqueueWriteBuffer(tree_record->gpu_command_queue, tree_record->device_leave_bounds, CL_FALSE, 0,
			tree_record->n_leaves * 2 * sizeof(FLOAT_TYPE), tree_record->leaves, 0, NULL, &event_copy_leaves);
	check_cl_error(err, __FILE__, __LINE__);

	// constants for all kernels
	char constants[MAX_KERNEL_CONSTANTS_LENGTH];
	sprintf(constants,
			"#define DIM %d\n\
			#define NUM_NEIGHBORS %d\n\
			#define K_NN %d\n\
			#define MAX_VISITED %i\n\
			#define TREE_DEPTH %i\n\
			#define NUM_NODES %i\n\
			#define USE_DOUBLE %d\n",
			tree_record->dXtrain, params->n_neighbors, params->n_neighbors, tree_record->max_visited,
			params->tree_depth, tree_record->n_nodes, USE_DOUBLE);

    char kernel_final_path [4096];

	// compile kernels
    strcpy(kernel_final_path, params->kernels_source_directory);
    strcat(kernel_final_path, "find_leaves_idx_batch_float.cl");
	tree_record->find_leaves_kernel = make_kernel_from_file((tree_record->gpu_context),
			tree_record->gpu_device, constants, kernel_final_path, "find_leaf_idx_batch");

    strcpy(kernel_final_path, params->kernels_source_directory);
    strcat(kernel_final_path, "brute_all_leaves_nearest_neighbors.cl");
	tree_record->brute_nn_kernel = make_kernel_from_file((tree_record->gpu_context),
			tree_record->gpu_device, constants, kernel_final_path,
			"do_bruteforce_all_leaves_nearest_neighbors");

    strcpy(kernel_final_path, params->kernels_source_directory);
    strcat(kernel_final_path, "update_distances.cl");
	tree_record->update_dist_kernel = make_kernel_from_file((tree_record->gpu_context),
			tree_record->gpu_device, constants, kernel_final_path,
			"do_update_distances");
	tree_record->retrieve_dist_kernel = make_kernel_from_file((tree_record->gpu_context),
			tree_record->gpu_device, constants, kernel_final_path,
			"do_retrieve_distances");
	tree_record->compute_final_dists_idxs_kernel = make_kernel_from_file((tree_record->gpu_context),
			tree_record->gpu_device, constants, kernel_final_path,
			"do_compute_final_distances_indices");

    strcpy(kernel_final_path, params->kernels_source_directory);
    strcat(kernel_final_path, "generate_subset_test_patterns.cl");
	tree_record->generate_test_subset_kernel = make_kernel_from_file((tree_record->gpu_context),
			tree_record->gpu_device, constants, kernel_final_path,
			"do_generate_subset_test_patterns");

    strcpy(kernel_final_path, params->kernels_source_directory);
    strcat(kernel_final_path, "init_arrays.cl");
	tree_record->init_dists_kernel = make_kernel_from_file((tree_record->gpu_context),
			tree_record->gpu_device, constants, kernel_final_path, "do_init_distances");
	tree_record->init_stacks_kernel = make_kernel_from_file((tree_record->gpu_context),
			tree_record->gpu_device, constants, kernel_final_path, "do_init_allstacks");
	tree_record->init_depths_idxs_kernel = make_kernel_from_file((tree_record->gpu_context),
			tree_record->gpu_device, constants, kernel_final_path, "do_init_depths_idx");

	err = clWaitForEvents(1, &event_copy_nodes);
	err |= clReleaseEvent(event_copy_nodes);
	check_cl_error(err, __FILE__, __LINE__);

	err = clWaitForEvents(1, &event_copy_leaves);
	err |= clReleaseEvent(event_copy_leaves);
	check_cl_error(err, __FILE__, __LINE__);

	tree_record->device_query_buffers_allocated = 0;

	PRINT(params)("GPU initialized successfully!\n");

}

/* -------------------------------------------------------------------------------- 
 * After having performed all queries: Free memory etc.
 * --------------------------------------------------------------------------------
 */
void free_opencl_devices(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	cl_int err;

	err = clReleaseMemObject(tree_record->device_nodes);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseMemObject(tree_record->device_leave_bounds);
	check_cl_error(err, __FILE__, __LINE__);

	//free_train_buffers_gpu(tree_record, params);
	//free_query_buffers_gpu(tree_record, params);

	err = clReleaseKernel(tree_record->find_leaves_kernel);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseKernel(tree_record->update_dist_kernel);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseKernel(tree_record->retrieve_dist_kernel);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseKernel(tree_record->brute_nn_kernel);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseKernel(tree_record->generate_test_subset_kernel);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseKernel(tree_record->compute_final_dists_idxs_kernel);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseKernel(tree_record->init_dists_kernel);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseKernel(tree_record->init_stacks_kernel);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseKernel(tree_record->init_depths_idxs_kernel);
	check_cl_error(err, __FILE__, __LINE__);

	err = clReleaseCommandQueue(tree_record->gpu_command_queue);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseCommandQueue(tree_record->gpu_command_queue_chunk_0);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseCommandQueue(tree_record->gpu_command_queue_chunk_1);
	check_cl_error(err, __FILE__, __LINE__);
	err = clReleaseContext(tree_record->gpu_context);
	check_cl_error(err, __FILE__, __LINE__);

	// check: http://stackoverflow.com/questions/15855759/opencl-1-2-c-wrapper-undefined-reference-to-clreleasedevice
	err = clReleaseDevice(tree_record->gpu_device);
	check_cl_error(err, __FILE__, __LINE__);

}

/* --------------------------------------------------------------------------------
 * Free buffers used during training.
 * --------------------------------------------------------------------------------
 */
void free_train_buffers_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	free_train_patterns_device(tree_record, params, TRAIN_CHUNK_0);
	if (!training_chunks_inactive(tree_record, params)) {
		free_train_patterns_device(tree_record, params, TRAIN_CHUNK_1);
	}
}

/* --------------------------------------------------------------------------------
 * Free buffers needed for querying phase.
 * --------------------------------------------------------------------------------
 */
void free_query_buffers_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	cl_int err;

	if (tree_record->device_query_buffers_allocated == 1){

		err = clReleaseMemObject(tree_record->device_test_patterns);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_d_mins);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_idx_mins);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_all_stacks);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_all_depths);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_all_idxs);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_dist_mins_tmp);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_idx_mins_tmp);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_test_patterns_subset_tmp);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_fr_indices);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_to_indices);
		check_cl_error(err, __FILE__, __LINE__);
		err = clReleaseMemObject(tree_record->device_test_indices_removed_from_all_buffers);
		check_cl_error(err, __FILE__, __LINE__);

		tree_record->device_query_buffers_allocated = 0;
	} else {
		printf("no buffers allocated ...?\n");
	}

}

/* --------------------------------------------------------------------------------
 * Allocates memory for testing phase.
 * -------------------------------------------------------------------------------- 
 */
void allocate_memory_opencl_devices(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	cl_int err;
	cl_event event_copy_test, event_distance_kernel;
	cl_event event_init_all_stacks, event_all_depths_idxs;

	// if the training data fits on the device: copy it once at the beginning!
	if (training_chunks_inactive(tree_record, params)) {
		init_train_patterns_buffers(tree_record, params, TRAIN_CHUNK_0, tree_record->nXtrain);
		copy_train_patterns_to_device(tree_record, params, TRAIN_CHUNK_0, 0, tree_record->nXtrain);
		// otherwise: initialize buffers for chunks
	} else {
		tree_record->n_patts_per_chunk = (INT_TYPE) floor(((FLOAT_TYPE) tree_record->nXtrain) / params->n_train_chunks) + 1;
		// first chunk is copied here; afterwards, this chunk is always copied after the final
		// chunk (i.e., before processing all chunks, chunk 0 is already copied!)
		init_train_patterns_buffers(tree_record, params, TRAIN_CHUNK_0, tree_record->n_patts_per_chunk);
		copy_train_patterns_to_device(tree_record, params, TRAIN_CHUNK_0, 0, tree_record->n_patts_per_chunk);
		init_train_patterns_buffers(tree_record, params, TRAIN_CHUNK_1, tree_record->n_patts_per_chunk);
	}
	tree_record->current_chunk_id = TRAIN_CHUNK_0;

	// test_patterns
	tree_record->device_test_patterns = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_ONLY,
			tree_record->nXtest * tree_record->dXtrain * sizeof(FLOAT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);
	err = clEnqueueWriteBuffer(tree_record->gpu_command_queue, tree_record->device_test_patterns, CL_FALSE, 0,
			tree_record->nXtest * tree_record->dXtrain * sizeof(FLOAT_TYPE), tree_record->Xtest, 0, NULL,
			&event_copy_test);
	check_cl_error(err, __FILE__, __LINE__);

	// all_stacks
	tree_record->device_all_stacks = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_WRITE,
			tree_record->nXtest * params->tree_depth * sizeof(INT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// all_depths
	tree_record->device_all_depths = clCreateBuffer(tree_record->gpu_context,
	CL_MEM_READ_WRITE, tree_record->nXtest * sizeof(INT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// all_idxs
	tree_record->device_all_idxs = clCreateBuffer(tree_record->gpu_context,
	CL_MEM_READ_WRITE, tree_record->nXtest * sizeof(INT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// dist_mins_global
	tree_record->device_d_mins = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_WRITE,
			tree_record->nXtest * params->n_neighbors * sizeof(FLOAT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// idx_mins_global
	tree_record->device_idx_mins = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_WRITE,
			tree_record->nXtest * params->n_neighbors * sizeof(INT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	INT_TYPE num_elts = tree_record->nXtest * params->n_neighbors;

	size_t global_size_1[] = { WORKGROUP_SIZE_COPY_INIT * ((INT_TYPE) num_elts / WORKGROUP_SIZE_COPY_INIT)
			+ WORKGROUP_SIZE_COPY_INIT };
	size_t local_size_1[] = { WORKGROUP_SIZE_COPY_INIT };

	// execute init_distances_kernel
	err = clSetKernelArg(tree_record->init_dists_kernel, 0, sizeof(INT_TYPE), &num_elts);
	err |= clSetKernelArg(tree_record->init_dists_kernel, 1, sizeof(cl_mem), &(tree_record->device_d_mins));
	err |= clSetKernelArg(tree_record->init_dists_kernel, 2, sizeof(cl_mem), &(tree_record->device_idx_mins));
	err |= clEnqueueNDRangeKernel(tree_record->gpu_command_queue, tree_record->init_dists_kernel, 1,
			NULL, global_size_1, local_size_1, 0, NULL, &event_distance_kernel);
	check_cl_error(err, __FILE__, __LINE__);

	// allstacks
	num_elts = tree_record->nXtest * params->tree_depth;
	size_t global_size_2[] = { WORKGROUP_SIZE_COPY_INIT * ((INT_TYPE) num_elts / WORKGROUP_SIZE_COPY_INIT)
			+ WORKGROUP_SIZE_COPY_INIT };
	size_t local_size_2[] = { WORKGROUP_SIZE_COPY_INIT };
	err = clSetKernelArg(tree_record->init_stacks_kernel, 0, sizeof(INT_TYPE), &num_elts);
	err |= clSetKernelArg(tree_record->init_stacks_kernel, 1, sizeof(cl_mem), &(tree_record->device_all_stacks));
	err |= clEnqueueNDRangeKernel(tree_record->gpu_command_queue, tree_record->init_stacks_kernel, 1,
			NULL, global_size_2, local_size_2, 0, NULL, &event_init_all_stacks);
	check_cl_error(err, __FILE__, __LINE__);

	// all_depths and all_idxs
	num_elts = tree_record->nXtest;
	size_t global_size_3[] = { WORKGROUP_SIZE_COPY_INIT * ((INT_TYPE) num_elts / WORKGROUP_SIZE_COPY_INIT) + WORKGROUP_SIZE_COPY_INIT };
	size_t local_size_3[] = { WORKGROUP_SIZE_COPY_INIT };
	err = clSetKernelArg(tree_record->init_depths_idxs_kernel, 0, sizeof(INT_TYPE), &num_elts);
	err |= clSetKernelArg(tree_record->init_depths_idxs_kernel, 1, sizeof(cl_mem), &(tree_record->device_all_depths));
	err |= clSetKernelArg(tree_record->init_depths_idxs_kernel, 2, sizeof(cl_mem), &(tree_record->device_all_idxs));
	err |= clEnqueueNDRangeKernel(tree_record->gpu_command_queue, tree_record->init_depths_idxs_kernel, 1, NULL, global_size_3, local_size_3, 0, NULL, &event_all_depths_idxs);
	check_cl_error(err, __FILE__, __LINE__);

	// we allocate host space for consecutive buffers to store intermediate results
	// (we can have at most number_test_patterns elements in each round)
	tree_record->device_dist_mins_tmp = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_WRITE, tree_record->nXtest * params->n_neighbors * sizeof(FLOAT_TYPE),NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);
	tree_record->device_idx_mins_tmp = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_WRITE, tree_record->nXtest * params->n_neighbors * sizeof(INT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// we also allocate additional memory for the test patterns on the GPU device to
	// store intermediate but consecutive subsets of the patterns in each iteration
	tree_record->device_test_patterns_subset_tmp = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_WRITE, tree_record->nXtest * tree_record->dXtrain * sizeof(FLOAT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	err = clWaitForEvents(1, &event_copy_test);
	err |= clReleaseEvent(event_copy_test);
	check_cl_error(err, __FILE__, __LINE__);

	err = clWaitForEvents(1, &event_distance_kernel);
	err |= clReleaseEvent(event_distance_kernel);
	check_cl_error(err, __FILE__, __LINE__);

	err = clWaitForEvents(1, &event_init_all_stacks);
	err |= clReleaseEvent(event_init_all_stacks);
	check_cl_error(err, __FILE__, __LINE__);

	err = clWaitForEvents(1, &event_all_depths_idxs);
	err |= clReleaseEvent(event_all_depths_idxs);
	check_cl_error(err, __FILE__, __LINE__);

	// fr_indices
	tree_record->device_fr_indices = clCreateBuffer(tree_record->gpu_context,
			CL_MEM_READ_ONLY, tree_record->nXtest * sizeof(INT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// to_indices
	tree_record->device_to_indices = clCreateBuffer(tree_record->gpu_context,
			CL_MEM_READ_ONLY, tree_record->nXtest * sizeof(INT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	tree_record->device_test_indices_removed_from_all_buffers = clCreateBuffer(tree_record->gpu_context,
			CL_MEM_READ_ONLY,  tree_record->nXtest * sizeof(INT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	tree_record->device_query_buffers_allocated = 1;

	PRINT(params)("GPU memory allocated successfully!\n");

}

/* -------------------------------------------------------------------------------- 
 * Processes all buffers on the GPU. It is important that only INDICES are moved
 * from the CPU to the GPU (only CPU->GPU; GPU->CPU is not necessary). Further, the
 * global distances and the indices are updated ON THE GPU.
 * -------------------------------------------------------------------------------- 
 */
void process_all_buffers_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	START_MY_TIMER(tree_record->timers + 12);

	// check if no query indices are left in both queues (-> all indices in buffers)
	INT_TYPE no_indices_left = (tree_record->current_test_index == tree_record->nXtest
			&& cb_is_empty(&(tree_record->queue_reinsert)));

	if (tree_record->buffer_full_warning || no_indices_left) {

		tree_record->empty_all_buffers_calls++;
		process_buffers_brute_force_gpu(tree_record, params, DO_ALL_BRUTE_NO);
		tree_record->buffer_full_warning = 0;

		// all buffers are empty now: let's check if enough work is still there for another round!
		INT_TYPE num_elts_in_queue = cb_get_number_items(&(tree_record->queue_reinsert));
		if (tree_record->current_test_index == tree_record->nXtest
				&& num_elts_in_queue < params->bf_remaining_threshold) {
			process_buffers_brute_force_gpu(tree_record, params, DO_ALL_BRUTE_YES);
		}

	}

	STOP_MY_TIMER(tree_record->timers + 12);

}

void process_buffers_brute_force_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params, INT_TYPE all_brute) {

	// test indices and leaf bounds
	START_MY_TIMER(tree_record->timers + 11);
	INT_TYPE *tindices_removed = (INT_TYPE*) malloc(tree_record->nXtest * sizeof(INT_TYPE));
	INT_TYPE *fr_indices = (INT_TYPE *) malloc(tree_record->nXtest * sizeof(INT_TYPE));
	INT_TYPE *to_indices = (INT_TYPE *) malloc(tree_record->nXtest * sizeof(INT_TYPE));
	INT_TYPE n_tindices_removed = retrieve_indices_from_buffers_gpu(tree_record, params, all_brute, tindices_removed, fr_indices, to_indices);
	STOP_MY_TIMER(tree_record->timers + 11);

	START_MY_TIMER(tree_record->timers + 18);

	// if training data fits on GPU: process in one chunk
	if (n_tindices_removed > 0) {

		if (training_chunks_inactive(tree_record, params)) {
			INT_TYPE chunk_offset = 0;
			do_brute_force_all_leaves_FIRST_gpu(tindices_removed, n_tindices_removed, fr_indices, to_indices, tree_record,
					params, tree_record->nXtrain, tree_record->device_train_patterns_chunk_0, chunk_offset, all_brute, TRAIN_CHUNK_0);
			do_brute_force_all_leaves_SECOND_gpu(tindices_removed, n_tindices_removed, fr_indices, to_indices, tree_record,
					params, tree_record->nXtrain, tree_record->device_train_patterns_chunk_0, chunk_offset, all_brute, TRAIN_CHUNK_0);
		} else {

			process_buffers_brute_force_in_chunks_gpu(tree_record, params, all_brute, tindices_removed,
					n_tindices_removed, fr_indices, to_indices);

		}


	}

	STOP_MY_TIMER(tree_record->timers + 18);

	free(fr_indices);
	free(to_indices);
	free(tindices_removed);

}

INT_TYPE retrieve_indices_from_buffers_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params,
		INT_TYPE all_brute, INT_TYPE *tindices_removed, INT_TYPE *fr_indices, INT_TYPE *to_indices) {

	UINT_TYPE i, leaf_idx;
	INT_TYPE n_added = 0;

	if (all_brute == DO_ALL_BRUTE_NO) {

		for (leaf_idx = 0; leaf_idx < tree_record->n_leaves; leaf_idx++) {
			if (cb_get_number_items(tree_record->buffers[leaf_idx]) > 0) {

				INT_TYPE n_tindices_buffer = cb_get_number_items(tree_record->buffers[leaf_idx]);
				cb_read_batch(tree_record->buffers[leaf_idx], tindices_removed + n_added, n_tindices_buffer);

				for (i = 0; i < n_tindices_buffer; i++) {
					fr_indices[n_added + i] = tree_record->leaves[leaf_idx * LEAF_WIDTH];
					to_indices[n_added + i] = tree_record->leaves[leaf_idx * LEAF_WIDTH + 1];
					// reinsert test pattern into test queue
					cb_add_elt(&(tree_record->queue_reinsert), tindices_removed + n_added + i);
				}
				n_added += n_tindices_buffer;
			}
		}

	} else { // DO_ALL_BRUTE_YES

		n_added = cb_get_number_items(&(tree_record->queue_reinsert));
		cb_read_batch(&(tree_record->queue_reinsert), tindices_removed, n_added);
		for (i = 0; i < n_added; i++) {
			fr_indices[i] = 0;
			to_indices[i] = tree_record->nXtrain;
		}
		// do not reinsert test queries (done)

	}

	return n_added;

}

void process_buffers_brute_force_in_chunks_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params, INT_TYPE all_brute,
		INT_TYPE *tindices_removed, INT_TYPE n_tindices_removed, INT_TYPE *fr_indices, INT_TYPE *to_indices) {

	INT_TYPE i;
	INT_TYPE chunk_start = 0;
	cl_mem current_train_patterns_chunk;

	// first chunk is copied in the initialization phase or at the end of the previous call

	// retrieve test indices that should be part of the current chunk
	INT_TYPE *tindices_chunk = (INT_TYPE*) malloc(tree_record->nXtest * sizeof(INT_TYPE));
	INT_TYPE *fr_indices_chunk = (INT_TYPE*) malloc(tree_record->nXtest * sizeof(INT_TYPE));
	INT_TYPE *to_indices_chunk = (INT_TYPE*) malloc(tree_record->nXtest * sizeof(INT_TYPE));

	INT_TYPE last_processed_tindex = 0;

	while (chunk_start < tree_record->nXtrain) {

		INT_TYPE chunk_end = chunk_start + tree_record->n_patts_per_chunk;
		if (chunk_end > tree_record->nXtrain) {
			chunk_end = tree_record->nXtrain;
		}
		INT_TYPE chunk_offset = chunk_start;

		// initiate next copying process (next chunk)
		INT_TYPE chunk_next_end = chunk_end + tree_record->n_patts_per_chunk;
		if (chunk_next_end > tree_record->nXtrain) {
			chunk_next_end = tree_record->nXtrain;
		}

		START_MY_TIMER(tree_record->timers + 23);
		// prepare chunk of test indices with associated bounds
		INT_TYPE n_tindices_chunk = 0;
		if (!all_brute) {
			for (i = last_processed_tindex; i < n_tindices_removed; i++) {
				// three cases
				if ((fr_indices[i] >= chunk_start) && (to_indices[i] <= chunk_end)) {
					tindices_chunk[n_tindices_chunk] = tindices_removed[i];
					fr_indices_chunk[n_tindices_chunk] = fr_indices[i];
					to_indices_chunk[n_tindices_chunk] = to_indices[i];
					n_tindices_chunk++;
				} else if ((fr_indices[i] < chunk_end) && (to_indices[i] > chunk_end)) {
					tindices_chunk[n_tindices_chunk] = tindices_removed[i];
					fr_indices_chunk[n_tindices_chunk] = fr_indices[i];
					to_indices_chunk[n_tindices_chunk] = chunk_end;
					n_tindices_chunk++;
				} else if ((fr_indices[i] < chunk_start) && (to_indices[i] > chunk_start)) {
					tindices_chunk[n_tindices_chunk] = tindices_removed[i];
					fr_indices_chunk[n_tindices_chunk] = chunk_start;
					to_indices_chunk[n_tindices_chunk] = to_indices[i];
					n_tindices_chunk++;
				}
			}
			last_processed_tindex = n_tindices_chunk;
		} else {
			for (i = 0; i < n_tindices_removed; i++) {
				tindices_chunk[n_tindices_chunk] = tindices_removed[i];
				fr_indices_chunk[n_tindices_chunk] = chunk_start;
				to_indices_chunk[n_tindices_chunk] = chunk_end;
				n_tindices_chunk++;
			}

		}

		if (tree_record->current_chunk_id == TRAIN_CHUNK_0) {
			current_train_patterns_chunk = tree_record->device_train_patterns_chunk_0;
		} else {
			current_train_patterns_chunk = tree_record->device_train_patterns_chunk_1;
		}
		STOP_MY_TIMER(tree_record->timers + 23);

		// define a new brute force flag: if processed in chunks,
		// then one should NOT do brute force in the remaining chunks
		INT_TYPE do_all_brute = all_brute;
		if (all_brute && chunk_start > 0) {
			do_all_brute = 0;
		}

		START_MY_TIMER(tree_record->timers + 24);
		if (n_tindices_chunk > 0) {
			do_brute_force_all_leaves_FIRST_gpu(tindices_chunk, n_tindices_chunk, fr_indices_chunk, to_indices_chunk,
					tree_record, params, chunk_end - chunk_start, current_train_patterns_chunk, chunk_offset,
					do_all_brute, tree_record->current_chunk_id);
		}

		// copy next chunk for current iteration
		if (chunk_end < tree_record->nXtrain) {
			copy_train_patterns_to_device(tree_record, params, (tree_record->current_chunk_id + 1) % 2, chunk_end, chunk_next_end);
		// copy initial chunk for NEXT iteration
		} else {
			copy_train_patterns_to_device(tree_record, params, (tree_record->current_chunk_id + 1) % 2, 0, tree_record->n_patts_per_chunk);
		}

		if (n_tindices_chunk > 0) {

			do_brute_force_all_leaves_SECOND_gpu(tindices_chunk, n_tindices_chunk, fr_indices_chunk, to_indices_chunk,
					tree_record, params, chunk_end - chunk_start, current_train_patterns_chunk, chunk_offset,
					do_all_brute, tree_record->current_chunk_id);
		}

		chunk_start += tree_record->n_patts_per_chunk;
		tree_record->current_chunk_id = (tree_record->current_chunk_id + 1) % 2;
		STOP_MY_TIMER(tree_record->timers + 24);

	}

	// free memory
	free(tindices_chunk);
	free(fr_indices_chunk);
	free(to_indices_chunk);

}

/* -------------------------------------------------------------------------------- 
 * Apply brute-force approach for all leaves
 * -------------------------------------------------------------------------------- 
 */
void do_brute_force_all_leaves_FIRST_gpu(INT_TYPE *test_indices,
	INT_TYPE n_test_indices,
	INT_TYPE *fr_indices,
	INT_TYPE *to_indices, TREE_RECORD *tree_record, TREE_PARAMETERS *params,
	INT_TYPE n_device_train_patterns, cl_mem device_train_patterns,
	INT_TYPE chunk_offset,
	INT_TYPE all_brute, INT_TYPE current_chunk) {

	cl_int err;
	cl_event event;
	cl_event event1, event_fr, event_to;

	START_MY_TIMER(tree_record->timers + 25);

	if (all_brute) {
		PRINT(params)("Starting final brute-force phase ...\n");
		fflush(stdout);
		START_MY_TIMER(tree_record->timers + 20);
	}

	// write data to buffers
	err = clEnqueueWriteBuffer(tree_record->gpu_command_queue,
			tree_record->device_test_indices_removed_from_all_buffers, CL_FALSE, 0,
			n_test_indices * sizeof(INT_TYPE), test_indices, 0, NULL, &event1);
	check_cl_error(err, __FILE__, __LINE__);

	err = clEnqueueWriteBuffer(tree_record->gpu_command_queue, tree_record->device_fr_indices, CL_FALSE, 0,
			n_test_indices * sizeof(INT_TYPE), fr_indices, 0, NULL, &event_fr);
	check_cl_error(err, __FILE__, __LINE__);

	err = clEnqueueWriteBuffer(tree_record->gpu_command_queue, tree_record->device_to_indices, CL_FALSE, 0,
			n_test_indices * sizeof(INT_TYPE), to_indices, 0, NULL, &event_to);
	check_cl_error(err, __FILE__, __LINE__);

	/************************************* SET SUBSET KERNEL ************************************/
	START_MY_TIMER(tree_record->timers + 13);

	size_t global_size_test_subset[] = { WORKGROUP_SIZE_TEST_SUBSET
			* ((int) n_test_indices / WORKGROUP_SIZE_TEST_SUBSET) + WORKGROUP_SIZE_TEST_SUBSET };
	size_t local_size_test_subset[] = { WORKGROUP_SIZE_TEST_SUBSET };

	// set kernel parameters: generate_test_subset_kernel
	err = clSetKernelArg(tree_record->generate_test_subset_kernel, 0, sizeof(INT_TYPE), &n_test_indices);
	err |= clSetKernelArg(tree_record->generate_test_subset_kernel, 1, sizeof(cl_mem),
			&(tree_record->device_test_indices_removed_from_all_buffers));
	err |= clSetKernelArg(tree_record->generate_test_subset_kernel, 2, sizeof(cl_mem),
			&(tree_record->device_test_patterns));
	err |= clSetKernelArg(tree_record->generate_test_subset_kernel, 3, sizeof(cl_mem),
			&(tree_record->device_test_patterns_subset_tmp));
	check_cl_error(err, __FILE__, __LINE__);

	// wait for test indices to be copied
	err = clWaitForEvents(1, &event1);
	err |= clReleaseEvent(event1);
	check_cl_error(err, __FILE__, __LINE__);

	// call kernel generate_test_subset_kernel
	err |= clEnqueueNDRangeKernel(tree_record->gpu_command_queue, tree_record->generate_test_subset_kernel, 1, NULL,
			global_size_test_subset, local_size_test_subset, 0, NULL, &event);
	err |= clWaitForEvents(1, &event);
	err |= clReleaseEvent(event);
	check_cl_error(err, __FILE__, __LINE__);

	err = clFinish(tree_record->gpu_command_queue);
	check_cl_error(err, __FILE__, __LINE__);

	STOP_MY_TIMER(tree_record->timers + 13);
	/********************************************************************************************/

	/******************************** RETRIEVE KNOWN DISTANCES **********************************/
	START_MY_TIMER(tree_record->timers + 15);

	size_t global_size_retrieve[] = { WORKGROUP_SIZE_UPDATE * ((INT_TYPE) n_test_indices / WORKGROUP_SIZE_UPDATE)
			+ WORKGROUP_SIZE_UPDATE };
	size_t local_size_retrieve[] = { WORKGROUP_SIZE_UPDATE };

	// set kernel parameters: retrieve_dist_kernel
	err |= clSetKernelArg(tree_record->retrieve_dist_kernel, 0, sizeof(INT_TYPE), &n_test_indices);
	err |= clSetKernelArg(tree_record->retrieve_dist_kernel, 1, sizeof(cl_mem), &(tree_record->device_test_indices_removed_from_all_buffers));
	err |= clSetKernelArg(tree_record->retrieve_dist_kernel, 2, sizeof(cl_mem), &(tree_record->device_d_mins));
	err |= clSetKernelArg(tree_record->retrieve_dist_kernel, 3, sizeof(cl_mem), &(tree_record->device_idx_mins));
	err |= clSetKernelArg(tree_record->retrieve_dist_kernel, 4, sizeof(cl_mem), &(tree_record->device_dist_mins_tmp));
	err |= clSetKernelArg(tree_record->retrieve_dist_kernel, 5, sizeof(cl_mem), &(tree_record->device_idx_mins_tmp));
	check_cl_error(err, __FILE__, __LINE__);

	// call kernel update_distances_kernel
	err = clEnqueueNDRangeKernel(tree_record->gpu_command_queue, tree_record->retrieve_dist_kernel, 1, NULL,
			global_size_retrieve, local_size_retrieve, 0, NULL, &event);
	check_cl_error(err, __FILE__, __LINE__);
	err = clWaitForEvents(1, &event);
	err |= clReleaseEvent(event);
	check_cl_error(err, __FILE__, __LINE__);

	STOP_MY_TIMER(tree_record->timers + 15);
	/********************************************************************************************/

	/************************************ BRUTE FORCE KERNEL ************************************/
	// wait for fr_indices and to_indices
	err = clWaitForEvents(1, &event_fr);
	err |= clReleaseEvent(event_fr);
	check_cl_error(err, __FILE__, __LINE__);
	err = clWaitForEvents(1, &event_to);
	err |= clReleaseEvent(event_to);
	check_cl_error(err, __FILE__, __LINE__);

	cl_command_queue cmd_queue_chunk;
	if (current_chunk == TRAIN_CHUNK_0) {
		cmd_queue_chunk = tree_record->gpu_command_queue_chunk_0;
	} else {
		cmd_queue_chunk = tree_record->gpu_command_queue_chunk_1;
	}

	// set kernel parameters: brute_all_leaves_nearest_neighbors_kernel
	err = clSetKernelArg(tree_record->brute_nn_kernel, 0, sizeof(INT_TYPE), &n_test_indices);
	err |= clSetKernelArg(tree_record->brute_nn_kernel, 3, sizeof(INT_TYPE), &n_device_train_patterns);
	err |= clSetKernelArg(tree_record->brute_nn_kernel, 4, sizeof(INT_TYPE), &all_brute);
	err |= clSetKernelArg(tree_record->brute_nn_kernel, 5, sizeof(cl_mem), &device_train_patterns);
	err |= clSetKernelArg(tree_record->brute_nn_kernel, 6, sizeof(cl_mem), &(tree_record->device_test_patterns_subset_tmp));
	err |= clSetKernelArg(tree_record->brute_nn_kernel, 7, sizeof(cl_mem), &(tree_record->device_fr_indices));
	err |= clSetKernelArg(tree_record->brute_nn_kernel, 8, sizeof(cl_mem), &(tree_record->device_to_indices));
	err |= clSetKernelArg(tree_record->brute_nn_kernel, 9, sizeof(cl_mem), &(tree_record->device_dist_mins_tmp));
	err |= clSetKernelArg(tree_record->brute_nn_kernel, 10, sizeof(cl_mem), &(tree_record->device_idx_mins_tmp));
	err |= clSetKernelArg(tree_record->brute_nn_kernel, 11, sizeof(INT_TYPE), &chunk_offset);
	check_cl_error(err, __FILE__, __LINE__);

	INT_TYPE n_chunks = 1;
	if (n_test_indices > MAX_CHUNK_BRUTE_KERNEL){
		n_chunks = (INT_TYPE) ceil(((double)n_test_indices) / MAX_CHUNK_BRUTE_KERNEL);
	}
	INT_TYPE n_instances_chunk = (INT_TYPE) ceil(((double)n_test_indices) / n_chunks);

	INT_TYPE chunk_start = 0;
	INT_TYPE chunk_end = n_instances_chunk;

	size_t global_size_nn[] = { WORKGROUP_SIZE_BRUTE * ((INT_TYPE) n_instances_chunk / WORKGROUP_SIZE_BRUTE) + WORKGROUP_SIZE_BRUTE };
	size_t localSize_nn[] = { WORKGROUP_SIZE_BRUTE };

	// wait for chunk to be ready for processing
	START_MY_TIMER(tree_record->timers + 22);
	err = clFinish(cmd_queue_chunk);
	check_cl_error(err, __FILE__, __LINE__);
	STOP_MY_TIMER(tree_record->timers + 22);

	STOP_MY_TIMER(tree_record->timers + 25);
	START_MY_TIMER(tree_record->timers + 14);
	while (chunk_start < n_test_indices){

		INT_TYPE test_indices_offset = chunk_start;
		if (chunk_end > n_test_indices){
			chunk_end = n_test_indices;
		}
		INT_TYPE n_indices = chunk_end - chunk_start;

		err = clSetKernelArg(tree_record->brute_nn_kernel, 1, sizeof(INT_TYPE), &n_indices);
		err |= clSetKernelArg(tree_record->brute_nn_kernel, 2, sizeof(INT_TYPE), &test_indices_offset);
		check_cl_error(err, __FILE__, __LINE__);

		// call kernel brute_all_leaves_nearest_neighbors_kernel
		err = clEnqueueNDRangeKernel(cmd_queue_chunk,
				tree_record->brute_nn_kernel, 1, NULL, global_size_nn,
				localSize_nn, 0, NULL, NULL);
		check_cl_error(err, __FILE__, __LINE__);

		chunk_start += n_instances_chunk;
		chunk_end += n_instances_chunk;

	}

	// CONTINUED in do_brute_force_all_leaves_SECOND_gpu

}

void do_brute_force_all_leaves_SECOND_gpu(INT_TYPE *test_indices,
	INT_TYPE n_test_indices,
	INT_TYPE *fr_indices,
	INT_TYPE *to_indices, TREE_RECORD *tree_record, TREE_PARAMETERS *params,
	INT_TYPE n_device_train_patterns, cl_mem device_train_patterns,
	INT_TYPE chunk_offset,
	INT_TYPE all_brute, INT_TYPE current_chunk) {

	cl_int err;

	cl_command_queue cmd_queue_chunk;
	if (current_chunk == TRAIN_CHUNK_0) {
		cmd_queue_chunk = tree_record->gpu_command_queue_chunk_0;
	} else {
		cmd_queue_chunk = tree_record->gpu_command_queue_chunk_1;
	}

	// wait for brute force nearest neighbor computation
	// if errors occur here, use an event to wait for the previous
	// kernel launch (above: tree_record->brute_nn_kernel) to check
	// for the particular error
	err = clFinish(cmd_queue_chunk);
	check_cl_error(err, __FILE__, __LINE__);

	STOP_MY_TIMER(tree_record->timers + 14);
	START_MY_TIMER(tree_record->timers + 26);
	/********************************************************************************************/

	/************************************* UPDATE DISTANCES *************************************/
	START_MY_TIMER(tree_record->timers + 15);

	size_t global_size_update[] = { WORKGROUP_SIZE_UPDATE * ((INT_TYPE) n_test_indices / WORKGROUP_SIZE_UPDATE) + WORKGROUP_SIZE_UPDATE };
	size_t local_size_update[] = { WORKGROUP_SIZE_UPDATE };

	// set kernel parameters: update_distances_kernel
	err |= clSetKernelArg(tree_record->update_dist_kernel, 0, sizeof(INT_TYPE), &n_test_indices);
	err |= clSetKernelArg(tree_record->update_dist_kernel, 1, sizeof(INT_TYPE), &all_brute);
	err |= clSetKernelArg(tree_record->update_dist_kernel, 2, sizeof(cl_mem),
			&(tree_record->device_test_indices_removed_from_all_buffers));
	err |= clSetKernelArg(tree_record->update_dist_kernel, 3, sizeof(cl_mem), &(tree_record->device_d_mins));
	err |= clSetKernelArg(tree_record->update_dist_kernel, 4, sizeof(cl_mem), &(tree_record->device_idx_mins));
	err |= clSetKernelArg(tree_record->update_dist_kernel, 5, sizeof(cl_mem), &(tree_record->device_dist_mins_tmp));
	err |= clSetKernelArg(tree_record->update_dist_kernel, 6, sizeof(cl_mem), &(tree_record->device_idx_mins_tmp));
	check_cl_error(err, __FILE__, __LINE__);

	// call kernel update_distances_kernel
	err = clEnqueueNDRangeKernel(tree_record->gpu_command_queue, tree_record->update_dist_kernel, 1, NULL,
			global_size_update, local_size_update, 0, NULL, NULL);
	check_cl_error(err, __FILE__, __LINE__);

	err = clFinish(tree_record->gpu_command_queue);
	check_cl_error(err, __FILE__, __LINE__);

	STOP_MY_TIMER(tree_record->timers + 15);
	/********************************************************************************************/

	// free memory
	START_MY_TIMER(tree_record->timers + 27);


	tree_record->counters[1]++;
	STOP_MY_TIMER(tree_record->timers + 27);

	if (all_brute) {
		PRINT(params)("Final brute-force done!\n");
		STOP_MY_TIMER(tree_record->timers + 20);
	}

	STOP_MY_TIMER(tree_record->timers + 26);

}

/* -------------------------------------------------------------------------------- 
 * Finds the next leaf indices for all test patterns indexed by all_next_indices.
 * -------------------------------------------------------------------------------- 
 */
void find_leaf_idx_batch_gpu(INT_TYPE *all_next_indices, INT_TYPE num_all_next_indices, INT_TYPE *ret_vals,
		TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	cl_int err;
	cl_event event;

	START_MY_TIMER(tree_record->timers + 16);

	tree_record->find_leaf_idx_calls++;

	// generate buffers on the GPU
	tree_record->device_all_next_indices = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_all_next_indices * sizeof(INT_TYPE), all_next_indices, &err);
	check_cl_error(err, __FILE__, __LINE__);

	tree_record->device_ret_vals = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_WRITE, num_all_next_indices * sizeof(INT_TYPE), NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// set kernel parameters
	err = clSetKernelArg(tree_record->find_leaves_kernel, 0, sizeof(cl_mem), &(tree_record->device_all_next_indices));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 1, sizeof(INT_TYPE), &num_all_next_indices);
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 2, sizeof(cl_mem), &(tree_record->device_test_patterns));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 3, sizeof(cl_mem), &(tree_record->device_ret_vals));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 4, sizeof(cl_mem), &(tree_record->device_all_depths));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 5, sizeof(cl_mem), &(tree_record->device_all_idxs));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 6, sizeof(cl_mem), &(tree_record->device_all_stacks));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 7, sizeof(INT_TYPE), &(params->tree_depth));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 8, sizeof(cl_mem), &(tree_record->device_d_mins));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 9, sizeof(cl_mem), &(tree_record->device_idx_mins));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 10, sizeof(cl_mem), &(tree_record->device_nodes));
	err |= clSetKernelArg(tree_record->find_leaves_kernel, 11, sizeof(cl_mem), &(tree_record->device_leave_bounds));
	check_cl_error(err, __FILE__, __LINE__);

	size_t global_size[] = { WORKGROUP_SIZE_LEAVES * ((INT_TYPE) num_all_next_indices / WORKGROUP_SIZE_LEAVES) + WORKGROUP_SIZE_LEAVES };
	size_t local_size[] = { WORKGROUP_SIZE_LEAVES };

	err = clEnqueueNDRangeKernel(tree_record->gpu_command_queue, tree_record->find_leaves_kernel, 1, NULL, global_size, local_size, 0, NULL, &event);
	// we wait here until the nearest neighbor computations are done (could be optimized)
	err |= clWaitForEvents(1, &event);
	err |= clReleaseEvent(event);
	check_cl_error(err, __FILE__, __LINE__);

	// copy indices found back to CPU
	err = clEnqueueReadBuffer(tree_record->gpu_command_queue, tree_record->device_ret_vals,
			CL_TRUE, 0, num_all_next_indices * sizeof(INT_TYPE), tree_record->leaf_indices_batch_ret_vals, 0, NULL, &event);
	// we wait here until the nearest neighbor computations are done (could be optimized)
	err |= clWaitForEvents(1, &event);
	err |= clReleaseEvent(event);
	check_cl_error(err, __FILE__, __LINE__);

	// free memory
	clReleaseMemObject(tree_record->device_all_next_indices);
	clReleaseMemObject(tree_record->device_ret_vals);

	STOP_MY_TIMER(tree_record->timers + 16);

}

/* -------------------------------------------------------------------------------- 
 * Copies the arrays dist_min_global and idx_min_global from GPU to CPU
 * Updates the distances and indices (w.r.t the original indices)
 * -------------------------------------------------------------------------------- 
 */
void get_distances_and_indices_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	cl_int err1, err2;
	cl_event event1, event2; // event_indices_copy;

	size_t global_size[] = { WORKGROUP_SIZE_COMBINE
			* ((INT_TYPE) tree_record->nXtest * params->n_neighbors / WORKGROUP_SIZE_COMBINE) + WORKGROUP_SIZE_COMBINE };
	size_t local_size[] = { WORKGROUP_SIZE_COMBINE };

	err1 = clSetKernelArg(tree_record->compute_final_dists_idxs_kernel, 0, sizeof(INT_TYPE), &(tree_record->nXtest));
	err1 |= clSetKernelArg(tree_record->compute_final_dists_idxs_kernel, 1, sizeof(cl_mem),
			&(tree_record->device_d_mins));
	err1 |= clSetKernelArg(tree_record->compute_final_dists_idxs_kernel, 2, sizeof(cl_mem),
			&(tree_record->device_idx_mins));
	check_cl_error(err1, __FILE__, __LINE__);

	err1 = clEnqueueNDRangeKernel(tree_record->gpu_command_queue, tree_record->compute_final_dists_idxs_kernel, 1, NULL,
			global_size, local_size, 0, NULL, &event1);
	err1 |= clWaitForEvents(1, &event1);
	err1 |= clReleaseEvent(event1);
	check_cl_error(err1, __FILE__, __LINE__);

	// copy back to host system
	err1 = clEnqueueReadBuffer(tree_record->gpu_command_queue, tree_record->device_d_mins, CL_FALSE, 0,
			tree_record->nXtest * params->n_neighbors * sizeof(FLOAT_TYPE), tree_record->dist_mins_global, 0, NULL,
			&event1);
	err2 = clEnqueueReadBuffer(tree_record->gpu_command_queue, tree_record->device_idx_mins, CL_TRUE, 0,
			tree_record->nXtest * params->n_neighbors * sizeof(INT_TYPE), tree_record->idx_mins_global, 0, NULL,
			&event2);
	err1 |= clWaitForEvents(1, &event1);
	err1 |= clReleaseEvent(event1);
	check_cl_error(err1, __FILE__, __LINE__);
	err2 |= clWaitForEvents(1, &event2);
	err2 |= clReleaseEvent(event2);
	check_cl_error(err2, __FILE__, __LINE__);

	// overwrite indices with original ones
	INT_TYPE i, j;
	for (i = 0; i < tree_record->nXtest; i++) {
		for (j = 0; j < params->n_neighbors; j++) {
			INT_TYPE idx = tree_record->idx_mins_global[i * params->n_neighbors + j];
			tree_record->idx_mins_global[i * params->n_neighbors + j] = tree_record->Itrain_sorted[idx];
		}
	}
}

/* -------------------------------------------------------------------------------- 
 * Writes the training patterns in a specific ordering
 * (needed for coalesced data access on the GPU etc.
 * -------------------------------------------------------------------------------- 
 */
void write_sorted_training_patterns_gpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	INT_TYPE i;

	for (i = 0; i < tree_record->nXtrain; i++) {

		memcpy(tree_record->Xtrain_sorted + i * tree_record->dXtrain,
				tree_record->XtrainI + i * (tree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(INT_TYPE)),
				tree_record->dXtrain * sizeof(FLOAT_TYPE));
		tree_record->Itrain_sorted[i] = *((INT_TYPE *) (tree_record->XtrainI + tree_record->dXtrain * sizeof(FLOAT_TYPE)
				+ i * (tree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(INT_TYPE))));

	}

}

void init_train_patterns_buffers(TREE_RECORD *tree_record, TREE_PARAMETERS *params,
		INT_TYPE chunk, INT_TYPE n_indices) {

	cl_int err;

	if (chunk == TRAIN_CHUNK_0) {
		tree_record->device_train_patterns_chunk_0 = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
				n_indices * tree_record->dXtrain * sizeof(FLOAT_TYPE), NULL, &err);
		tree_record->host_pinned_train_patterns_chunk_0 = (FLOAT_TYPE*)clEnqueueMapBuffer(tree_record->gpu_command_queue_chunk_0,
				tree_record->device_train_patterns_chunk_0, CL_TRUE, CL_MAP_WRITE, 0,
				n_indices * tree_record->dXtrain * sizeof(FLOAT_TYPE), 0, NULL, NULL, &err);
		check_cl_error(err, __FILE__, __LINE__);
	} else if (chunk == TRAIN_CHUNK_1) {
		tree_record->device_train_patterns_chunk_1 = clCreateBuffer(tree_record->gpu_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
				n_indices * tree_record->dXtrain * sizeof(FLOAT_TYPE), NULL, &err);
		tree_record->host_pinned_train_patterns_chunk_1 = (FLOAT_TYPE*)clEnqueueMapBuffer(tree_record->gpu_command_queue_chunk_1,
				tree_record->device_train_patterns_chunk_1, CL_TRUE, CL_MAP_WRITE, 0,
				n_indices * tree_record->dXtrain * sizeof(FLOAT_TYPE), 0, NULL, NULL, &err);
		check_cl_error(err, __FILE__, __LINE__);
	} else {
		printf("Only two chunks allowed: %i\n", chunk);
		exit(EXIT_FAILURE);
	}

}

void copy_train_patterns_to_device(TREE_RECORD *tree_record, TREE_PARAMETERS *params, INT_TYPE chunk,
		INT_TYPE start_idx, INT_TYPE end_idx) {

	// Interleaved Compute/Copy: http://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/OpenCL_Best_Practices_Guide.pdf

	cl_int err;

	tree_record->counters[0] += 1;
	INT_TYPE num_indices = end_idx - start_idx;

	if (chunk == TRAIN_CHUNK_0) {

		START_MY_TIMER(tree_record->timers + 21);
		memcpy(tree_record->host_pinned_train_patterns_chunk_0,
				tree_record->Xtrain_sorted + start_idx * tree_record->dXtrain,
				num_indices * tree_record->dXtrain * sizeof(FLOAT_TYPE));
		STOP_MY_TIMER(tree_record->timers + 21);

		START_MY_TIMER(tree_record->timers + 19);
		err = clEnqueueWriteBuffer(tree_record->gpu_command_queue_chunk_0, tree_record->device_train_patterns_chunk_0,
				CL_FALSE, 0, num_indices * tree_record->dXtrain * sizeof(FLOAT_TYPE),
				tree_record->host_pinned_train_patterns_chunk_0, 0, NULL, NULL);
		check_cl_error(err, __FILE__, __LINE__);
		STOP_MY_TIMER(tree_record->timers + 19);

	} else if (chunk == TRAIN_CHUNK_1) {

		START_MY_TIMER(tree_record->timers + 21);
		memcpy(tree_record->host_pinned_train_patterns_chunk_1,
				tree_record->Xtrain_sorted + start_idx * tree_record->dXtrain,
				num_indices * tree_record->dXtrain * sizeof(FLOAT_TYPE));
		STOP_MY_TIMER(tree_record->timers + 21);

		START_MY_TIMER(tree_record->timers + 19);
		err = clEnqueueWriteBuffer(tree_record->gpu_command_queue_chunk_1, tree_record->device_train_patterns_chunk_1,
				CL_FALSE, 0, num_indices * tree_record->dXtrain * sizeof(FLOAT_TYPE),
				tree_record->host_pinned_train_patterns_chunk_1, 0, NULL, NULL);
		check_cl_error(err, __FILE__, __LINE__);
		STOP_MY_TIMER(tree_record->timers + 19);

	} else {

		printf("Only two chunks allowed: %i\n", chunk);
		exit(EXIT_FAILURE);

	}

}

void free_train_patterns_device(TREE_RECORD *tree_record, TREE_PARAMETERS *params, INT_TYPE chunk) {

	cl_int err;

	if (chunk == TRAIN_CHUNK_0) {
		err = clReleaseMemObject(tree_record->device_train_patterns_chunk_0);
		check_cl_error(err, __FILE__, __LINE__);
	} else if (chunk == TRAIN_CHUNK_1) {
		err = clReleaseMemObject(tree_record->device_train_patterns_chunk_1);
		check_cl_error(err, __FILE__, __LINE__);
	} else {
		printf("Only two chunks allowed: %i\n", chunk);
		exit(EXIT_FAILURE);
	}

}

int training_chunks_inactive(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	return params->n_train_chunks == 1;

}
