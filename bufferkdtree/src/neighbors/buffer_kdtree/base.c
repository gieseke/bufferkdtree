/* 
 * bufferkdtree.c
 */

#include "include/base.h"

// OUT_OF_RESOURCES (OpenCL error -5)
// http://stackoverflow.com/questions/3988645/cl-out-of-resources-for-2-millions-floats-with-1gb-vram

/* --------------------------------------------------------------------------------
 * Interface (extern): Initialize components
 * --------------------------------------------------------------------------------
 */
void init_extern(int n_neighbors, int tree_depth, int num_threads, int num_nXtrain_chunks, int platform_id,
		int device_id, double allowed_train_mem_percent_chunk, double allowed_test_mem_percent,
		int splitting_type, char *kernels_source_directory,
		int verbosity_level, TREE_PARAMETERS *params) {

	set_default_parameters(params);

	params->n_neighbors = n_neighbors;
	params->tree_depth = tree_depth;
	params->num_threads = num_threads;
	params->n_train_chunks = num_nXtrain_chunks;
	params->splitting_type = splitting_type;
	params->kernels_source_directory = (char*) malloc((strlen(kernels_source_directory) + 10) * sizeof(char));
	strcpy(params->kernels_source_directory, kernels_source_directory);
	params->verbosity_level = verbosity_level;
	params->platform_id = platform_id;
	params->device_id = device_id;
	params->allowed_train_mem_percent_chunk = allowed_train_mem_percent_chunk;
	params->allowed_test_mem_percent = allowed_test_mem_percent;
	check_parameters(params);

	omp_set_num_threads(params->num_threads);

}

/* -------------------------------------------------------------------------------- 
 * Builds a buffer k-d-tree
 * -------------------------------------------------------------------------------- 
 */
void build_bufferkdtree(FLOAT_TYPE * Xtrain, INT_TYPE nXtrain, INT_TYPE dXtrain, TREE_RECORD *tree_record,
		TREE_PARAMETERS *params) {

	int i;
	for (i = 0; i < 25; i++) {
		INIT_MY_TIMER(tree_record->timers + i);
	}

	int err_device = get_device_infos(params->platform_id, params->device_id, &(tree_record->device_infos));

	if (err_device < 0){
		printf("Error: Could not retrieve device information!");
		exit(EXIT_FAILURE);
	}

	// update tree record parameters
	tree_record->dXtrain = dXtrain;
	tree_record->nXtrain = nXtrain;
	tree_record->n_nodes = pow(2, params->tree_depth) - 1;
	tree_record->n_leaves = pow(2, params->tree_depth);
	tree_record->max_visited = 6 * (tree_record->n_leaves - 3 + 1);

	// variable buffer sizes: increase/decrease depending on the three depth
	if (params->tree_depth > 16) {
		tree_record->leaves_initial_buffer_sizes = 128;
		PRINT(params)("Warning: tree depth %i might be too large (memory consumption)!", params->tree_depth);
	} else {
		tree_record->leaves_initial_buffer_sizes = pow(2, 24 - params->tree_depth);
	}

	// memory needed for storing training data (in bytes)
	double device_mem_bytes = (double)tree_record->device_infos.device_mem_bytes;
	double device_max_alloc_bytes = (double)tree_record->device_infos.device_max_alloc_bytes;

	double train_mem_bytes = get_raw_train_mem_device_bytes(tree_record, params);
	PRINT(params)("Memory needed for all training patterns: %f (GB)\n", train_mem_bytes / MEM_GB);

	if (train_mem_bytes / params->n_train_chunks > device_mem_bytes * params->allowed_train_mem_percent_chunk) {
		params->n_train_chunks = (INT_TYPE) ceil(train_mem_bytes / (device_mem_bytes * params->allowed_train_mem_percent_chunk));
		// if set automatically, then use at least 3 chunks (hide computations and data transfer)
		if (params->n_train_chunks < 3){
			params->n_train_chunks = 3;
		}

		PRINT(params)("WARNING: Increasing number of chunks to %i ...\n", params->n_train_chunks);
	}

	double train_chunk_gb = get_train_mem_with_chunks_device_bytes(tree_record, params);
	if (params->n_train_chunks > 1){
		PRINT(params)("Memory allocated for both chunks: %f (GB)\n", (2*train_chunk_gb) / MEM_GB);
	}


	// we empty a buffer as soon as it has reached a certain filling status (here: 50%)
	tree_record->leaves_buffer_sizes_threshold = 0.9 * tree_record->leaves_initial_buffer_sizes;

	// the amount of indices removed from both queues (input and reinsert) in each round; has to
	// be reasonably large to provide sufficient work for a call to FIND_LEAF_IDX_BATCH
	tree_record->approx_number_of_avail_buffer_slots = 10 * tree_record->leaves_initial_buffer_sizes;

	PRINT(params)("Number of nodes (internal and leaves) in the top tree: %i\n",
			tree_record->n_nodes + tree_record->n_leaves);
	PRINT(params)("Number of buffers attached to the top tree: %i\n", tree_record->n_leaves);
	PRINT(params)("Buffer sizes (leaf structure): %i\n", tree_record->leaves_initial_buffer_sizes);
	PRINT(params)("Buffer empty thresholds: %i\n", tree_record->leaves_buffer_sizes_threshold);
	PRINT(params)("Indices fetched in each round (to fill buffers): %i\n",
			tree_record->approx_number_of_avail_buffer_slots);

	// array that contains the training patterns and the (original indices)
	tree_record->XtrainI = (void*) malloc(
			tree_record->nXtrain * (sizeof(FLOAT_TYPE) * tree_record->dXtrain + sizeof(INT_TYPE)));

	// the nodes and leaves arrays of the buffer kd-tree (host)
	tree_record->nodes = (TREE_NODE *) malloc(tree_record->n_nodes * sizeof(TREE_NODE));
	tree_record->leaves = (FLOAT_TYPE *) malloc(tree_record->n_leaves * LEAF_WIDTH * sizeof(FLOAT_TYPE));

	// create copy of training patterns (along with the original indices)
	kd_tree_generate_training_patterns_indices(tree_record->XtrainI, Xtrain, tree_record->nXtrain,
			tree_record->dXtrain);

	// build kd-tree and store it in nodes (medians) and leaves (fr,to values)
	kd_tree_build_tree(tree_record, params);

	// create copy of sorted training patterns (will be called by brute-force NN search)
	tree_record->Xtrain_sorted = (FLOAT_TYPE *) malloc(tree_record->nXtrain * tree_record->dXtrain * sizeof(FLOAT_TYPE));

	// create copy of original training indices on host system
	tree_record->Itrain_sorted = (INT_TYPE*) malloc(tree_record->nXtrain * sizeof(INT_TYPE));

	WRITE_SORTED_TRAINING_PATTERNS(tree_record, params);

	/* ------------------------------------- OPENCL -------------------------------------- */
	INIT_OPENCL_DEVICES(tree_record, params);
	/* ------------------------------------- OPENCL -------------------------------------- */

}

/* -------------------------------------------------------------------------------- 
 * Testing: Perform the test queries (using the buffer kd-tree).
 * -------------------------------------------------------------------------------- 
 */
void neighbors_extern(FLOAT_TYPE * Xtest, INT_TYPE nXtest, INT_TYPE dXtest,
		FLOAT_TYPE * distances, INT_TYPE ndistances, INT_TYPE ddistances,
		INT_TYPE *indices, INT_TYPE nindices, INT_TYPE dindices,
		TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	START_MY_TIMER(tree_record->timers + 1);

	UINT_TYPE i, j;
	tree_record->find_leaf_idx_calls = 0;
	tree_record->empty_all_buffers_calls = 0;
	tree_record->Xtest = Xtest;
	tree_record->nXtest = nXtest;
	tree_record->dist_mins_global = distances;
	tree_record->idx_mins_global = indices;

	long device_mem_bytes = tree_record->device_infos.device_mem_bytes;
	double test_mem_bytes = get_test_tmp_mem_device_bytes(tree_record, params);
	PRINT(params)("Memory needed for test patterns: %f (GB)\n", test_mem_bytes / MEM_GB);
	if (test_mem_bytes > device_mem_bytes * params->allowed_test_mem_percent) {
		PRINT(params)("Too much memory used for test patterns and temporary data!\n");
		FREE_OPENCL_DEVICES(tree_record, params);
		exit(EXIT_FAILURE);
	}

	double total_device_bytes = get_total_mem_device_bytes(tree_record, params);
	PRINT(params)("Total memory needed on device: %f (GB)\n", total_device_bytes / MEM_GB);

	START_MY_TIMER(tree_record->timers + 4);

	/* ------------------------------------- OPENCL -------------------------------------- */
	INIT_ARRAYS(tree_record, params);
	/* ------------------------------------- OPENCL -------------------------------------- */

	// initialize leaf buffer for test queries (circular buffers)
	tree_record->buffers = (circular_buffer **) malloc(tree_record->n_leaves * sizeof(circular_buffer*));
	for (i = 0; i < tree_record->n_leaves; i++) {
		tree_record->buffers[i] = (circular_buffer *) malloc(sizeof(circular_buffer));
		cb_init(tree_record->buffers[i], tree_record->leaves_initial_buffer_sizes);
	}

	tree_record->buffer_full_warning = 0;

	// initialize queue "input" (we can have at most number_test_patterns in there)
	cb_init(&(tree_record->queue_reinsert), tree_record->nXtest);

	/* ------------------------------------- OPENCL -------------------------------------- */
	START_MY_TIMER(tree_record->timers + 3);
	ALLOCATE_MEMORY_OPENCL_DEVICES(tree_record, params);
	STOP_MY_TIMER(tree_record->timers + 3);
	/* ------------------------------------- OPENCL -------------------------------------- */

	UINT_TYPE iter = 0;
	UINT_TYPE test_printed = 0;

	// allocate space for the indices added in each round; we cannot have more than original test patterns ...
	INT_TYPE *all_next_indices = (INT_TYPE *) malloc(
			tree_record->approx_number_of_avail_buffer_slots * sizeof(INT_TYPE));

	// allocate space for all return values (by FIND_LEAF_IDX_BATCH)
	tree_record->leaf_indices_batch_ret_vals = (INT_TYPE *) malloc(
			tree_record->approx_number_of_avail_buffer_slots * sizeof(INT_TYPE));

	UINT_TYPE num_elts_added;
	tree_record->current_test_index = 0;
	INT_TYPE reinsert_counter = 0;

	PRINT(params)("Starting Querying process via buffer tree...\n");

	STOP_MY_TIMER(tree_record->timers + 4);
	START_MY_TIMER(tree_record->timers + 2);

	do {

		iter++;

		// try to get elements from both queues until buffers are full
		// (each buffer is either empty or has at least space for leaves_buffer_sizes_threshold elements)
		num_elts_added = 0;

		// add enough elements to the buffers ("batch filling")
		while (num_elts_added < tree_record->approx_number_of_avail_buffer_slots
				&& (tree_record->current_test_index < tree_record->nXtest
						|| !cb_is_empty(&(tree_record->queue_reinsert)))) {

			// we remove indices from both queues here (add one element from each queue, if not empty)
			if (!cb_is_empty(&(tree_record->queue_reinsert))) {
				cb_read(&(tree_record->queue_reinsert), all_next_indices + num_elts_added);
			} else {
				all_next_indices[num_elts_added] = tree_record->current_test_index;
				tree_record->current_test_index++;
			}
			num_elts_added++;
		}

		/* ------------------------------------- OPENCL -------------------------------------- */
		FIND_LEAF_IDX_BATCH(all_next_indices, num_elts_added, tree_record->leaf_indices_batch_ret_vals, tree_record,
				params);
		/* ------------------------------------- OPENCL -------------------------------------- */

		// we have added num_elts_added indices to the all_next_indices array
		for (j = 0; j < num_elts_added; j++) {

			INT_TYPE leaf_idx = tree_record->leaf_indices_batch_ret_vals[j];

			// if not done: add the index to the appropriate buffer
			if (leaf_idx != -1) {

				// enlarge buffer if needed
				if (cb_is_full(tree_record->buffers[leaf_idx])) {
					PRINT(params)("Increasing buffer size ...\n");
					tree_record->buffers[leaf_idx] = cb_double_size(tree_record->buffers[leaf_idx]);
				}

				// add next_indices[j] to buffer leaf_idx
				cb_write(tree_record->buffers[leaf_idx], all_next_indices + j);

				if (cb_get_number_items(tree_record->buffers[leaf_idx]) >= tree_record->leaves_buffer_sizes_threshold) {
					tree_record->buffer_full_warning = 1;
				}

			} // else: traversal of test pattern has reached root: done!
		}

		/* ------------------------------------- OPENCL -------------------------------------- */
		PROCESS_ALL_BUFFERS(tree_record, params);
		/* ------------------------------------- OPENCL -------------------------------------- */

		if (tree_record->current_test_index == tree_record->nXtest && !test_printed) {
			PRINT(params)("All query indices are in the buffer tree now (buffers or reinsert queue)...\n");
			test_printed = 1;
		}

	} while (tree_record->current_test_index < tree_record->nXtest || !cb_is_empty(&(tree_record->queue_reinsert)));

	STOP_MY_TIMER(tree_record->timers + 2);

	START_MY_TIMER(tree_record->timers + 5);
	/* ------------------------------------- OPENCL -------------------------------------- */
	GET_DISTANCES_AND_INDICES(tree_record, params);
	/* ------------------------------------- OPENCL -------------------------------------- */

	// free space generated by testing
	for (i = 0; i < tree_record->n_leaves; i++) {
		cb_free(tree_record->buffers[i]);
	}
	STOP_MY_TIMER(tree_record->timers + 5);
	STOP_MY_TIMER(tree_record->timers + 1);

	PRINT(params)("Buffer full indices (overhead)=%i\n", reinsert_counter);
	PRINT(params)("\nNumber of iterations in while loop: \t\t\t\t\t\t\t%i\n", iter);
	PRINT(params)("Number of empty_all_buffers calls: \t\t\t\t\t\t\t%i\n", tree_record->empty_all_buffers_calls);
	PRINT(params)("Number of find_leaf_idx_calls: \t\t\t\t\t\t\t\t%i\n\n", tree_record->find_leaf_idx_calls);

	PRINT(params)("Elapsed total time for querying: \t\t\t\t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 1));
	PRINT(params)("-----------------------------------------------------------------------------------------------------------------------------\n");
	PRINT(params)("(Overhead)  Elapsed time for BEFORE WHILE: \t\t\t\t\t%2.10f\n",
			GET_MY_TIMER(tree_record->timers + 4));
	PRINT(params)("(Overhead)  -> ALLOCATE_MEMORY_OPENCL_DEVICES: \t\t\t\t\t%2.10f\n",
			GET_MY_TIMER(tree_record->timers + 3));

	PRINT(params)(
			"-----------------------------------------------------------------------------------------------------------------------------\n");
	PRINT(params)("Elapsed time in while-loop: \t\t\t\t\t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 2));
	PRINT(params)("(I)    Elapsed time for PROCESS_ALL_BUFFERS: \t\t\t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 12));
	PRINT(params)("(I.A)  Function: retrieve_indices_from_buffers_gpu: \t\t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 11));
	PRINT(params)("(I.B)  Do brute-force (do_brute.../process_buffers_...chunks_gpu : \t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 18));
	PRINT(params)("(I.B.1) -> Elapsed time for clEnqueueWriteBuffer (INTERLEAVED): \t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 19));
	PRINT(params)("(I.B.1) -> Elapsed time for memcpy (INTERLEAVED): \t\t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 21));
	PRINT(params)("(I.B.1) -> Elapsed time for waiting for chunk (in seconds): \t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 22));
	PRINT(params)("(I.B.2) -> Number of copy calls: %i\n", tree_record->counters[0]);

	if (!training_chunks_inactive(tree_record, params)) {
		PRINT(params)("(I.B.4) -> Overhead distributing indices to chunks (in seconds): \t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 23));
		PRINT(params)("(I.B.5) -> Processing of whole chunk (all three phases, in seconds): \t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 24));
		PRINT(params)("(I.B.6) -> Processing of chunk before brute (in seconds): \t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 25));
		PRINT(params)("(I.B.7) -> Processing of chunk after brute (in seconds): \t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 26));
		PRINT(params)("(I.B.8) -> Processing of chunk after brute, buffer release (in seconds): \t%2.10f\n", GET_MY_TIMER(tree_record->timers + 27));
		PRINT(params)("(I.B.9) -> Number of release buffer calls: %i\n", tree_record->counters[0]);
	}
	if (USE_GPU) {

		PRINT(params)("(I.B.3)   -> Elapsed time for TEST_SUBSET (in seconds): \t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 13));
		PRINT(params)("(I.B.4)   -> Elapsed time for NN Search (in seconds): \t\t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 14));
		PRINT(params)("(I.B.5)   -> Elapsed time for UPDATE (in seconds): \t\t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 15));
		PRINT(params)("(I.B.6)   -> Elapsed time for OVERHEAD (in seconds): \t\t\t\t%2.10f\n",
				GET_MY_TIMER(tree_record->timers + 12)
				- GET_MY_TIMER(tree_record->timers + 14)
		    	- GET_MY_TIMER(tree_record->timers + 15)
				- GET_MY_TIMER(tree_record->timers + 13));

	}

	PRINT(params)("(II)   FIND_LEAF_IDX_BATCH : \t\t\t\t\t\t\t%2.10f\n", GET_MY_TIMER(tree_record->timers + 16));
	PRINT(params)("(III) Elapsed time for final brute-force step : \t\t\t\t%2.10f\n\n",
			GET_MY_TIMER(tree_record->timers + 20));

	PRINT(params)("-----------------------------------------------------------------------------------------------------------------------------\n");
	PRINT(params)("(DIFF) While - PROCESS_ALL_BUFFERS - FIND_LEAF_IDX_BATCH: \t\t\t%2.10f\n",
			GET_MY_TIMER(tree_record->timers + 2) - GET_MY_TIMER(tree_record->timers + 12)
					- GET_MY_TIMER(tree_record->timers + 16));
	PRINT(params)("(Overhead)  Elapsed time for AFTER WHILE : \t\t\t\t\t%2.10f\n",
			GET_MY_TIMER(tree_record->timers + 5));
	PRINT(params)("-----------------------------------------------------------------------------------------------------------------------------\n\n");

	PRINT(params)("-----------------------------------------------------------------------------------------------------------------------------\n");
	PRINT(params)("QUERY RUNTIME: %2.10f ", GET_MY_TIMER(tree_record->timers + 1));
	PRINT(params)("PROCESS_ALL_BUFFERS: %2.10f ", GET_MY_TIMER(tree_record->timers + 12));
	PRINT(params)("FIND_LEAF_IDX_BATCH: %2.10f ", GET_MY_TIMER(tree_record->timers + 16));
	PRINT(params)("WHILE_OVERHEAD: %2.10f ",
			GET_MY_TIMER(tree_record->timers + 2) - GET_MY_TIMER(tree_record->timers + 12)
					- GET_MY_TIMER(tree_record->timers + 16));
	PRINT(params)("\n");
	PRINT(params)("-----------------------------------------------------------------------------------------------------------------------------\n");

	// free all allocated memory related to querying
	for (i = 0; i < tree_record->n_leaves; i++) {
		free(tree_record->buffers[i]);
	}
	free(tree_record->buffers);

	// free arrays
	free(tree_record->all_stacks);
	free(tree_record->all_depths);
	free(tree_record->all_idxs);
	free(all_next_indices);
	free(tree_record->leaf_indices_batch_ret_vals);

}

/* -------------------------------------------------------------------------------- 
 * Frees host resources.
 * -------------------------------------------------------------------------------- 
 */
void extern_free_resources(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	/* ------------------------------------- OPENCL -------------------------------------- */
	FREE_OPENCL_DEVICES(tree_record, params);
	/* ------------------------------------- OPENCL -------------------------------------- */

	free(tree_record->XtrainI);
	free(tree_record->nodes);
	free(tree_record->leaves);
	free(tree_record->Itrain_sorted);
	free(tree_record->Xtrain_sorted);

	free(params->kernels_source_directory);

}

/* --------------------------------------------------------------------------------
 * Frees opencl test buffers
 * --------------------------------------------------------------------------------
 */
void extern_free_query_buffers(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	free_train_buffers_gpu(tree_record, params);
	free_query_buffers_gpu(tree_record, params);

}

/* --------------------------------------------------------------------------------
 * Get maximum number of test queries that can be processed.
 * --------------------------------------------------------------------------------
 */
long get_max_nXtest_extern(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	long device_mem_bytes = tree_record->device_infos.device_mem_bytes;

	// w.r.t. to total memory consumption (see types.h)
	double total_test_mem = sizeof(FLOAT_TYPE) * (2 * tree_record->dXtrain + 2 * params->n_neighbors);
	total_test_mem += sizeof(INT_TYPE) * (2 * params->n_neighbors + params->tree_depth + 2);
	total_test_mem += sizeof(INT_TYPE) * 3;
	long n_max_total = (long) ((device_mem_bytes * params->allowed_test_mem_percent - 5E6 * sizeof(INT_TYPE)) / total_test_mem);

	// per single buffer
	long device_max_alloc = tree_record->device_infos.device_max_alloc_bytes;
	long n_max_buffer = device_max_alloc / (MAX(tree_record->dXtrain, params->n_neighbors) * sizeof(FLOAT_TYPE));
	n_max_buffer = (long) (0.33 * (double)n_max_buffer);

	return MIN(n_max_total, n_max_buffer);

}

/* --------------------------------------------------------------------------------
 * Checks if platform and device are valid
 * --------------------------------------------------------------------------------
 */
int extern_check_platform_device(int platform_id, int device_id){

	return get_device_infos(platform_id, device_id, NULL);

}
