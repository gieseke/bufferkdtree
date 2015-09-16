/* 
 * cpu.c
 */

#include "include/cpu.h"

/* -------------------------------------------------------------------------------- 
 * Initializes all arrays.
 * -------------------------------------------------------------------------------- 
 */
void init_arrays_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	INT_TYPE i;
	for (i = 0; i < tree_record->nXtest * params->n_neighbors; i++) {
		tree_record->dist_mins_global[i] = MAX_FLOAT_TYPE;
		tree_record->idx_mins_global[i] = 0;
	}

	// initialize all_stacks, all_depths, and all_idxs (used in find_leaf_idx)
	tree_record->all_stacks = (INT_TYPE *) calloc(tree_record->nXtest * params->tree_depth, sizeof(UINT_TYPE));
	tree_record->all_depths = (INT_TYPE *) calloc(tree_record->nXtest, sizeof(UINT_TYPE));
	tree_record->all_idxs = (INT_TYPE *) calloc(tree_record->nXtest, sizeof(UINT_TYPE));
}

/* -------------------------------------------------------------------------------- 
 * Brute-force nearest neigbhor search in a leaf of the tree (determined by fr_idx,
 * to_idx, and XI).
 * -------------------------------------------------------------------------------- 
 */
void brute_force_leaf_cpu(INT_TYPE fr_idx, INT_TYPE to_idx, FLOAT_TYPE * test_patterns,
		INT_TYPE ntest_patterns, FLOAT_TYPE * d_min, INT_TYPE *idx_min,
		TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	UINT_TYPE j;
	FLOAT_TYPE *train_patt;
	for (j = fr_idx; j < to_idx; j++) {
		// pointer to a location of consecutively stored FLOAT_TYPEs
		train_patt = tree_record->Xtrain_sorted + j * tree_record->dXtrain;
		dist_insert_batch(train_patt, j, test_patterns, ntest_patterns, d_min,
				idx_min, tree_record->dXtrain, params->n_neighbors);
	}

}

/* -------------------------------------------------------------------------------- 
 * Processes all buffers on the CPU.
 * -------------------------------------------------------------------------------- 
 */
void process_all_buffers_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	INT_TYPE no_input_patterns_left = (tree_record->current_test_index == tree_record->nXtest
			&& cb_is_empty(&(tree_record->queue_reinsert)));

	if (tree_record->buffer_full_warning || no_input_patterns_left) {
		START_MY_TIMER(tree_record->timers + 11);
		tree_record->empty_all_buffers_calls++;
		UINT_TYPE leaf_idx;
		UINT_TYPE i;

		// get the total number of indices removed in this round (needed for space allocation)
		INT_TYPE total_number_test_indices_removed = 0;
		for (leaf_idx = 0; leaf_idx < tree_record->n_leaves; leaf_idx++) {
			if (cb_get_number_items(tree_record->buffers[leaf_idx]) > 0) { // we can always empty ALL buffers, no overhead!
				total_number_test_indices_removed += cb_get_number_items(
						tree_record->buffers[leaf_idx]);
			}
		}

		// intialize arrays that need to be transferred to the GPU
		INT_TYPE *test_indices_removed_from_all_buffers = (INT_TYPE *) malloc(
				total_number_test_indices_removed * sizeof(INT_TYPE));
		INT_TYPE *fr_indices = (INT_TYPE *) malloc(
				total_number_test_indices_removed * sizeof(INT_TYPE));
		INT_TYPE *to_indices = (INT_TYPE *) malloc(
				total_number_test_indices_removed * sizeof(INT_TYPE));
		INT_TYPE number_added_test_elements = 0;
		INT_TYPE number_t_indices_in_buffer;
		for (leaf_idx = 0; leaf_idx < tree_record->n_leaves; leaf_idx++) {
			if (cb_get_number_items(tree_record->buffers[leaf_idx]) > 0) {
				// get number of indices that are in the buffer
				number_t_indices_in_buffer = cb_get_number_items(
						tree_record->buffers[leaf_idx]);
				// read indices from buffer
				cb_read_batch(tree_record->buffers[leaf_idx],
						test_indices_removed_from_all_buffers
								+ number_added_test_elements,
						number_t_indices_in_buffer);
				// we generate copies of data items that are needed by ALL KERNELS (to avoid bank conflicts)
				// Thus, we are transferring more elements as needed to the GPU
				for (i = 0; i < number_t_indices_in_buffer; i++) {
					fr_indices[number_added_test_elements + i] = tree_record->leaves[leaf_idx
							* LEAF_WIDTH]; // fr_idx
					to_indices[number_added_test_elements + i] = tree_record->leaves[leaf_idx
							* LEAF_WIDTH + 1];     // to_idx
					// reinsert test pattern into test queue: can already be done here, we have to wait at the end once...
					cb_add_elt(&(tree_record->queue_reinsert),
							test_indices_removed_from_all_buffers
									+ number_added_test_elements + i);
				}
				// increase number of added elements (needed as offset)
				number_added_test_elements += number_t_indices_in_buffer;

			}
		}

		// all buffers are empty now
		tree_record->buffer_full_warning = 0;
		STOP_MY_TIMER(tree_record->timers + 11);
		START_MY_TIMER(tree_record->timers + 12);
		do_bruteforce_all_leaves_cpu(test_indices_removed_from_all_buffers,
				total_number_test_indices_removed, fr_indices, to_indices,
				tree_record, params);
		STOP_MY_TIMER(tree_record->timers + 12);
		// free memory
		free(test_indices_removed_from_all_buffers);
		free(fr_indices);
		free(to_indices);
		// all buffers are empty now: let's check if enough work is still there for another round!
		INT_TYPE num_elts_in_queue = cb_get_number_items(&(tree_record->queue_reinsert));
		if (tree_record->current_test_index == tree_record->nXtest
				&& num_elts_in_queue < params->bf_remaining_threshold) {
			START_MY_TIMER(tree_record->timers + 12);
			process_queue_via_brute_force_cpu(tree_record, params);
			STOP_MY_TIMER(tree_record->timers + 12);
		}
	}
}

/* -------------------------------------------------------------------------------- 
 * Performs a brute-force in the leaves.
 * -------------------------------------------------------------------------------- 
 */
void do_bruteforce_all_leaves_cpu(INT_TYPE *test_indices_removed_from_all_buffers,
		INT_TYPE total_number_test_indices_removed, INT_TYPE *fr_indices, INT_TYPE *to_indices,
		TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	UINT_TYPE i;
	UINT_TYPE train_idx;
	FLOAT_TYPE *train_patt;

#pragma omp parallel for
	for (i = 0; i < total_number_test_indices_removed; i++) {
		UINT_TYPE j;
		FLOAT_TYPE d;
		INT_TYPE test_idx = test_indices_removed_from_all_buffers[i];
		INT_TYPE fr_idx = fr_indices[i];
		INT_TYPE to_idx = to_indices[i];
		FLOAT_TYPE tmp;
		INT_TYPE tmp_idx;
		// initialize local copies
		FLOAT_TYPE *dist_mins_local = (FLOAT_TYPE *) malloc(
				params->n_neighbors * sizeof(FLOAT_TYPE));
		for (j = params->n_neighbors; j--;) {
			dist_mins_local[j] = MAX_FLOAT_TYPE;
		}
		INT_TYPE *idx_mins_local = (INT_TYPE *) calloc(params->n_neighbors, sizeof(INT_TYPE));
		// 1) iterate over all training patterns and compute distances/indices
		for (train_idx = fr_idx; train_idx < to_idx; train_idx++) {
			train_patt = tree_record->Xtrain_sorted + train_idx * tree_record->dXtrain;
			// compute distance between training and test pattern
			d = 0.0;
			for (j = tree_record->dXtrain; j--;) {
				tmp = (train_patt[j] - tree_record->Xtest[test_idx * tree_record->dXtrain + j]);
				d += tmp * tmp;
			}
			// insert dist/idx in local arrays
			j = params->n_neighbors - 1;
			if (dist_mins_local[j] > d) {
				//tmp=0.0;
				//tmp_idx=0;
				dist_mins_local[j] = d;
				idx_mins_local[j] = train_idx;
				for (; j > 0; j--) {
					if (dist_mins_local[j] < dist_mins_local[j - 1]) {
						//swap dist
						tmp = dist_mins_local[j];
						dist_mins_local[j] = dist_mins_local[j - 1];
						dist_mins_local[j - 1] = tmp;
						//swap idx
						tmp_idx = idx_mins_local[j];
						idx_mins_local[j] = idx_mins_local[j - 1];
						idx_mins_local[j - 1] = tmp_idx;
					} else
						break;
				}
			}
		}
		// 2) merge local copies into global array (GPU: barrier?)
		FLOAT_TYPE *new_dists = (FLOAT_TYPE *) malloc(
				params->n_neighbors * sizeof(FLOAT_TYPE));
		INT_TYPE *new_idx = (INT_TYPE *) malloc(params->n_neighbors * sizeof(INT_TYPE));
		INT_TYPE counter_global = 0;
		INT_TYPE counter_local = 0;
		for (j = 0; j < params->n_neighbors; j++) {
			if (dist_mins_local[counter_local]
					< tree_record->dist_mins_global[test_idx * params->n_neighbors + counter_global]) {
				new_dists[j] = dist_mins_local[counter_local];
				new_idx[j] = idx_mins_local[counter_local];
				counter_local++;
			} else {
				new_dists[j] =
						tree_record->dist_mins_global[test_idx * params->n_neighbors + counter_global];
				new_idx[j] = tree_record->idx_mins_global[test_idx * params->n_neighbors + counter_global];
				counter_global++;
			}
		}
		// write content backup to global memory (can also be done in a coalesced manner if necessary)
		memcpy(tree_record->dist_mins_global + test_idx * params->n_neighbors, new_dists,
				params->n_neighbors * sizeof(FLOAT_TYPE));
		memcpy(tree_record->idx_mins_global + test_idx * params->n_neighbors, new_idx, params->n_neighbors * sizeof(INT_TYPE));

		// free memory
		free(dist_mins_local);
		free(idx_mins_local);
		free(new_dists);
		free(new_idx);
	}

}

/* -------------------------------------------------------------------------------- 
 * If only few elements are left in the queues after the buffers have been emptied, 
 * we do a simple brute force step to compute the nearest neighbors.
 * -------------------------------------------------------------------------------- 
 */
void process_queue_via_brute_force_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {
	START_MY_TIMER(tree_record->timers + 11);
	PRINT(params)("Doing brute force for the remaining patterns in the tree...\n");
	UINT_TYPE i;
	UINT_TYPE j;
	INT_TYPE num_elts_in_queue = cb_get_number_items(&(tree_record->queue_reinsert));
	INT_TYPE *indices_removed_from_queue = (INT_TYPE *) malloc(
			num_elts_in_queue * sizeof(INT_TYPE));
	cb_read_batch(&(tree_record->queue_reinsert), indices_removed_from_queue,
			num_elts_in_queue);
	// create local copy of test patterns
	FLOAT_TYPE *test_patterns_subset = (FLOAT_TYPE *) malloc(
			num_elts_in_queue * tree_record->dXtrain * sizeof(FLOAT_TYPE));
	for (i = 0; i < num_elts_in_queue; i++) {
		INT_TYPE t_idx = indices_removed_from_queue[i];
		for (j = 0; j < tree_record->dXtrain; j++) {
			test_patterns_subset[i * tree_record->dXtrain + j] = tree_record->Xtest[t_idx * tree_record->dXtrain + j];
		}
	}
	// do brute-force nn search on ALL leaves
	INT_TYPE fr_idx = 0;
	INT_TYPE to_idx = tree_record->nXtrain;
	FLOAT_TYPE *dist_mins_local = (FLOAT_TYPE *) malloc(
			num_elts_in_queue * params->n_neighbors * sizeof(FLOAT_TYPE));
	INT_TYPE *idx_mins_local = (INT_TYPE *) calloc(num_elts_in_queue * params->n_neighbors, sizeof(INT_TYPE));

	for (i = num_elts_in_queue * params->n_neighbors; i--;) {
		dist_mins_local[i] = MAX_FLOAT_TYPE;
	}
	STOP_MY_TIMER(tree_record->timers + 11);
	brute_force_leaf_cpu(fr_idx, to_idx, test_patterns_subset,
			num_elts_in_queue, dist_mins_local, idx_mins_local, tree_record, params);
	for (i = num_elts_in_queue; i--;) {
		// fast: we just have to copy everything to the global arrays (no merging needed here!)
		INT_TYPE t_idx = indices_removed_from_queue[i];
		memcpy(tree_record->dist_mins_global + t_idx * params->n_neighbors, dist_mins_local + i * params->n_neighbors,
				params->n_neighbors * sizeof(FLOAT_TYPE));
		memcpy(tree_record->idx_mins_global + t_idx * params->n_neighbors, idx_mins_local + i * params->n_neighbors,
				params->n_neighbors * sizeof(INT_TYPE));
	}
	free(dist_mins_local);
	free(idx_mins_local);
	free(indices_removed_from_queue);
	free(test_patterns_subset);
}

/* -------------------------------------------------------------------------------- 
 * Finds the next leaf indices for all test patterns indixed by all_next_indices.
 * -------------------------------------------------------------------------------- 
 */
void find_leaf_idx_batch_cpu(INT_TYPE *all_next_indices, INT_TYPE num_all_next_indices,
		INT_TYPE *ret_vals, TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	START_MY_TIMER(tree_record->timers + 16);
	tree_record->find_leaf_idx_calls++;
	INT_TYPE j;

	for (j = 0; j < num_all_next_indices; j++) {
		INT_TYPE test_idx = all_next_indices[j];
		INT_TYPE *depth = tree_record->all_depths + test_idx;
		INT_TYPE *idx = tree_record->all_idxs + test_idx;
		INT_TYPE *all_stacks_local = tree_record->all_stacks + test_idx * params->tree_depth;
		static INT_TYPE axis = 0;
		static INT_TYPE status = 0;
		INT_TYPE test_idx_dim = test_idx * tree_record->dXtrain;
		FLOAT_TYPE dist_mins_global_test_idx_KNN_KNN_M1 = tree_record->dist_mins_global[test_idx * params->n_neighbors + params->n_neighbors - 1];
		while (1) {
			if (*depth == params->tree_depth) {
				INT_TYPE leaf_idx = *idx - tree_record->n_nodes;
				*idx = (*idx - 1) * 0.5;
				*depth = *depth - 1;
				ret_vals[j] = leaf_idx;
				break;
			}
			status = all_stacks_local[*depth];

			//axis = *depth % tree_record->dXtrain;
			axis = tree_record->nodes[*idx].axis;

			if (status == 0) {
				// if first visit
				if (tree_record->Xtest[test_idx_dim + axis] - tree_record->nodes[*idx].splitting_value >= 0) {
					// go right (down)
					all_stacks_local[*depth] = 1;
					*idx = 2 * (*idx) + 2;
					*depth = *depth + 1;
				} else {
					// go left (down)
					all_stacks_local[*depth] = 2;
					*idx = 2 * (*idx) + 1;
					*depth = *depth + 1;
				}
			} else if (status == 2) {
				// if left child was visited first, we have to check the right side of the node
				FLOAT_TYPE tmp = tree_record->Xtest[test_idx_dim + axis] - tree_record->nodes[*idx].splitting_value;
				if (tmp * tmp <= dist_mins_global_test_idx_KNN_KNN_M1) {
					// the test instance is nearer to the median than to the nearest neighbors
					// found so far: go right!
					*idx = 2 * (*idx) + 2;
					all_stacks_local[*depth] = 3;
					*depth = *depth + 1;
				} else { // if the median is "far away", we can go up (without checking the right side)
					*idx = (*idx - 1) * 0.5;
					all_stacks_local[*depth] = 0;
					*depth = *depth - 1;
					if (*depth < 0) {
						ret_vals[j] = -1;
						break;
					}
				}
			} else if (status == 1) {
				FLOAT_TYPE tmp = tree_record->Xtest[test_idx_dim + axis]
						- tree_record->nodes[*idx].splitting_value;
				// if right child was visited first, we have to check the left side of the node
				if (tmp * tmp <= dist_mins_global_test_idx_KNN_KNN_M1) {
					// the test instance is nearer to the median than to the nearest neighbors
					// found so far: go left!
					*idx = 2 * (*idx) + 1;
					all_stacks_local[*depth] = 3;
					*depth = *depth + 1;
				} else { // if the median is "far away", we can go up (without checking the left side)
					*idx = (*idx - 1) * 0.5;
					all_stacks_local[*depth] = 0;
					*depth = *depth - 1;
					if (*depth < 0) {
						ret_vals[j] = -1;
						break;
					}
				}
			} else {    // status == 3
				// both children have been visited...time to go up!
				*idx = (*idx - 1) * 0.5;
				all_stacks_local[*depth] = 0;
				*depth = *depth - 1;
				if (*depth < 0) {
					ret_vals[j] = -1;
					break;
				}
			}
		}
	}
	STOP_MY_TIMER(tree_record->timers + 16);

}

/* -------------------------------------------------------------------------------- 
 * Copies the arrays dist_min_global and idx_min_global from GPU to CPU
 * Updates the distances and indices (w.r.t the original indices)
 * -------------------------------------------------------------------------------- 
 */
void get_distances_and_indices_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	INT_TYPE i, j;

	for (i = 0; i < tree_record->nXtest; i++) {
		for (j = 0; j < params->n_neighbors; j++) {
			tree_record->idx_mins_global[i * params->n_neighbors + j] =
					tree_record->Itrain_sorted[tree_record->idx_mins_global[i * params->n_neighbors + j]];
			// take roots of the squared distances
			tree_record->dist_mins_global[i * params->n_neighbors + j] = sqrt(
					tree_record->dist_mins_global[i * params->n_neighbors + j]);
		}
	}

}

/* -------------------------------------------------------------------------------- 
 * Writes the training patterns in a specific ordering.
 * -------------------------------------------------------------------------------- 
 */
void write_sorted_training_patterns_cpu(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {
	INT_TYPE i;
	for (i = 0; i < tree_record->nXtrain; i++) {
		memcpy(tree_record->Xtrain_sorted + i * tree_record->dXtrain,
				tree_record->XtrainI + i * (tree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(INT_TYPE)),
				tree_record->dXtrain * sizeof(FLOAT_TYPE));
		tree_record->Itrain_sorted[i] = *((INT_TYPE *) (tree_record->XtrainI + tree_record->dXtrain * sizeof(FLOAT_TYPE)
				+ i * (tree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(INT_TYPE))));
	}
}
