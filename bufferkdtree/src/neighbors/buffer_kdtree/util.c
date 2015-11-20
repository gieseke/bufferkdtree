/* 
 * util.c
 */

#include "include/util.h"

/* --------------------------------------------------------------------------------
 * Sets default parameters.
 * --------------------------------------------------------------------------------
 */
void set_default_parameters(TREE_PARAMETERS *params) {

	// default parameter values
	params->n_neighbors = 10;
	params->tree_depth = 8;
	params->num_threads = 1;
	params->verbosity_level = 1;
	params->bf_remaining_threshold = 8192;
	params->n_train_chunks = 1;
	params->allowed_test_mem_percent = 0.8;
	params->allowed_train_mem_percent_chunk = 0.2;

}

void check_parameters(TREE_PARAMETERS *params) {

	// parameter checks
	if ((params->n_neighbors < 1) || (params->n_neighbors > 100)) {
		printf("Error: The parameter k must be > 0 and <= 100)\nExiting ...\n");
		exit(1);
	}

	// parameter checks
	if ((params->tree_depth < 2) || (params->tree_depth > 50)) {
		printf("Error: The parameter tree_depth must be > 1 and <= 50)\nExiting ...\n");
		exit(1);
	}

	// parameter checks
	if ((params->num_threads < 1) || (params->tree_depth > 10000)) {
		printf("Error: The parameter num_threads must be > 0 and <= 10000)\nExiting ...\n");
		exit(1);
	}

	// parameter checks
	if ((params->allowed_test_mem_percent < 0.0) || (params->allowed_train_mem_percent_chunk < 0.0)) {
		printf("Error: The parameters allowed_test_mem_percent and allowed_train_mem_percent_chunk must be non-negative!\n");
		exit(1);
	}
	if ((params->allowed_test_mem_percent + params->allowed_train_mem_percent_chunk > 1.00001)) {
		printf("Error: allowed_test_mem_percent + allowed_train_mem_percent_chunk must be less or equal to 1.0 ...\n");
		exit(1);
	}

}

/* -------------------------------------------------------------------------------- 
 * Computes the distances between a training pattern (train_patt) and several test
 * pattern (test_patterns). The results are inserted in the two lists d_min and 
 * idx_min.
 * -------------------------------------------------------------------------------- 
 */
void dist_insert_batch(FLOAT_TYPE * train_patt, INT_TYPE train_idx,
		FLOAT_TYPE * test_patterns, INT_TYPE ntest_patterns,
		FLOAT_TYPE * d_min, INT_TYPE *idx_min, INT_TYPE dim, UINT_TYPE K) {
	register UINT_TYPE t_idx;
	register UINT_TYPE i;
	register INT_TYPE j;
	register FLOAT_TYPE d;
	register FLOAT_TYPE tmp;
	register UINT_TYPE tmp_idx;
	register FLOAT_TYPE *nearest_dist;
	register INT_TYPE *nearest_idx;
	for (t_idx = ntest_patterns; t_idx--;) {
		d = 0.0;
		for (i = dim; i--;) {
			tmp = (train_patt[i] - test_patterns[t_idx * dim + i]);
			d += tmp * tmp;
		}
		nearest_dist = d_min + K * t_idx;
		nearest_idx = idx_min + K * t_idx;
		j = K - 1;
		if (nearest_dist[j] > d) {
			nearest_dist[j] = d;
			nearest_idx[j] = train_idx;
			for (; j > 0; j--) {
				if (nearest_dist[j] < nearest_dist[j - 1]) {
					//swap dist
					tmp = nearest_dist[j];
					nearest_dist[j] = nearest_dist[j - 1];
					nearest_dist[j - 1] = tmp;
					//swap idx
					tmp_idx = nearest_idx[j];
					nearest_idx[j] = nearest_idx[j - 1];
					nearest_idx[j - 1] = tmp_idx;
				} else
					break;
			}
		}

	}
}

/* -------------------------------------------------------------------------------- 
 * Computes the distance between point a and b in R^dim
 * -------------------------------------------------------------------------------- 
 */
FLOAT_TYPE kd_tree_dist(FLOAT_TYPE * a, FLOAT_TYPE * b, INT_TYPE dim) {
	register UINT_TYPE i;
	FLOAT_TYPE d = 0.0;
	for (i = dim; i--;) {
		d += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return d;
}

/* -------------------------------------------------------------------------------- 
 * Helper function.
 * -------------------------------------------------------------------------------- 
 */
FLOAT_TYPE squared(FLOAT_TYPE a) {
	return a * a;
}

/* -------------------------------------------------------------------------------- 
 * Inserts the value pattern_dist with index pattern_idx in the list nearest_dist
 * of FLOAT_TYPEs and the list nearest_idx of indices. Both lists contain at most K
 * elements.
 * -------------------------------------------------------------------------------- 
 */
void kd_tree_insert(FLOAT_TYPE pattern_dist, INT_TYPE pattern_idx, FLOAT_TYPE * nearest_dist,
		INT_TYPE *nearest_idx, UINT_TYPE K) {
	register UINT_TYPE j = K - 1;
	if (nearest_dist[j] <= pattern_dist)
		return; //rightmost is smaller
	FLOAT_TYPE tmp_d = 0.0;
	UINT_TYPE tmp_idx = 0;
	nearest_dist[j] = pattern_dist;
	nearest_idx[j] = pattern_idx;
	for (; j > 0; j--) {
		if (nearest_dist[j] < nearest_dist[j - 1]) {
			//swap dist
			tmp_d = nearest_dist[j];
			nearest_dist[j] = nearest_dist[j - 1];
			nearest_dist[j - 1] = tmp_d;
			//swap idx
			tmp_idx = nearest_idx[j];
			nearest_idx[j] = nearest_idx[j - 1];
			nearest_idx[j - 1] = tmp_idx;
		} else
			return;
	}
}

/* -------------------------------------------------------------------------------- 
 * The circular buffer implementation is partly based on code taken
 * from Wikipedia (http://en.wikipedia.org/wiki/Circular_buffer).
 * -------------------------------------------------------------------------------- 
 */

void cb_init(circular_buffer * cb, INT_TYPE size) {
	cb->size = size + 1;
	cb->start = 0;
	cb->end = 0;
	cb->items = (INT_TYPE *) calloc(cb->size, sizeof(INT_TYPE));
}

INT_TYPE cb_get_number_items(circular_buffer * cb) {
	if (cb->end >= cb->start) {
		return cb->end - cb->start;
	} else {
		return cb->size + cb->end - cb->start;
	}
}

void cb_read_batch(circular_buffer * cb, INT_TYPE *items_array,
		INT_TYPE num_elts_to_remove) {
	UINT_TYPE i;
	for (i=0; i< num_elts_to_remove; i++) {
		items_array[i] = cb->items[cb->start];
		cb->start = (cb->start + 1) % cb->size;
	}
}

//// BUG= not yet used
//void cb_read_batch_fast(circular_buffer * cb, INT_TYPE *items_array,
//		INT_TYPE num_elts_to_remove) {
//
//	INT_TYPE end = (cb->start + num_elts_to_remove) % cb->size;
//
//	if (end >= cb->start){
//		memcpy(items_array, cb->items, end - cb->start);
//
//	} else {
//		memcpy(items_array, cb->items + cb->start, cb->size - cb->start);
//		memcpy(items_array, cb->items, end);
//	}
//	cb->start = end;
//
//}

void cb_free(circular_buffer * cb) {
	free(cb->items);
}

INT_TYPE cb_is_full(circular_buffer * cb) {
	return (cb->end + 1) % cb->size == cb->start;
}

INT_TYPE cb_is_empty(circular_buffer * cb) {
	return cb->end == cb->start;
}

void cb_add_elt(circular_buffer * cb, INT_TYPE *item) {
	cb_write(cb, item);
}

void cb_write(circular_buffer * cb, INT_TYPE *item) {
	cb->items[cb->end] = *item;
	cb->end = (cb->end + 1) % cb->size;
	if (cb->end == cb->start)
		cb->start = (cb->start + 1) % cb->size;
}

void cb_read(circular_buffer * cb, INT_TYPE *item) {
	*item = cb->items[cb->start];
	cb->start = (cb->start + 1) % cb->size;
}

circular_buffer *cb_double_size(circular_buffer *cb){

	circular_buffer *new_cb = (circular_buffer *) malloc(sizeof(circular_buffer));
	cb_init(new_cb, 2 * cb->size);

	INT_TYPE num_elts = cb_get_number_items(cb);
	//INT_TYPE *items_array = (INT_TYPE *) malloc(num_elts);
	//cb_read_batch_fast(cb, items_array, num_elts);

	int i;
	int elt;
	for (i=0; i<num_elts; i++){
		cb_read(cb, &elt);
		cb_add_elt(new_cb, &elt);
	}

	cb_free(cb);

	return new_cb;

}


double get_train_mem_with_chunks_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params){

	// see definition of tree_record in types.h
	double mem = 0;
	double ntrain = (double) (tree_record->nXtrain / params->n_train_chunks);

	// train
	mem += 2 * ntrain * tree_record->dXtrain * sizeof(FLOAT_TYPE);
	mem += tree_record->n_nodes * sizeof(FLOAT_TYPE);
	mem += tree_record->n_leaves * 2 * sizeof(FLOAT_TYPE);

	return mem;

}

double get_raw_train_mem_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params){

	// see definition of tree_record in types.h
	double mem = 0;

	// train
	mem += 2 * tree_record->nXtrain * tree_record->dXtrain * sizeof(FLOAT_TYPE);
	mem += tree_record->n_nodes * sizeof(FLOAT_TYPE);
	mem += tree_record->n_leaves * 2 * sizeof(FLOAT_TYPE);

	return mem;

}

double get_test_tmp_mem_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params){

	// see definition of tree_record in types.h
	double mem = 0;

	// test
	mem += tree_record->nXtest * sizeof(FLOAT_TYPE) * (2 * tree_record->dXtrain + 2 * params->n_neighbors);
	mem += tree_record->nXtest * sizeof(INT_TYPE) * (2 * params->n_neighbors + params->tree_depth + 2);

	// tmp
	mem += tree_record->nXtest * sizeof(INT_TYPE) * 3;
	mem += 5E6 * sizeof(INT_TYPE);

	return mem;
}


/* --------------------------------------------------------------------------------
 * Returns the number of bytes needed by the largest single training buffer
 * --------------------------------------------------------------------------------
 */
double get_train_max_buffer_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params){

	double ntrain = (double) (tree_record->nXtrain / params->n_train_chunks);
	return ntrain * tree_record->dXtrain * sizeof(FLOAT_TYPE);

}

/* --------------------------------------------------------------------------------
 * Returns the number of bytes needed by the largest single test buffer
 * --------------------------------------------------------------------------------
 */
double get_test_max_buffer_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params){

	return tree_record->nXtest * MAX(tree_record->dXtrain, params->n_neighbors) * sizeof(FLOAT_TYPE);

}

double get_total_mem_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params){

	double mem_train = get_train_mem_with_chunks_device_bytes(tree_record, params);
	double mem_test_tmp = get_test_tmp_mem_device_bytes(tree_record, params);

	return mem_train + mem_test_tmp;

}

void partition_array_via_pivot(void *array, INT_TYPE count, INT_TYPE axis, INT_TYPE size_per_elt, FLOAT_TYPE pivot_value){

	INT_TYPE i;

	// partition array
	INT_TYPE low = 0;
	INT_TYPE high = count - 1;

	void *array_new = (void*) malloc(count * size_per_elt);

	for (i = 0; i < count; i++) {

		FLOAT_TYPE val = ((FLOAT_TYPE*) (array + i * size_per_elt))[axis];

		if (val < pivot_value) {

			copy_element(array_new + low * size_per_elt,
					array + i * size_per_elt, size_per_elt);
			low++;

		} else {

			copy_element(array_new + high * size_per_elt,
					array + i * size_per_elt, size_per_elt);
			high--;

		}
	}

	high++;
	for (i=high; i<count; i++){
		FLOAT_TYPE val = ((FLOAT_TYPE*) (array_new + i * size_per_elt))[axis];
		if (val == pivot_value){
			// move pivot to the correct position
			swap_elements(array_new + i * size_per_elt,
					array_new + high * size_per_elt,
					size_per_elt);
			high++;
		}
	}

	// copy array
	memcpy(array, array_new, count * size_per_elt);
	free(array_new);

}

/* --------------------------------------------------------------------------------
 * Swaps two elements (used by kd_tree_split_training_patterns_via_pivot)
 * --------------------------------------------------------------------------------
 */
inline void swap_elements(void *p1, void *p2, int size_elt) {

	void *tmp_elt = (void*) malloc(size_elt);
	memcpy(tmp_elt, p1, size_elt);
	memcpy(p1, p2, size_elt);
	memcpy(p2, tmp_elt, size_elt);
	free(tmp_elt);

}

/* --------------------------------------------------------------------------------
 * Copies an element
 * --------------------------------------------------------------------------------
 */
inline void copy_element(void *dest, const void *src, INT_TYPE size_elt) {
	memcpy(dest, src, size_elt);
}
