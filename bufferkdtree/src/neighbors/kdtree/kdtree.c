/*
 * kdtree.c
 *
 *  Created on: 31.10.2014
 *      Author: Fabian Gieseke
 */

#include "include/kdtree.h"

/* --------------------------------------------------------------------------------
 * Initializes space for the kd tree (e.g., nodes and leaves)
 * --------------------------------------------------------------------------------
 */
void kd_tree_init_tree_record(KD_TREE_RECORD *record, int kd_tree_depth,
		FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain) {

	record->Xtrain = Xtrain;
	record->nXtrain = nXtrain;
	record->dXtrain = dXtrain;

	record->XI = (void *) malloc(nXtrain * (sizeof(FLOAT_TYPE) * dXtrain + sizeof(int)));
	record->nodes = (KD_TREE_NODE *) malloc((pow(2, kd_tree_depth) - 1) * sizeof(KD_TREE_NODE));
	record->leaves = (int *) malloc(2 * pow(2, kd_tree_depth) * sizeof(int));
	record->tree_depth = kd_tree_depth;

}

/* --------------------------------------------------------------------------------
 * Frees space for the kd tree (e.g., nodes and leaves)
 * --------------------------------------------------------------------------------
 */
void kd_tree_free_tree_record(KD_TREE_RECORD *record) {

	free(record->XI);
	free(record->nodes);
	free(record->leaves);
	free(record);

}

/* --------------------------------------------------------------------------------
 * Builds the kd-tree (recursive construction)
 * --------------------------------------------------------------------------------
 */
void kd_tree_build_tree(KD_TREE_RECORD *kdtree_record, KD_TREE_PARAMETERS *params) {

	kd_tree_build_recursive(kdtree_record, params, 0, kdtree_record->nXtrain, 0, 0);

}

/* --------------------------------------------------------------------------------
 * Helper method to build up the kd-tree in a recursive manner.
 * --------------------------------------------------------------------------------
 */
void kd_tree_build_recursive(KD_TREE_RECORD *kdtree_record, KD_TREE_PARAMETERS *params,
		int left, int right, int idx, int depth) {

	assert(left < right);

	// if maximum tree depth is reached
	if (depth == params->tree_depth) {

		int leave_idx = idx - (pow(2, params->tree_depth) - 1);
		kdtree_record->leaves[leave_idx * 2] = left;
		kdtree_record->leaves[leave_idx * 2 + 1] = right;
		return;

	}

	int axis = kd_tree_find_best_split(depth, left, right, kdtree_record, params);
	int pivot_idx = kd_tree_split_training_patterns_via_pivot(kdtree_record->XI, left, right, axis, kdtree_record->dXtrain);

	// determine splitting axis and value
	FLOAT_TYPE *median_elt = (FLOAT_TYPE *) (kdtree_record->XI + pivot_idx * (kdtree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(int)));
	kdtree_record->nodes[idx].splitting_value = median_elt[axis];
	kdtree_record->nodes[idx].axis = axis;

	// recurse
	kd_tree_build_recursive(kdtree_record, params, left, pivot_idx, 2 * idx + 1, depth + 1);
	kd_tree_build_recursive(kdtree_record, params, pivot_idx, right, 2 * idx + 2, depth + 1);

}

/* --------------------------------------------------------------------------------
 * Queries the kd-tree to obtain the nearest neigbhors, sequential version.
 * --------------------------------------------------------------------------------
 */
void kd_tree_query_tree_sequential(FLOAT_TYPE *test_pattern, FLOAT_TYPE *d_min,
		int *idx_min, int K, KD_TREE_RECORD *record) {

	int i;

	// initialize list of nearest neighbors
	for (i = 0; i < K; i++) {
		idx_min[i] = 0;
		d_min[i] = MAX_FLOAT_TYPE;
	}

	// local variables
	void *XI = record->XI;
	int kd_tree_depth = record->tree_depth;
	int *leaves = record->leaves;
	KD_TREE_NODE *nodes = record->nodes;
	int dim = record->dXtrain;

	// initialize stack with 0 values
	int *stack = (int *) malloc(kd_tree_depth * sizeof(int));
	for (i = 0; i < kd_tree_depth; i++) {
		stack[i] = 0;
	}

	int idx = 0;                // the current index
	int depth = 0;              // the current depth
	int axis = 0;               // the current axis
	int status = -1;            // the current (stack) status
	int leaf_width = 2;         // the width of each leaf (in leaves)

	// until root has not been visited twice ...
	while (1) {

		// if leaf is reached
		if (depth == kd_tree_depth) {

			int leaf_idx = idx - (pow(2, kd_tree_depth) - 1);
			int fr_idx = leaves[leaf_idx * leaf_width];
			int to_idx = leaves[leaf_idx * leaf_width + 1];
			kd_tree_brute_force_leaf(XI, dim, fr_idx, to_idx, test_pattern, d_min, idx_min, K);

			// go up again
			idx = (idx - 1) / 2;
			depth--;
			if (depth < 0){break;}

		} else {

			status = stack[depth];
			axis = nodes[idx].axis;

			if (status == 0) {

				// if first visit
				if (test_pattern[axis] - nodes[idx].splitting_value >= 0) {
					// go right (down)
					stack[depth] = 1;
					idx = 2 * idx + 2;
					depth++;
				} else {
					// go left (down)
					stack[depth] = 2;
					idx = 2 * idx + 1;
					depth++;
				}

			} else if (status == 2) {

				// if left child was visited first, we have to check the right side of the node
				if (squared(test_pattern[axis] - nodes[idx].splitting_value)
						<= d_min[K - 1]) {
					// the test instance is nearer to the median than to the nearest neighbors
					// found so far: go right!
					idx = 2 * idx + 2;
					stack[depth] = 3;
					depth++;
				} else { // if the median is "far away", we can go up (without checking the right side)
					idx = (idx - 1) / 2;
					stack[depth] = 0;
					depth--;
					if (depth < 0)
						break;
				}

			} else if (status == 1) {

				// if right child was visited first, we have to check the left side of the node
				if (squared(test_pattern[axis] - nodes[idx].splitting_value)
						<= d_min[K - 1]) {
					// the test instance is nearer to the median than to the nearest neighbors
					// found so far: go left!
					idx = 2 * idx + 1;
					stack[depth] = 3;
					depth++;
				} else { // if the median is "far away", we can go up (without checking the left side)
					idx = (idx - 1) / 2;
					stack[depth] = 0;
					depth--;
					if (depth < 0)
						break;
				}

			} else { // status == 3

				// both children have been visited: go up!
				idx = (idx - 1) / 2;
				stack[depth] = 0;
				depth--;
				if (depth < 0) {break;}

			}
		}
	}

	free(stack);

}

/* --------------------------------------------------------------------------------
 * Brute-force nearest neigbhor search in a leaf of the tree (determined by fr_idx,
 * to_idx, and XI).
 * --------------------------------------------------------------------------------
 */
void kd_tree_brute_force_leaf(void *XI, int dim, int fr_idx, int to_idx,
		FLOAT_TYPE *test_pattern, FLOAT_TYPE *d_min, int *idx_min, int K) {

	int i;

	for (i = fr_idx; i < to_idx; i++) {
		FLOAT_TYPE *patt = (FLOAT_TYPE *) (XI + i * (dim * sizeof(FLOAT_TYPE) + sizeof(int)));
		FLOAT_TYPE d = kd_tree_dist(patt, test_pattern, dim);
		kd_tree_insert(d, i, d_min, idx_min, K);
	}

}

/* --------------------------------------------------------------------------------
 * Finds the splitting axis.
 * --------------------------------------------------------------------------------
 */
int kd_tree_find_best_split(int depth, int left, int right, KD_TREE_RECORD *kdtree_record, KD_TREE_PARAMETERS *params){

	if (params->splitting_type == SPLITTING_TYPE_CYCLIC){

		return depth % kdtree_record->dXtrain;

	} else if (params->splitting_type == SPLITTING_TYPE_LONGEST_BOX){

		int i, j;
		int dim = kdtree_record->dXtrain;

		// intialize empty box
		FLOAT_TYPE *mins = (FLOAT_TYPE*) malloc(kdtree_record->dXtrain * sizeof(FLOAT_TYPE));
		FLOAT_TYPE *maxs = (FLOAT_TYPE*) malloc(kdtree_record->dXtrain * sizeof(FLOAT_TYPE));
		for (j=0; j<dim; j++){
			mins[j] =  MAX_FLOAT_TYPE;
			maxs[j] = -MAX_FLOAT_TYPE;
		}

		// compute bounding box
		int elt_size = kdtree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(int);
		for (i=left; i<right; i++){
			FLOAT_TYPE *patt = (FLOAT_TYPE*) (kdtree_record->XI + i*elt_size);
			for (j=0; j < dim; j++){
				if (patt[j] < mins[j]){ mins[j] = patt[j]; }
				if (patt[j] > maxs[j]){ maxs[j] = patt[j]; }
			}
		}

		// find longest axis
		FLOAT_TYPE longest = 0.0;
		int longest_axis = 0;

		for (j=0; j<dim; j++){
			FLOAT_TYPE width = maxs[j] - mins[j];
			if (width <= 0.0){
				printf("width <= 0.01\n");
				exit(EXIT_FAILURE);
			}
			if (width > longest){
				longest = width;
				longest_axis = j;
			}
		}

		free(mins);
		free(maxs);

		return longest_axis;

	} else {

		printf("Error: Unknown splitting type!\n");
		exit(EXIT_FAILURE);

	}

}

/* --------------------------------------------------------------------------------
 * Parse patterns and store the original indices (this array of both the FLOAT_TYPEs
 * and the indices) is sorted in-place during the construction of the kd-tree.
 * --------------------------------------------------------------------------------
 */
void kd_tree_generate_training_patterns_indices(KD_TREE_RECORD *kdtree_record) {

	int i, j;

	for (i = 0; i < kdtree_record->nXtrain; i++) {

		FLOAT_TYPE *pattern = (FLOAT_TYPE *) (kdtree_record->XI + i * (kdtree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(int)));
		for (j = 0; j < kdtree_record->dXtrain; j++) {
			pattern[j] = kdtree_record->Xtrain[i * kdtree_record->dXtrain + j];
		}

		// get pointer and set value
		int *index = (int *) (kdtree_record->XI + i * (kdtree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(int)) + kdtree_record->dXtrain * sizeof(FLOAT_TYPE));
		*index = i;

	}

}

/* --------------------------------------------------------------------------------
 * Sorts the training patterns in range left to right (inclusive) with respect to
 * "axis".
 * --------------------------------------------------------------------------------
 */
int kd_tree_split_training_patterns_via_pivot(void *XI, int left, int right,
		int axis, int dim) {

	int i;

	int count = right - left;
	int size_per_elt = dim * sizeof(FLOAT_TYPE) + sizeof(int);

	void *array = (void *) (XI + left * size_per_elt);

	FLOAT_TYPE *tmp = (FLOAT_TYPE*) malloc(count * sizeof(FLOAT_TYPE));
	for (i = 0; i < count; i++) {
		// pattern is stored first, so we get all values of all
		// patterns here w.r.t. to dimension 'axis'
		tmp[i] = ((FLOAT_TYPE*) (array + i * size_per_elt))[axis];
	}
	FLOAT_TYPE pivot_value = kth_smallest(tmp, count, count / 2);
	free(tmp);

	partition_array_via_pivot(array, count, axis, size_per_elt, pivot_value);

	return left + count / 2;

}

/* --------------------------------------------------------------------------------
 * Inserts the value pattern_dist with index pattern_idx in the list nearest_dist
 * of FLOAT_TYPEs and the list nearest_idx of indices. Both lists contain at most K
 * elements.
 * --------------------------------------------------------------------------------
 */
void kd_tree_insert(FLOAT_TYPE pattern_dist, int pattern_idx,
		FLOAT_TYPE *nearest_dist, int *nearest_idx, int K) {

	int j = K - 1;
	if (nearest_dist[j] <= pattern_dist)
		return; //rightmost is smaller

	nearest_dist[j] = pattern_dist;
	nearest_idx[j] = pattern_idx;
	FLOAT_TYPE tmp_d = 0.0;
	int tmp_idx = 0;

	// This is usually very fast since most candidates are not closer
	// (thus, we are exiting here directly, better than logarithmic search)
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
			break;
	}
}
