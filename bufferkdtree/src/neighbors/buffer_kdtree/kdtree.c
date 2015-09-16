/* 
 * kdtree.c
 */

#include "include/kdtree.h"

/* -------------------------------------------------------------------------------- 
 * Builds the kd-tree (recursive construction)
 * -------------------------------------------------------------------------------- 
 */
void kd_tree_build_tree(TREE_RECORD *tree_record, TREE_PARAMETERS *params) {

	kd_tree_build_recursive(tree_record, params, 0, tree_record->nXtrain, 0, 0);
}

/* --------------------------------------------------------------------------------
 * Finds the splitting axis.
 * --------------------------------------------------------------------------------
 */
void kd_tree_find_best_split(int depth, int left, int right,
		TREE_RECORD *tree_record, TREE_PARAMETERS *params,
		int *axis, int *pivot_idx, FLOAT_TYPE *splitting_value){

	if (params->splitting_type == SPLITTING_TYPE_CYCLIC){

		*axis = depth % tree_record->dXtrain;
		*pivot_idx = kd_tree_split_training_patterns_via_pivot(tree_record->XtrainI, left, right, *axis, tree_record->dXtrain);
		FLOAT_TYPE *median_elt = (FLOAT_TYPE *) (tree_record->XtrainI + *pivot_idx * (tree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(INT_TYPE)));
		*splitting_value = median_elt[*axis];

	} else if (params->splitting_type == SPLITTING_TYPE_LONGEST_BOX){

		int i, j;
		int dim = tree_record->dXtrain;

		// initialize empty box
		FLOAT_TYPE *mins = (FLOAT_TYPE*) malloc(tree_record->dXtrain * sizeof(FLOAT_TYPE));
		FLOAT_TYPE *maxs = (FLOAT_TYPE*) malloc(tree_record->dXtrain * sizeof(FLOAT_TYPE));
		for (j=0; j<dim; j++){
			mins[j] = MAX_FLOAT_TYPE;
			maxs[j] = -MAX_FLOAT_TYPE;
		}

		// compute bounding box
		int elt_size = tree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(int);
		for (i=left; i<right; i++){
			FLOAT_TYPE *patt = (FLOAT_TYPE*) (tree_record->XtrainI + i*elt_size);
			for (j=0; j < dim; j++){
				if (patt[j] < mins[j]){ mins[j] = patt[j]; }
				if (patt[j] > maxs[j]){ maxs[j] = patt[j]; }
			}
		}

		// find longest axis
		FLOAT_TYPE longest = 0.0;
		int longest_axis = -1;

		for (j=0; j<dim; j++){
			FLOAT_TYPE width = maxs[j] - mins[j];
			if (width > longest){
				longest = width;
				longest_axis = j;
			}
		}

		*axis = longest_axis;

		*pivot_idx = kd_tree_split_training_patterns_via_pivot(tree_record->XtrainI, left, right, *axis, tree_record->dXtrain);
		FLOAT_TYPE *median_elt = (FLOAT_TYPE *) (tree_record->XtrainI + *pivot_idx * (tree_record->dXtrain * sizeof(FLOAT_TYPE) + sizeof(INT_TYPE)));
		*splitting_value = median_elt[*axis];



		free(mins);
		free(maxs);



	} else {
		printf("Error: Unknown splitting type!\n");
		exit(EXIT_FAILURE);
	}


}


/* -------------------------------------------------------------------------------- 
 * Helper method to build up the kd-tree in a recursive manner.
 * -------------------------------------------------------------------------------- 
 */
void kd_tree_build_recursive(TREE_RECORD *tree_record, TREE_PARAMETERS *params,
		INT_TYPE left, INT_TYPE right, INT_TYPE idx, INT_TYPE depth) {

	// if tree depth is reached
	if (depth == params->tree_depth) {
		INT_TYPE width = 2;          // fr and to
		INT_TYPE leave_idx = idx - (pow(2, params->tree_depth) - 1);
		tree_record->leaves[leave_idx * width] = left;
		tree_record->leaves[leave_idx * width + 1] = right;
		return;
	}

	INT_TYPE axis, pivot_idx;
	FLOAT_TYPE splitting_value;

	kd_tree_find_best_split(depth, left, right, tree_record, params, &axis, &pivot_idx, &splitting_value);

	tree_record->nodes[idx].splitting_value = splitting_value;
	tree_record->nodes[idx].axis = axis;

	kd_tree_build_recursive(tree_record, params, left, pivot_idx, 2 * idx + 1, depth + 1);
	kd_tree_build_recursive(tree_record, params, pivot_idx, right, 2 * idx + 2, depth + 1);

}

/* -------------------------------------------------------------------------------- 
 * Parse patterns and store the original indices (this array of both the FLOAT_TYPEs
 * and the indices) is sorted in-place during the construction of the kd-tree.
 * -------------------------------------------------------------------------------- 
 */
void kd_tree_generate_training_patterns_indices(void *XI, FLOAT_TYPE * X, INT_TYPE n,
		INT_TYPE dim) {
	INT_TYPE i;
	INT_TYPE j;
	for (i = 0; i < n; i++) {
		FLOAT_TYPE *pattern = (FLOAT_TYPE *) (XI
				+ i * (dim * sizeof(FLOAT_TYPE) + sizeof(INT_TYPE)));
		for (j = 0; j < dim; j++) {
			pattern[j] = X[i * dim + j];
		}
		INT_TYPE *index = (INT_TYPE *) (XI + i * (dim * sizeof(FLOAT_TYPE) + sizeof(INT_TYPE))
				+ dim * sizeof(FLOAT_TYPE));
		*index = i;
	}
}

/* --------------------------------------------------------------------------------
 * Sorts the training patterns in range left to right (inclusive) with respect to
 * "axis".
 * --------------------------------------------------------------------------------
 */
INT_TYPE kd_tree_split_training_patterns_via_pivot(void *XI, INT_TYPE left, INT_TYPE right,
		INT_TYPE axis, INT_TYPE dim) {

	INT_TYPE i;
	INT_TYPE count = right - left;

	// starting address
	void *array = (void *) XI;

	// size per element (in bytes)
	INT_TYPE size_per_elt = dim * sizeof(FLOAT_TYPE) + sizeof(INT_TYPE);
	array += left * size_per_elt;

	// find pivot value
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


