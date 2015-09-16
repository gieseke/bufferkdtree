/*
 * util.c
 */

#include "include/util.h"

/* --------------------------------------------------------------------------------
 * Sets default parameters.
 * --------------------------------------------------------------------------------
 */
void set_default_parameters(KD_TREE_PARAMETERS *params) {

	params->n_neighbors = 10;
	params->tree_depth = 0;
	params->num_threads = 1;
	params->verbosity_level = 1;

}

void check_parameters(KD_TREE_PARAMETERS *params) {

	if ((params->n_neighbors < 1) || (params->n_neighbors > 1000)) {
		printf("Error: k must be > 0 and <= 1000)\nExiting ...\n");
		exit(EXIT_FAILURE);
	}

	if ((params->tree_depth < 0) || (params->tree_depth > 1000)) {
		printf("Error: tree_depth must positive and <= 1000)\nExiting ...\n");
		exit(EXIT_FAILURE);
	}

	if ((params->num_threads < 1) || (params->tree_depth > 10000)) {
		printf("Error: num_threads must be > 0 and <= 10000)\nExiting ...\n");
		exit(EXIT_FAILURE);
	}

	if ((params->verbosity_level < 0) || (params->verbosity_level > 1)) {
		printf("Error: verbosity_level must either 0 or 1\nExiting ...\n");
		exit(EXIT_FAILURE);
	}

}

void partition_array_via_pivot(void *array, int count, int axis, int size_per_elt, FLOAT_TYPE pivot_value){

	int i;

	// partition array
	int low = 0;
	int high = count - 1;

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
 * Copies an element (used by kd_tree_split_training_patterns_via_pivot)
 * --------------------------------------------------------------------------------
 */
inline void copy_element(void *dest, const void *src, int size_elt) {
	memcpy(dest, src, size_elt);
}

/* --------------------------------------------------------------------------------
 * Helper function.
 * --------------------------------------------------------------------------------
 */
inline FLOAT_TYPE squared(FLOAT_TYPE a) {
	return a * a;
}

/* --------------------------------------------------------------------------------
 * Computes the distance between point a and b in R^dim
 * --------------------------------------------------------------------------------
 */
inline FLOAT_TYPE kd_tree_dist(const FLOAT_TYPE *a, const FLOAT_TYPE *b, int dim) {

	int i;

	FLOAT_TYPE d = 0.0;
	FLOAT_TYPE tmp;

	for (i = 0; i < dim; i++) {
		tmp = a[i] - b[i];
		d += tmp * tmp;
	}

	return d;
}
