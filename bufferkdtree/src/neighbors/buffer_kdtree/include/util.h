/* 
 * util.h
 */

#ifndef NEIGHBORS_BUFFER_KD_TREE_UTIL_H_
#define NEIGHBORS_BUFFER_KD_TREE_UTIL_H_

#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <malloc.h>
#include <errno.h>
#include <ctype.h>

#include "types.h"

/* --------------------------------------------------------------------------------
 * Sets default parameters.
 * --------------------------------------------------------------------------------
 */
void set_default_parameters(TREE_PARAMETERS *params);

void check_parameters(TREE_PARAMETERS *params);

/* -------------------------------------------------------------------------------- 
 * Computes the distances between a training pattern (train_patt) and several test
 * pattern (test_patterns). The results are inserted in the two lists d_min and 
 * idx_min.
 * -------------------------------------------------------------------------------- 
 */
void dist_insert_batch(FLOAT_TYPE * train_patt, INT_TYPE train_idx,
		FLOAT_TYPE * test_patterns, INT_TYPE ntest_patterns,
		FLOAT_TYPE * d_min, INT_TYPE *idx_min, INT_TYPE dim, UINT_TYPE K);

/* -------------------------------------------------------------------------------- 
 * Computes the distance between point a and b in R^dim
 * -------------------------------------------------------------------------------- 
 */
FLOAT_TYPE kd_tree_dist(FLOAT_TYPE * a, FLOAT_TYPE * b, INT_TYPE dim);

/* -------------------------------------------------------------------------------- 
 * Helper function.
 * -------------------------------------------------------------------------------- 
 */
FLOAT_TYPE squared(FLOAT_TYPE a);

/* -------------------------------------------------------------------------------- 
 * Inserts the value pattern_dist with index pattern_idx in the list nearest_dist
 * of FLOAT_TYPEs and the list nearest_idx of indices. Both lists contain at most K
 * elements.
 * -------------------------------------------------------------------------------- 
 */
void kd_tree_insert(FLOAT_TYPE pattern_dist, INT_TYPE pattern_idx, FLOAT_TYPE * nearest_dist,
		INT_TYPE *nearest_idx, UINT_TYPE K);

/* -------------------------------------------------------------------------------- 
 * The circular buffer implementation is partly based on code taken
 * from Wikipedia (http://en.wikipedia.org/wiki/Circular_buffer).
 * -------------------------------------------------------------------------------- 
 */


void cb_init(circular_buffer * cb, INT_TYPE size);
void cb_free(circular_buffer * cb);
INT_TYPE cb_is_full(circular_buffer * cb);
void cb_add_elt(circular_buffer * cb, INT_TYPE *item);
INT_TYPE cb_is_empty(circular_buffer * cb);
void cb_write(circular_buffer * cb, INT_TYPE *item);
void cb_read(circular_buffer * cb, INT_TYPE *item);
INT_TYPE cb_get_number_items(circular_buffer * cb);
void cb_read_batch(circular_buffer * cb, INT_TYPE *items_array,
		INT_TYPE num_elts_to_remove);
void cb_read_batch_fast(circular_buffer * cb, INT_TYPE *items_array,
		INT_TYPE num_elts_to_remove);

circular_buffer *cb_double_size(circular_buffer *cb);

/* -------------------------------------------------------------------------------- 
 * Reads a line from an input file.
 * -------------------------------------------------------------------------------- 
 */
char* readline(FILE *input);

/* -------------------------------------------------------------------------------- 
 * Reads patterns and labels from an input file
 * -------------------------------------------------------------------------------- 
 */
void read_patterns(const char *ifile, FLOAT_TYPE **patterns,
		FLOAT_TYPE **labels, INT_TYPE *num, INT_TYPE *dim);

double get_raw_train_mem_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

double get_train_mem_with_chunks_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

double get_test_tmp_mem_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

double get_total_mem_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

void partition_array_via_pivot(void *array, INT_TYPE count, INT_TYPE axis, INT_TYPE size_per_elt, FLOAT_TYPE pivot_value);

/* --------------------------------------------------------------------------------
 * Swaps to elements
 * --------------------------------------------------------------------------------
 */
inline void swap_elements(void *p1, void *p2, int size_elt);

/* --------------------------------------------------------------------------------
 * Copies an element
 * --------------------------------------------------------------------------------
 */
inline void copy_element(void *dest, const void *src, INT_TYPE size_elt);

/* --------------------------------------------------------------------------------
 * Returns the number of bytes needed by the largest single training buffer
 * --------------------------------------------------------------------------------
 */
double get_train_max_buffer_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Returns the number of bytes needed by the largest single test buffer
 * --------------------------------------------------------------------------------
 */
double get_test_max_buffer_device_bytes(TREE_RECORD *tree_record, TREE_PARAMETERS *params);


#endif /* NEIGHBORS_BUFFER_KD_TREE_UTIL_H_ */

