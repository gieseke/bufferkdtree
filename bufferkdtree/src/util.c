/*
 * util.c
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 *               1976 Niklaus Wirth
 * License: GPL v2
 *
 * The median search (see below) is based on:
 *
 * Author: Wirth, Niklaus
 * Title: Algorithms + data structures = programs
 * Publisher: Englewood Cliffs: Prentice-Hall, 1976
 * Physical description: 366 p.
 * Series: Prentice-Hall Series in Automatic Computation
 *
 */
#include "include/util.h"
#include "include/float.h"

/**
 * Transposes a given float array
 *
 * @param array Pointer to float array
 * @param n number of "rows" in a
 * @param d number of "columns" in a
 * @param array_transposed The transposed array
 *
 */
void transpose_array_float(float* array,
		int n,
		int d,
		float* array_transposed){
	int i, j;
	for (j = 0; j < d; j++) {
		for (i = 0; i < n; i++) {
			array_transposed[j * n + i] = array[i * d + j];
		}
	}
}

/**
 * Transposes a given double array
 *
 * @param array Pointer to double array
 * @param n number of "rows" in a
 * @param d number of "columns" in a
 * @param array_transposed The transposed array
 *
 */
void transpose_array_double(double* array,
		int n,
		int d,
		double* array_transposed){
	int i, j;
	for (j = 0; j < d; j++) {
		for (i = 0; i < n; i++) {
			array_transposed[j * n + i] = array[i * d + j];
		}
	}
}

/**
 * Transposes a given int array
 *
 * @param array Pointer to int array
 * @param n number of "rows" in a
 * @param d number of "columns" in a
 * @param array_transposed The transposed array
 *
 */
void transpose_array_int(int* array,
		int n,
		int d,
		int* array_transposed){
	int i, j;
	for (j = 0; j < d; j++) {
		for (i = 0; i < n; i++) {
			array_transposed[j * n + i] = array[i * d + j];
		}
	}
}


/**
 * Compares two float values
 *
 * @param Pointer to first float
 * @param Pointer to second float
 */
int compare_floats(const void *p1,
		const void *p2) {

	// the index is stored at the end of each element...
	FLOAT_TYPE *p1_point, *p2_point;
	p1_point = (FLOAT_TYPE *) p1;
	p2_point = (FLOAT_TYPE *) p2;
	if (*p1_point < *p2_point){
		return -1;
	}
	if (*p1_point > *p2_point){
		return +1;
	}
	return 0;

}

/**
 * Compares to integers
 *
 *
 * @param Pointer to first int
 * @param Pointer to second int
 */
int compare_ints(const void *p1,
		const void *p2) {

	// the index is stored at the end of each element...
	int *p1_point, *p2_point;
	p1_point = (int *) p1;
	p2_point = (int *) p2;
	if (*p1_point < *p2_point){
		return -1;
	}
	if (*p1_point > *p2_point){
		return +1;
	}
	return 0;

}

/**
 Function :   kth_smallest()
 In       :   array of elements, # of elements in the array, rank k
 Out      :   one element
 Job      :   find the kth smallest element in the array
 Notice   :   use the median() macro defined below to get the median.

 Reference:

 Author: Wirth, Niklaus
 Title: Algorithms + data structures = programs
 Publisher: Englewood Cliffs: Prentice-Hall, 1976
 Physical description: 366 p.
 Series: Prentice-Hall Series in Automatic Computation
*/
elem_type kth_smallest(elem_type a[],
		int n,
		int k) {
	return a[kth_smallest_idx(a, n, k)];
}

/**
 * Computes the kth smallest index
 *
 * @param a[] Array of type elem_type
 * @param n Number of rows
 * @param Number of columns
 */
int kth_smallest_idx(elem_type a[],
		int n,
		int k) {
	register unsigned int i, j, l, m;
	register elem_type x;

	l = 0;
	m = n - 1;
	while (l < m) {
		x = a[k];
		i = l;
		j = m;
		do {
			while (a[i] < x)
				i++;
			while (x < a[j])
				j--;
			if (i <= j) {
				ELEM_SWAP(a[i], a[j]);
				i++;
				j--;
			}
		} while (i <= j);
		if (j < k)
			l = i;
		if (k < i)
			m = j;
	}
	return k;
}
