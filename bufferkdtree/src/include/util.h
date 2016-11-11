/*
 * util.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 *               1976 Niklaus Wirth
 * License: GPL v2
 *
 */
#ifndef INCLUDE_UTIL_H_
#define INCLUDE_UTIL_H_

#include "float.h"

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
#include <errno.h>
#include <string.h>
#include <ctype.h>

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
		float* array_transposed);

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
		double* array_transposed);

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
		int* array_transposed);

/**
 * Compares two float values
 *
 * @param Pointer to first float
 * @param Pointer to second float
 */
int compare_floats(const void *p1,
		const void *p2);

/**
 * Compares to integers
 *
 *
 * @param Pointer to first int
 * @param Pointer to second int
 */
int compare_ints(const void *p1, 
		const void *p2);

typedef FLOAT_TYPE elem_type;

#define ELEM_SWAP(a,b) { register elem_type t=(a);(a)=(b);(b)=t; }
#define median(a,n) kth_smallest(a,n,((n)/2))

/*---------------------------------------------------------------------------
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

 ---------------------------------------------------------------------------*/
elem_type kth_smallest(elem_type a[], int n, int k);

/**
 * Computes the kth smallest index
 *
 * @param a[] Array of type elem_type
 * @param n Number of rows
 * @param Number of columns
 */
int kth_smallest_idx(elem_type a[],
		int n,
		int k);

#endif /* INCLUDE_UTIL_H_ */
