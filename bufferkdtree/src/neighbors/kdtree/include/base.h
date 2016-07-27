/*
 * base.h
 *
 *  Created on: 31.10.2014
 *      Author: Fabian Gieseke
 */

#ifndef NEIGHBORS_KDTREE_INCLUDE_BASE_H_
#define NEIGHBORS_KDTREE_INCLUDE_BASE_H_

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

#include "global.h"
#include "util.h"
#include "kdtree.h"

/* --------------------------------------------------------------------------------
 * Interface (extern): Initialize components
 * --------------------------------------------------------------------------------
 */
void init_extern(int n_neighbors, int tree_depth, int num_threads, int splitting_type,
		int verbosity_level, KD_TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Fits the nearest neighbor model (build kd-tree)
 * --------------------------------------------------------------------------------
 */
void fit_extern(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain,
		KD_TREE_RECORD *kdtree_record, KD_TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Computes nearest neighbors
 * --------------------------------------------------------------------------------
 */
void neighbors_extern(FLOAT_TYPE * Xtest, int nXtest, int dXtest,
		FLOAT_TYPE * distances, int ndistances, int ddistances,
		int *indices, int nindices, int dindices,
		KD_TREE_RECORD *kdtree_record,
		KD_TREE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Frees resources
 * --------------------------------------------------------------------------------
 */
void free_resources_extern(void);

#endif /* NEIGHBORS_KDTREE_INCLUDE_BASE_H_ */
