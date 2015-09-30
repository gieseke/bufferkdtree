/*
 * global.h
 *
 *  Created on: 31.10.2014
 *      Author: Fabian Gieseke
 */

#ifndef NEIGHBORS_KDTREE_INCLUDE_GLOBAL_H_
#define NEIGHBORS_KDTREE_INCLUDE_GLOBAL_H_

#include "../../../include/float.h"

#define SPLITTING_TYPE_CYCLIC 0
#define SPLITTING_TYPE_LONGEST_BOX 1

// struct for input parameters
typedef struct kd_tree_parameters {

	int n_neighbors;
	int tree_depth;
	int num_threads;
	int verbosity_level;
	int splitting_type;

} KD_TREE_PARAMETERS;

// struct for storing a single node
typedef struct kdtree_node {

	int axis;
	FLOAT_TYPE splitting_value;

} KD_TREE_NODE;

// struct for storing tree
typedef struct kdtree_record {

	FLOAT_TYPE *Xtrain;
	int nXtrain;
	int dXtrain;

	void *XI;
	KD_TREE_NODE *nodes;
	int *leaves;
	int tree_depth;

} KD_TREE_RECORD;




#endif /* NEIGHBORS_KDTREE_INCLUDE_GLOBAL_H_ */
