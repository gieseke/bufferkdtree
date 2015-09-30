"""
Nearest Neighbors
=================

This example demonstrates the use of both tree-based
implementations on a large-scale data set.
"""
print(__doc__)

# Authors: Fabian Gieseke 
# Licence: GNU GPL (v2)

import time
import generate
from bufferkdtree.neighbors import NearestNeighbors

# parameters
plat_dev_ids = {0:[0,1,2,3]}
n_jobs = 8
verbose = 0
n_neighbors=10

def run_algorithm(algorithm="buffer_kd_tree", tree_depth=None, leaf_size=None):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, \
                            algorithm=algorithm, \
                            tree_depth=tree_depth, \
                            leaf_size=leaf_size, \
                            n_jobs = n_jobs, \
                            plat_dev_ids=plat_dev_ids, \
                            verbose=verbose)

    start_time = time.time()
    nbrs.fit(Xtrain)
    end_time = time.time()
    print("Fitting time: %f" % (end_time-start_time))

    start_time = time.time()
    dists, inds = nbrs.kneighbors(Xtest)
    end_time = time.time()
    print("Testing time: %f" % (end_time-start_time))

# get/download data
Xtrain, Ytrain, Xtest = generate.get_data_set(NUM_TRAIN=2000000, NUM_TEST=10000000)
print "-------------------------------- DATA --------------------------------"
print "Number of training patterns:\t", Xtrain.shape[0]
print "Number of test patterns:\t", Xtest.shape[0]
print "Dimensionality of patterns:\t", Xtrain.shape[1]
print "----------------------------------------------------------------------"

print("\n\nRunning the GPU version ...")
run_algorithm(algorithm="buffer_kd_tree", tree_depth=9)

print("\n\nRunning the CPU version ...")
run_algorithm(algorithm="kd_tree", leaf_size=32)

