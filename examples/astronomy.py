"""
Nearest Neighbors in Astronomy
==============================

This example demonstrates the use of both k-d tree-based
implementations on a large-scale astronomical data set that is 
based on the Sloan Digitial Sky Survey (http://www.sdss.org). 
The data set contains 2,000,000 training points and 10,000,000
test points in a 10-dimensional feature space.

Note: The platform and the device needs to be specified below 
(via the parameter 'plat_dev_ids').
"""
print(__doc__)

# Authors: Fabian Gieseke
# Licence: GNU GPL (v2)

import time
import generate
from bufferkdtree import NearestNeighbors

# parameters
plat_dev_ids = {0:[0,1,2,3]}
n_jobs = 8
verbose = 1
float_type = "float" 
n_neighbors = 10

def run_algorithm(algorithm="buffer_kd_tree", tree_depth=None, leaf_size=None):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, \
                            algorithm=algorithm, \
                            tree_depth=tree_depth, \
                            leaf_size=leaf_size, \
                            float_type=float_type, \
                            n_jobs=n_jobs, \
                            plat_dev_ids=plat_dev_ids, \
                            verbose=verbose)

    start_time = time.time()
    nbrs.fit(Xtrain)
    end_time = time.time()
    print("Fitting time: %f" % (end_time - start_time))

    start_time = time.time()
    dists, inds = nbrs.kneighbors(Xtest)
    end_time = time.time()
    print("Testing time: %f" % (end_time - start_time))

print("Parsing data ...")
Xtrain, Ytrain, Xtest = generate.get_data_set(NUM_TRAIN=2000000, NUM_TEST=10000000)
print("-------------------------------- DATA --------------------------------")
print("Number of training patterns:\t %i" % Xtrain.shape[0])
print("Number of test patterns:\t %i" % Xtest.shape[0])
print("Dimensionality of patterns:\t%i" % Xtrain.shape[1])
print("----------------------------------------------------------------------")

print("\n\nRunning the GPU version ...")
run_algorithm(algorithm="buffer_kd_tree", tree_depth=9)

print("\n\nRunning the CPU version ...")
run_algorithm(algorithm="kd_tree", leaf_size=32)

