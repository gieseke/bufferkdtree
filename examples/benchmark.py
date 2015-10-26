"""
Benchmark Nearest Neighbors
===========================

This benchmark compares the performance of the different
implementations on a large-scale data set.
"""
print(__doc__)

# Authors: Fabian Gieseke 
# Licence: GNU GPL (v2)

import json
import time
import generate
from bufferkdtree.neighbors import NearestNeighbors

# NOTE: If the 'brute' implementation exits with error code -5 (OUT_OF_RESOURCES),
# then the "watchdog timer" of the driver might have killed to scheme due to
# the kernel taking too long time; an easy fix on Linux systems is to start the
# code without X (e.g., go to terminal 1 via ctrl+alt+F1 and start the code from here)

# platform dependent parameters
# (might have to be adapted!)
plat_dev_ids = {0:[0,1,2,3]}
n_jobs = 8

# parameters
ofilename = "results.json"
n_test_range = [1000000, 2500000, 5000000, 7500000, 10000000]
algorithms = ["brute", "kd_tree", "buffer_kd_tree"]
verbose = 1
n_neighbors = 10

results = {}

print("Parsing data ...")
Xtrain, Ytrain, Xtest = generate.get_data_set(NUM_TRAIN=2000000, NUM_TEST=10000000)
print("-------------------------------- DATA --------------------------------")
print("Number of training patterns:\t %i" % Xtrain.shape[0])
print("Number of test patterns:\t %i" % Xtest.shape[0])
print("Dimensionality of patterns:\t%i" % Xtrain.shape[1])
print("----------------------------------------------------------------------")

def run_algorithm(n_test, algorithm="buffer_kd_tree"):

    print("----------------------------------------------------------------------")
    print("\n\nRunning %s for n_test=%i ...\n" % (algorithm, n_test))
    print("----------------------------------------------------------------------")
    opt_tree_depth = None
    Xtest_local = Xtest[:n_test, :]

    if algorithm in ["buffer_kd_tree", "kd_tree"]:

        # the different tree depths that shall
        # be tested for this data set
        if algorithm == "buffer_kd_tree":
            tree_depths = range(3,11)
        elif algorithm == "kd_tree":
            tree_depths = range(8,16)

        # search for optimal tree depth
        nbrs_tree_test = NearestNeighbors(n_neighbors=n_neighbors, \
                                     algorithm=algorithm, \
                                     n_jobs=n_jobs, \
                                     plat_dev_ids=plat_dev_ids, \
                                     verbose=verbose)
        opt_tree_depth = nbrs_tree_test.compute_optimal_tree_depth(Xtrain, Xtest_local, \
                                                    target="test", tree_depths=tree_depths)
        print("Optimal tree depth found: %i " % opt_tree_depth)

        # instantiate final model
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, \
                                algorithm=algorithm, \
                                n_jobs=n_jobs, \
                                tree_depth=opt_tree_depth, \
                                plat_dev_ids=plat_dev_ids, \
                                verbose=verbose)

    elif algorithm == "brute":

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, \
                                algorithm=algorithm, \
                                n_jobs=n_jobs, \
                                plat_dev_ids=plat_dev_ids, \
                                verbose=verbose)

    else:

        raise Exception("Unkown algorithm ...")
        
    # train model
    start_time = time.time()
    nbrs.fit(Xtrain)
    end_time = time.time()
    train_time = (end_time-start_time)
    print("Fitting time: %f" % train_time)

    # apply model (testing phase)
    start_time = time.time()
    dists, inds = nbrs.kneighbors(Xtest_local)
    print "dists=", dists
    end_time = time.time()
    test_time = (end_time-start_time)
    print("Testing time: %f" % test_time)

    # store results 
    if algorithm not in results.keys():
        results[algorithm] = {}
    results[algorithm][n_test] = {'train':train_time, 'test':test_time, 'opt_tree_depth':opt_tree_depth}

# run all algorithms and all n_tests
for n_test in n_test_range:
    for algorithm in algorithms:
        run_algorithm(n_test, algorithm=algorithm)

        # write results after each step
        print("Writing results to %s ..." % ofilename)
        with open(ofilename, 'w') as f:
            json.dump(results, f)

