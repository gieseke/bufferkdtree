#
# Copyright (C) 2013-2019 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

"""
Benchmark Nearest Neighbors
===========================

This benchmark compares the performance of the different
implementations on a large-scale data set.

Note: The platform and the device needs to be specified below 
(via the parameter 'plat_dev_ids').
"""
print(__doc__)

import json
import time
import generate
from bufferkdtree.neighbors import NearestNeighbors, compute_optimal_tree_depth

# NOTE: If the 'brute' implementation exits with error code -5 (OUT_OF_RESOURCES),
# then the "watchdog timer" of the driver might have killed to scheme due to
# the kernel taking too long time; an easy fix on Linux systems is to start the
# code without X (e.g., go to tty 1 via Ctrl+Alt+F1 and start the code from here)

# platform dependent parameters
# (might have to be adapted!)
plat_dev_ids = {0:[0, 1]}
n_jobs = 8

# parameters
ofilename = "benchmark.json"
n_test_range = [1000000, 2500000, 5000000, 7500000, 10000000]
algorithms = ["brute", "kd_tree", "buffer_kd_tree"]

verbose = 0
n_neighbors = 10

print("Parsing data ...")
Xtrain, Ytrain, Xtest = generate.get_data_set(NUM_TRAIN=2000000, NUM_TEST=10000000)
print("-------------------------------- DATA --------------------------------")
print("Number of training patterns:\t %i" % Xtrain.shape[0])
print("Number of test patterns:\t %i" % Xtest.shape[0])
print("Dimensionality of patterns:\t%i" % Xtrain.shape[1])
print("----------------------------------------------------------------------")

def compute_opt_tree_depth(algorithm, n_test_tree=2000000):

    opt_tree_depth = None

    if algorithm in ["buffer_kd_tree", "kd_tree"]:

        # the different tree depths that shall
        # be tested for this data set
        if algorithm == "buffer_kd_tree":
            tree_depths = range(4, 12)
        elif algorithm == "kd_tree":
            tree_depths = range(8, 16)

        # search for optimal tree depth
        model = NearestNeighbors(n_neighbors=n_neighbors, 
                                 algorithm=algorithm, 
                                 n_jobs=n_jobs, 
                                 plat_dev_ids=plat_dev_ids, 
                                 verbose=verbose)
        opt_tree_depth = compute_optimal_tree_depth(model, 
                                                    Xtrain, 
                                                    Xtest[:n_test_tree], 
                                                    target="test", 
                                                    tree_depths=tree_depths)
        print("Optimal tree depth found: %i " % opt_tree_depth)
        
    return opt_tree_depth

def run_algorithm(n_test, tree_depth=None, algorithm="buffer_kd_tree"):

    print("----------------------------------------------------------------------")
    print("\n\nRunning %s for n_test=%i ...\n" % (algorithm, n_test))
    print("----------------------------------------------------------------------")

    Xtest_local = Xtest[:n_test, :]

    # instantiate model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, \
                            algorithm=algorithm, \
                            n_jobs=n_jobs, \
                            tree_depth=opt_tree_depth, \
                            plat_dev_ids=plat_dev_ids, \
                            verbose=verbose)
                
    # train model
    start_time = time.time()
    nbrs.fit(Xtrain)
    end_time = time.time()
    train_time = (end_time - start_time)
    print("Fitting time: %f" % train_time)

    # apply model (testing phase)
    start_time = time.time()
    _, _ = nbrs.kneighbors(Xtest_local)
    end_time = time.time()
    test_time = (end_time - start_time)
    print("Testing time: %f" % test_time)
    
    return train_time, test_time

results = {}

# run all algorithms and all n_tests
for i in range(len(algorithms)):
    
    algorithm = algorithms[i]
    results[algorithm] = {}

    print("----------------------------------------------------------------------")
    print("\n\nRunning %s ...\n" % (algorithm))
    print("----------------------------------------------------------------------")

    print("Searching for optimal tree depth. This may " + 
          "take a while for the k-d tree based schemes ...")
    opt_tree_depth = compute_opt_tree_depth(algorithm, n_test_tree=2000000)

    for n_test in n_test_range:
        train_time, test_time = run_algorithm(n_test,
                                              tree_depth=opt_tree_depth,
                                              algorithm=algorithm
                                              )
        results[algorithm][n_test] = {'train':train_time,
                                      'test':test_time,
                                      'opt_tree_depth':opt_tree_depth
                                      }
    print("\n\n")

# write results after each step
print("Writing results to %s ..." % ofilename)
with open(ofilename, 'w') as f:
    json.dump(results, f)
