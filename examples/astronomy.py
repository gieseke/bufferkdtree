'''
Created on 15.09.2015

@author: fgieseke
'''

import os
import time
import numpy
import generate
from bufferkdtree.neighbors.base import NearestNeighbors

# get data
Xtrain, Ytrain, Xtest = generate.get_data_set(NUM_TRAIN=20000, NUM_TEST=100000)
print "--------------------------------------- DATA -------------------------------------------------"
print "Number of training patterns:", Xtrain.shape[0]
print "Number of test patterns:", Xtest.shape[0]
print "Dimensionality of patterns:", Xtrain.shape[1]
print "----------------------------------------------------------------------------------------------"

def run_algorithm(n_neighbors=10, algorithm="buffer_kd_tree", tree_depth=None, leaf_size=32, n_jobs=1, verbose=1):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, \
                            algorithm=algorithm, \
                            tree_depth=tree_depth, \
                            leaf_size=leaf_size, \
                            n_jobs = n_jobs, \
                            plat_dev_ids={0:[0,1,2,3]}, \
                            verbose=verbose)    

    start_time = time.time()
    nbrs.fit(Xtrain)
    end_time = time.time()
    print("Fitting time: %f\n" % (end_time-start_time))

    start_time = time.time()
    dists, inds = nbrs.kneighbors(Xtest)
    end_time = time.time()

    print("Testing time: %f\n" % (end_time-start_time))
    print dists[:2]
    print inds[:2]
    print "---------------------------------------------------------------------------------------------\n"

run_algorithm(algorithm="kd_tree", leaf_size=32, n_jobs=8)
run_algorithm(algorithm="buffer_kd_tree", tree_depth=9)
