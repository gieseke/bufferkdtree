"""
Nearest Neighbors
=================

This example demonstrates the use of the different 
implementations given on a small artifical data set.
"""
print(__doc__)

# Authors: Fabian Gieseke 
# Licence: GNU GPL (v2)

import numpy
from bufferkdtree.neighbors import NearestNeighbors

n_neighbors = 10
plat_dev_ids = {0:[0]}
n_jobs = 1
verbose = 0

X = numpy.random.uniform(low=-1, high=1, size=(10000,10))

# (1) apply buffer k-d tree implementation
nbrs_buffer_kd_tree = NearestNeighbors(algorithm="buffer_kd_tree", \
                        tree_depth=9, \
                        plat_dev_ids=plat_dev_ids, \
                        verbose=verbose)    
nbrs_buffer_kd_tree.fit(X)
dists, inds = nbrs_buffer_kd_tree.kneighbors(X, n_neighbors=n_neighbors)
print("\nbuffer_kd_tree output\n" + unicode(dists[0]))

# (2) apply brute-force implementation
nbrs_brute = NearestNeighbors(algorithm="brute", \
                        plat_dev_ids=plat_dev_ids, \
                        verbose=verbose)    
nbrs_brute.fit(X)
dists, inds = nbrs_brute.kneighbors(X, n_neighbors=n_neighbors)
print("\nbrute output\n" + unicode(dists[0]))

# (3) apply k-d tree mplementation
nbrs_kd_tree = NearestNeighbors(algorithm="kd_tree", \
                        n_jobs=n_jobs, \
                        verbose=verbose)    
nbrs_kd_tree.fit(X)
dists, inds = nbrs_kd_tree.kneighbors(X, n_neighbors=n_neighbors)
print("\nkd_tree output\n" + unicode(dists[0]))
print("")
