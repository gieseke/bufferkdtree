'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

import numpy
import random
from bufferkdtree.neighbors.base import NearestNeighbors

def neighbors_artificial(ntrain=4096, ntest=4096, dim=32, seed=0):

    random.seed(seed)

    Xtrain = numpy.empty(ntrain*dim, dtype=numpy.float64).reshape((ntrain,dim))
    Xtest = numpy.empty(ntest*dim, dtype=numpy.float64).reshape((ntest,dim))

    for i in xrange(ntrain):
        for j in xrange(dim):
            Xtrain[i][j] = random.random()
    for i in xrange(ntest):
        for j in xrange(dim):
            Xtest[i][j] = random.random()

    return Xtrain, Xtest

n_neighbors = 5
use_gpu = True
verbose = 0
n_jobs = 4
float_type = "float"
algorithms = ["brute", "buffer_kd_tree", "kd_tree"]
leaf_size = 32
tree_depth = 10
plat_dev_ids = {0:[0]}

data_sets = []
for ntrain in [5000,2000]:
    for ntest in [1000,2000]:
        for dim in [5,10,15]:
            data_sets.append(["ntrain=%i, ntest=%i, dim=%i" % (ntrain, ntest, dim), neighbors_artificial(ntrain=ntrain, ntest=ntest, dim=dim)])

def run(Xtrain, Xtest, algorithm):

    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, float_type=float_type, \
                                    use_gpu=use_gpu, n_jobs=n_jobs, tree_depth=tree_depth, plat_dev_ids=plat_dev_ids, verbose=verbose)

    print("\t\tFitting ...")
    neigh.fit(Xtrain)

    print("\t\tQuerying ...")
    dists, inds = neigh.kneighbors(Xtest, n_neighbors)

    return dists, inds

def test():

    print("Testing nearest neighbors ...")

    for data_set in data_sets:

        print("Checking data set " + data_set[0])
        Xtrain, Xtest = data_set[1][0], data_set[1][1]

        dists_ref, inds_ref = run(Xtrain, Xtest, algorithms[0])

        for i in xrange(1, len(algorithms)):

            algorithm = algorithms[i]
            print("\tChecking " + unicode(algorithm) + " ...")
            dists, inds = run(Xtrain, Xtest, algorithms[i])
            
            assert numpy.allclose(dists_ref, dists)
            assert numpy.allclose(inds_ref, inds)

