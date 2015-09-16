'''
Created on 15.09.2015

@author: fgieseke
'''

import numpy
from bufferkdtree.neighbors.base import NearestNeighbors

X = numpy.random.uniform(low=-1,high=1,size=(10000,10))

nbrs = NearestNeighbors(n_neighbors=10, algorithm="buffer_kd_tree", \
                        tree_depth=9, plat_dev_ids={0:[1]}, verbose=0)    
nbrs.fit(X)
dists, inds = nbrs.kneighbors(X)
print dists
