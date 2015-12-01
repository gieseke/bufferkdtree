'''
Created on 18.11.2015

@author: Fabian Gieseke
'''

import math
import time
from bufferkdtree.neighbors import NearestNeighbors

def compute_optimal_tree_depth(params, Xtrain, Xtest, target="test", tree_depths=None, verbose=1):
    """ Computes the optimal tree depth.
    
    Returns
    -------
    opt_height : int
        The optimal tree depth based
        on the target provided.
    """
   
    ALLOWED_TARGETS = ['train', 'test', 'both']
    if target not in ALLOWED_TARGETS:
        raise Exception("Target is not valid (allowed ones are " + \
                        unicode(ALLOWED_TARGETS) + ": " + unicode(target))        
        
    if params["algorithm"] not in ["kd_tree", "buffer_kd_tree"]:
        raise Exception("Optimal tree depth can only be \
                determined for tree-based methods!")        
    
    max_depth = int(math.floor(math.log(len(Xtrain), 2)))

    if tree_depths is None:
        tree_depths = range(2, max_depth - 1)

    kwargs = {'target':target, 'tree_depths':tree_depths, 'verbose':verbose}    
    return _conduct_tree_depths_comparison(params, Xtrain, Xtest, **kwargs)

def _conduct_tree_depths_comparison(params, Xtrain, Xtest, target="test", tree_depths=None, verbose=1):
    
    runtimes = {}
    
    model = NearestNeighbors(**params)
    
    if target == "test":            
        
        for tree_depth in tree_depths:

            #model = copy.deepcopy(model)
            model.tree_depth = tree_depth
            model.fit(Xtrain)

            start = time.time()
            model.kneighbors(Xtest)
            end = time.time()

            if model.verbose:
                print("tree_depth %i -> %f" % (tree_depth, end - start))
            runtimes[tree_depth] = end - start
                    
    elif target == "train":

        for tree_depth in tree_depths:

            #model = copy.deepcopy(model)
            model.tree_depth = tree_depth
            start = time.time()
            model.fit(Xtrain)
            end = time.time()
            
            if model.verbose:
                print("tree_depth %i -> %f" % (tree_depth, end - start))
            runtimes[tree_depth] = end - start 
                            
    elif target == "both":
        
        for tree_depth in tree_depths:
                        
            #model = copy.deepcopy(model)
            model.tree_depth = tree_depth
            start = time.time()
            model.fit(Xtrain)
            model.kneighbors(Xtest)
            end = time.time()
            
            if verbose > 0:
                print("tree_depth %i -> %f" % (tree_depth, end - start))
            runtimes[tree_depth] = end - start

    else:

        raise Exception("Unknown target: " + unicode(target))

    return min(runtimes, key=runtimes.get)
