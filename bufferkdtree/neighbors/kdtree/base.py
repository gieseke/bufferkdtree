'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

import math
import time
import numpy as np
import wrapper_cpu_float, wrapper_cpu_double

class KDTreeNN(object):
    """
    Nearest neighbor queries based on k-d trees.
    """
    
    SPLITTING_TYPE_MAPPINGS = {'cyclic': 0, 'longest': 1}
    ALLOWED_FLOAT_TYPES = ['float', 'double']

    def __init__(self, \
                 n_neighbors=5, \
                 leaf_size=30, \
                 tree_depth=None, \
                 float_type="float", \
                 splitting_type="cyclic", \
                 n_jobs=1, \
                 verbose=0):
        """ Model for unsupervised nearest neighbor search (k-d-trees).
        
        Parameters
        ----------
        n_neighbors : int, optional (default = 5)
            Number of neighbors used for kneighbors (as default).
        ...
        
        """
        
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.tree_depth = tree_depth
        self.float_type = float_type
        self.splitting_type = splitting_type
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __del__(self):
        """ Free external resources if needed
        """

        try:
            if self.verbose > 0:
                print("Freeing external resources ...")
            self._get_wrapper_module().free_resources_extern()
        except Exception, e:
            if self.verbose > 0:
                print("Exception occured while freeing external resources: " + unicode(e))

    def get_params(self, deep=True):
        """ Get parameters for this estimator.
        
        Parameters
        ----------
        deep : boolean, optional
            If True: Will return the parameters for this 
            estimator and its contained estimator subobjects.
        
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        
        return {"n_neighbors": self.n_neighbors, \
                "leaf_size": self.leaf_size, \
                "tree_depth": self.tree_depth, \
                "float_type": self.float_type, \
                "splitting_type": self.splitting_type, \
                "n_jobs": self.n_jobs, \
                "verbose": self.verbose}
                    
    def fit(self, X):
        """ Fit the model to the given data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number 
            of patterns and n_features the number 
            of features.
            
        Returns
        -------
        self : object
            The object itself
        """
        
        assert self.float_type in self.ALLOWED_FLOAT_TYPES

        # set float and int type
        self._set_internal_data_types()
        
        # compute final tree depth (based on X and desired leaf size)
        final_tree_depth = self._compute_final_tree_depth(len(X), self.leaf_size, self.tree_depth)
        if self.verbose > 0:
            print("Final tree depth: " + unicode(final_tree_depth))
            
        # make sure that the array is contiguous
        # (needed for the swig module)
        X = np.ascontiguousarray(X)
        
        # convert input data to correct types (and generate local
        # variable to prevent destruction of external array)
        self.X = X.astype(self.numpy_dtype_float)
        
        # initialize device
        self.wrapper_params_struct = self._get_wrapper_module().KD_TREE_PARAMETERS()
        self._get_wrapper_module().init_extern(self.n_neighbors, final_tree_depth, self.n_jobs, \
                                               self.SPLITTING_TYPE_MAPPINGS[self.splitting_type], \
                                               self.verbose, self.wrapper_params_struct)
        self.wrapper_kdtree_struct = self._get_wrapper_module().KD_TREE_RECORD()
        self._get_wrapper_module().fit_extern(self.X, self.wrapper_kdtree_struct, self.wrapper_params_struct)

        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """ Finds the K-neighbors of a point.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The set of query points. If not provided,
            the neighbors of each point in the training
            data are returned (in this case, the query
            point itself is not considered its own 
            neighbor.
        n_neighbors : int
            The number of neighbors to get. The default
            values is the one passed to the 
            constructor.
        return_distance : boolean, optional, default is True
            If False, then the distances associated with 
            each query will not be returned.
            
        Returns
        -------
        dist : array
            The array containing the distances
        idx : array
            The array containing the indices
        """
        
        assert return_distance == True
        
        if n_neighbors is not None:
            if self.n_neighbors != n_neighbors:
                self.n_neighbors = n_neighbors
                if self.verbose > 0:
                    print("\nMODEL MUST BE RETRAINED ...\n")
                self.fit(self.X)

        if X is None:
            X = self.X
        else:
            # make sure that the array is contiguous
            # (needed for the swig module)
            X = np.ascontiguousarray(X)
            X = X.astype(self.numpy_dtype_float)

        d_mins = np.zeros((X.shape[0], n_neighbors), dtype=self.numpy_dtype_float)
        idx_mins = np.zeros((X.shape[0], n_neighbors), dtype=self.numpy_dtype_int)

        self._get_wrapper_module().neighbors_extern(X, d_mins, idx_mins, self.wrapper_kdtree_struct, self.wrapper_params_struct)

        return d_mins, idx_mins
        
    def _get_wrapper_module(self):
        """ Returns the corresponding swig 
        wrapper module.
        
        Returns
        -------
        wrapper : object
            The wrapper object
        """
        
        if self.float_type == "float":
            return wrapper_cpu_float
        elif self.float_type == "double":
            return wrapper_cpu_double
        else:
            raise Exception("Unknown float_type: " + unicode(self.float_type))

    def compute_optimal_tree_depth(self, Xtrain, Xtest, target="test", tree_depths=None):
        """ Computes the optimal tree depth.
        
        Returns
        -------
        opt_height : int
            The optimal tree depth based
            on the target provided.
        """
        
        max_depth = int(math.floor(math.log(len(Xtrain), 2)))

        if tree_depths is None:
            tree_depths = range(4, max_depth - 1)

        runtimes = {}
        
        if target == "test":            
            
            for tree_depth in tree_depths:

                new_model = eval(self.__class__.__name__)(**self.get_params())
                new_model.tree_depth = tree_depth
                new_model.fit(Xtrain)

                start = time.time()
                new_model.kneighbors(Xtest)
                end = time.time()

                if new_model.verbose:
                    print("tree_depth %i -> %f" % (tree_depth, end - start))
                runtimes[tree_depth] = end - start 
        
        elif target == "train":

            for tree_depth in tree_depths:

                new_model = eval(self.__class__.__name__)(**self.get_params())
                new_model.tree_depth = tree_depth
                start = time.time()
                new_model.fit(Xtrain)
                end = time.time()
                
                if new_model.verbose:
                    print("tree_depth %i -> %f" % (tree_depth, end - start))
                runtimes[tree_depth] = end - start 
                
        elif target == "both":
            
            for tree_depth in tree_depths:
                            
                new_model = eval(self.__class__.__name__)(**self.get_params())
                new_model.tree_depth = tree_depth
                start = time.time()
                new_model.fit(Xtrain)
                new_model.kneighbors(Xtest)
                end = time.time()
                
                if self.verbose > 0:
                    print("tree_depth %i -> %f" % (tree_depth, end - start))
                runtimes[tree_depth] = end - start

        else:

            raise Exception("Unknown target: " + unicode(target))
                                                                        
        return min(runtimes, key=runtimes.get)    
    
    def _compute_final_tree_depth(self, n, leaf_size, tree_depth):
        """ Computes tree depth for kd tree.
        
        Parameters
        ----------
        n : int
            The number of patterns
        leaf_size : int
            The desired leaf size
        tree_depth : int
            The desired tree depth
            
        Returns
        -------
        int : the final tree depth
        """

        # last level might be incomplete
        d = int(math.floor(math.log(n, 2)))

        if tree_depth is not None:

            if tree_depth > d:
                raise Warning("tree_depth %i too large (using smaller depth): %i.\n" % (tree_depth, d))
                return d
            else:
                return tree_depth
        
        else:
            
            if not self.leaf_size > 1:
                raise Exception("Parameter 'leaf_size' must be larger than 1")
            
            subtr = int(math.floor(math.log(leaf_size, 2)))

            if d - subtr < 0:
                # tree consisting of single leaf
                raise Warning("tree_depth smaller than 0; setting tree_depth to 0")
                return 0
            else:
                return d - subtr

    def _set_internal_data_types(self):
        """ Set numpy float and int dtypes
        """
        
        if self.float_type == "float":
            self.numpy_dtype_float = np.float32
        else:
            self.numpy_dtype_float = np.float64
        self.numpy_dtype_int = np.int32
        
    def __repr__(self):
        """ Reprentation of this object
        """
        return str(self.get_params())
    