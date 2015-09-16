'''
Created on 15.09.2015

@author: fgieseke
'''

import numpy
from bufferkdtree.neighbors.brute.base import BruteNN
from bufferkdtree.neighbors.kdtree.base import KDTreeNN
from bufferkdtree.neighbors.buffer_kdtree.base import BufferKDTreeNN

class NearestNeighbors(object):
    """ Provides access to the different implementations, in particular
    to the Buffer k-d Tree implementation. 
    
    The brute-force implementatation is only used for comparison 
    in relatively low-dimensional spaces; the performance is suboptimal
    for higher dimensional feature spaces.
    """

    ALLOWED_ALGORITHMS = ["brute", "kd_tree", "buffer_kd_tree"]
    ALLOWED_FLOAT_TYPES = ["float"]
    
    def __init__(self, \
                 n_neighbors=5, \
                 algorithm="buffer_kd_tree", \
                 float_type="float", \
                 tree_depth=None, \
                 splitting_type="cyclic", \
                 leaf_size=30, \
                 n_train_chunks=1, \
                 use_gpu=True, \
                 plat_dev_ids={0:[0]}, \
                 allowed_train_mem_percent_chunk=0.2, \
                 allowed_test_mem_percent=0.8, \
                 n_jobs=1, \
                 verbose=0, \
                 **kwargs):

        """ Model for unsupervised nearest neighbor search.
        
        Parameters
        ----------
        n_neighbors : int, optional (default = 5)
            Number of neighbors used for kneighbors (as default).
        ...
        
        """
        
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.float_type = float_type
        self.tree_depth = tree_depth
        self.splitting_type = splitting_type
        self.leaf_size = leaf_size
        self.use_gpu = use_gpu
        self.n_train_chunks = n_train_chunks        
        self.plat_dev_ids = plat_dev_ids
        self.allowed_train_mem_percent_chunk = allowed_train_mem_percent_chunk
        self.allowed_test_mem_percent = allowed_test_mem_percent        
        self.n_jobs = n_jobs
        self.verbose = verbose

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
                "algorithm": self.algorithm, \
                "float_type": self.float_type, \
                "tree_depth": self.tree_depth, \
                "splitting_type": self.splitting_type, \
                "leaf_size": self.leaf_size, \
                "use_gpu": self.use_gpu, \
                "n_train_chunks": self.n_train_chunks, \
                "plat_dev_ids": self.plat_dev_ids, \
                "allowed_train_mem_percent_chunk": self.allowed_train_mem_percent_chunk, \
                "allowed_test_mem_percent": self.allowed_test_mem_percent, \
                "n_jobs": self.n_jobs, \
                "verbose": self.verbose
                }

    def set_params(self, **params):
        """ Set the parameters of this estimator.
        
        Parameters
        ----------
        params : dict
            Dictionary containing the 
            parameters to be used.
        """
        
        for parameter, value in params.items():
            self.setattr(parameter, value)

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

        print self.algorithm
        assert self.algorithm in self.ALLOWED_ALGORITHMS
        
        self._set_internal_data_types()
        
        self.X = X.astype(self.numpy_dtype_float)
        
        self.wrapper = self._get_wrapper()
        self.wrapper.fit(X)
        
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

        if not hasattr(self, 'wrapper'):
            raise Exception("Model must be fitted before querying ...")
        
        self._set_internal_data_types()
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if X is not None:
            query_is_train = False
            X = X.astype(self.numpy_dtype_float)
        else:
            query_is_train = True
            X = self.X
            # same as in scikit-learn: Include an extra neighbor to account 
            # for the sample itself being returned, which is removed later
            n_neighbors += 1                

        # same as in scikit-learn
        train_size = self.X.shape[0]
        if n_neighbors > train_size:
            raise ValueError("Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %  (train_size, n_neighbors))

        # use wrapper to get nearest neighbors        
        result = self.wrapper.kneighbors(X=X, n_neighbors=n_neighbors, return_distance=return_distance)

        if not query_is_train:
            return result
        else:
            if return_distance:
                dist, neigh_ind = result
                dist = dist[:, 1:]
                neigh_ind = neigh_ind[:, 1:]                
                return dist, neigh_ind                
            else:
                neigh_ind = result            
                neigh_ind = neigh_ind[:, 1:]
                return neigh_ind        
    
    def compute_optimal_tree_depth(self, Xtrain, Xtest, target="test", tree_depths=None):
        """ Computes the optimal tree depth for a 
        tree-based nearest neighbor method.
        
        Returns
        -------
        opt_height : int
            The optimal tree depth based
            on the target provided.
        """
        
        ALLOWED_TARGETS = ['train', 'test', 'both']
        
        if self.algorithm not in ["kd_tree", "buffer_kd_tree"]:
            raise Exception("Optimal tree depth can only be determined for tree-based methods!")
        
        if target not in ALLOWED_TARGETS:
            raise Exception("Target is invalid: " + unicode(target))
        
        return self._get_wrapper().compute_optimal_tree_depth(Xtrain=Xtrain, Xtest=Xtest, \
                                                              target=target, tree_depths=tree_depths)

    def _set_internal_data_types(self):
        """ Set numpy float and int dtypes
        """
        
        if self.float_type == "float":
            self.numpy_dtype_float = numpy.float32
        else:
            self.numpy_dtype_float = numpy.float64
        self.numpy_dtype_int = numpy.int32
                
    def _get_wrapper(self):
        """ Returns the corresponding wrapper module.
        
        Returns
        -------
        wrapper : object
            The wrapper object
        """

        if self.algorithm == "brute":
            return BruteNN(n_neighbors=self.n_neighbors, float_type=self.float_type, \
                            use_gpu=self.use_gpu, plat_dev_ids=self.plat_dev_ids, \
                            n_jobs=self.n_jobs, verbose=self.verbose)

        elif self.algorithm == "kd_tree":
            return KDTreeNN(n_neighbors=self.n_neighbors, float_type=self.float_type, \
                            tree_depth=self.tree_depth, leaf_size=self.leaf_size, \
                            splitting_type=self.splitting_type, \
                            n_jobs=self.n_jobs, verbose=self.verbose)

        elif self.algorithm == "buffer_kd_tree":
            return BufferKDTreeNN(n_neighbors=self.n_neighbors, float_type=self.float_type, \
                            tree_depth=self.tree_depth, leaf_size=self.leaf_size, \
                            splitting_type=self.splitting_type, \
                            n_train_chunks=self.n_train_chunks, use_gpu=self.use_gpu, \
                            plat_dev_ids=self.plat_dev_ids, \
                            allowed_train_mem_percent_chunk=self.allowed_train_mem_percent_chunk, \
                            allowed_test_mem_percent=self.allowed_test_mem_percent, \
                            n_jobs=self.n_jobs, \
                            verbose=self.verbose)

        else:
            raise Exception("Invalid assignment for 'algorithm':" + unicode(self.algorithm))
        
    def __repr__(self):
        """ String representation.
        """
        
        return str(self.get_params())
    
