'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

import numpy
import warnings

class NearestNeighbors(object):
    """ The 'NearestNeighbors' provides access to all nearest neighbor
    implementations. It has simmilar parameters as the corresponding
    implementation of the `scikit-learn <http://scikit-learn.org>`_ 
    package.   
    
    The main method is the "buffer_kd_tree", which can be seen as
    mix between the "brute" and the "kd_tree" implementations. 
    
    Parameters
    ----------
    
    n_neighbors : int (default 5)
        Number of neighbors used
        
    algorithm : {"brute", "kd_tree", "buffer_kd_tree"}, optional (default="buffer_kd_tree")
        The algorithm that shall be used to compute 
        the nearest neighbors. One of  
        - 'brute': brute-force search
        - 'kd_tree': k-d tree based search
        - 'buffer_kd_tree': buffer k-d tree based search (with GPUs)
        
    tree_depth : int or None, optional (default=None)
        Passed to the 'kd_tree' and 'buffer_kd_tree' implementation. 
        In case 'tree_depth' is specified, a tree of such a depth
        is built ('tree_depth' has priority over 'leaf_size').
        
    leaf_size : int, optional (default=30)
        Passed to the 'kd_tree' and 'buffer_kd_tree' implementation. 
        In case 'leaf_size' is set, the corresponding tree depth
        is computed (is ignored in case tree_depth is not None).
        
    splitting_type : {'cyclic'}, optional (default='cyclic')
        Passed to the 'kd_tree' and 'buffer_kd_tree' implementation.
        The splitting rule that shall be used to 
        construct the kd tree. Currently, only
        "cyclic" is supported.
        
    n_train_chunks : int, optional (default=1)
        Passed to the 'buffer_kd_tree' implementation.
        The number of chunks the training patterns shall 
        be processed in; only needed in case the 
        training patterns do not fit on the GPU (in case 
        n_train_chunks is too small, it is increased
        automatically).
        
    plat_dev_ids : dict, optional (default={0:[0]})
        Passed to the 'brute' and the 'buffer_kd_tree' implementation.
        The platforms and devices that shall be used. E.g., 
        plat_dev_ids={0:[0,1]} makes use of platform 0 and
        the first two devices.
        
    allowed_train_mem_percent_chunk : float, optional (default=0.2)
        Passed to the 'buffer_kd_tree' implementation.
        The amount of memory (OpenCL) used for the 
        training patterns (in percent).
         
    allowed_test_mem_percent : float, optional (default=0.8)
        Passed to the 'buffer_kd_tree' implementation.
        The amount of memory (OpenCL) used for the 
        test/query patterns (in percent).
    
    n_jobs : int, optional (default=1)
        Passed to the 'kd_tree' implementation.
        The number of threads used for the querying phase.
        
    verbose : int, optional (default=0)
        The verbosity level (0=no output, 1=output)
        
    Examples
    --------
           
      >>> import numpy
      >>> from bufferkdtree.neighbors.base import NearestNeighbors
      >>> X = numpy.random.uniform(low=-1,high=1,size=(10000,10))
      >>> nbrs = NearestNeighbors(n_neighbors=10, algorithm="buffer_kd_tree", tree_depth=9, plat_dev_ids={0:[0]})    
      >>> nbrs.fit(X)
      >>> dists, inds = nbrs.kneighbors(X)   
       
    Notes
    -----
    
    The brute-force implementatation is only used for comparison 
    in relatively low-dimensional spaces; the performance is 
    suboptimal for higher dimensional feature spaces (but even 
    superior over other matrix based implementations making use
    e.g., CUBLAS).  
    
    The performance of the GPU implementations depends on the
    corresponding architecture. An important ingredient is 
    the support of automatic hardware caches.
    
    Only single-precision is supported until now.
    """

    ALLOWED_ALGORITHMS = ["brute", "kd_tree", "buffer_kd_tree"]
    N_NEIGHBORS_CL_THRES = 30
    
    def __init__(self, \
                 n_neighbors=5, \
                 algorithm="buffer_kd_tree", \
                 tree_depth=None, \
                 leaf_size=30, \
                 splitting_type="cyclic", \
                 n_train_chunks=1, \
                 plat_dev_ids={0:[0]}, \
                 allowed_train_mem_percent_chunk=0.2, \
                 allowed_test_mem_percent=0.8, \
                 n_jobs=1, \
                 verbose=0, \
                 **kwargs):

        """ Constructor
        """
        
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.tree_depth = tree_depth
        self.leaf_size = leaf_size
        self.splitting_type = splitting_type        
        self.n_train_chunks = n_train_chunks        
        self.plat_dev_ids = plat_dev_ids
        self.allowed_train_mem_percent_chunk = allowed_train_mem_percent_chunk
        self.allowed_test_mem_percent = allowed_test_mem_percent        
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.float_type = "float" 

    def get_params(self):
        """ Get parameters for this estimator.
        
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        
        return {"n_neighbors": self.n_neighbors, \
                "algorithm": self.algorithm, \
                "tree_depth": self.tree_depth, \
                "leaf_size": self.leaf_size, \
                "splitting_type": self.splitting_type, \
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
            The set of training/reference points, where
            'n_samples' is the number points and 
            'n_features' the number of features.
            
        Returns
        -------
        self : instance of NearestNeighbors
            The object itself
        """

        assert self.algorithm in self.ALLOWED_ALGORITHMS
        
        self._set_internal_data_types()
        self.X = X.astype(self.numpy_dtype_float)
        self.wrapper = self._get_wrapper()
        self.wrapper.fit(X)
        
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """ Finds the nearest neighbors for a given set of points.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The set of query points. If not provided,
            the neighbors of each point in the training
            data are returned (in this case, the query
            point itself is not considered its own  
            neighbor.
            
        n_neighbors : int or None, optional (default=None)
            The number of nearest neighbors that shall be
            returned for each query points. If 'None', then
            the default values provided to the constructor
            is used.
            
        return_distance : bool, optional (default=True)
            If False, then the distances associated with 
            each query will not be returned. Otherwise, they
            will be returned.
            
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

        # sanity checks: similar to scikit-learn
        train_size = self.X.shape[0]
        if n_neighbors > train_size:
            raise ValueError("n_neighbors must be <= n_samples, "
                " but n_samples=%d, n_neighbors=%d" % (train_size, n_neighbors))
        if n_neighbors > self.N_NEIGHBORS_CL_THRES \
            and self.algorithm in ["brute", "buffer_kd_tree"]:
            warnings.warn("""
                The performance of the many-core implementation 
                decreases for large values of 'n_neighbors'!"""
                , Warning)

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
        """ Computes the optimal tree depth for the 
        tree-based implementations. The method tests
        various assignments of the parameters and 
        simply measures the time needed for the approach
        tp finish.
        
        Parameters
        ----------
        Xtrain : array-like, shape (n_samples, n_features)
            The set of training/reference points, where
            'n_samples' is the number points and 
            'n_features' the number of features.
            
        Xtest : array-like, shape (n_samples, n_features)
            The set of testing/querying points, where
            'n_samples' is the number points and 
            'n_features' the number of features.
            
        target : {'train', 'test', 'both'}, optional (default='test')
            The runtime target, i.e., which phase shall 
            be optimized. Three choices:
            - 'train' : The training phase
            - 'test' : The testing phase
            - 'both' : Both phases
        
        tree_depths : list or None, optional
            The range of different tree depths that 
            shall be tested. If None, then the default
            ranges are used by the different implementations:
            
            - buffer_kd_tree : range(2, max_depth - 1)
            - kd_tree : range(4, max_depth - 1)
            
            where max_depth = int(math.floor(math.log(len(Xtrain), 2)))
        
        Returns
        -------
        opt_height : int
            The optimal tree depth
        """
        
        ALLOWED_TARGETS = ['train', 'test', 'both']
        
        if self.algorithm not in ["kd_tree", "buffer_kd_tree"]:
            raise Exception("Optimal tree depth can only be \
                    determined for tree-based methods!")
        
        if target not in ALLOWED_TARGETS:
            raise Exception("Target is not valid (allowed ones are " + \
                            unicode(ALLOWED_TARGETS) + ": " + unicode(target))
        
        return self._get_wrapper().compute_optimal_tree_depth(Xtrain=Xtrain, \
                                                              Xtest=Xtest, \
                                                              target=target, \
                                                              tree_depths=tree_depths)

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
        
        from .brute.base import BruteNN
        from .kdtree.base import KDTreeNN
        from .buffer_kdtree.base import BufferKDTreeNN
        
        if self.algorithm == "brute":
            return BruteNN(n_neighbors=self.n_neighbors, float_type=self.float_type, \
                            use_gpu=True, plat_dev_ids=self.plat_dev_ids, \
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
                            n_train_chunks=self.n_train_chunks, use_gpu=True, \
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
    