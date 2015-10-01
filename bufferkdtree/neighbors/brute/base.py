'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

import os
import numpy as np
import wrapper_cpu_float, wrapper_cpu_double
import wrapper_gpu_opencl_float, wrapper_gpu_opencl_double

class BruteNN(object):
    """ Brute-force nearest neighbor computations.
    """

    ALLOWED_FLOAT_TYPES = ['float', 'double']
    ALLOWED_USE_GPU = [True, False]

    def __init__(self, \
                 n_neighbors=5, \
                 float_type="float", \
                 use_gpu=False, \
                 plat_dev_ids={0:[0]}, \
                 n_jobs=1, \
                 verbose=0):
        """ Model for unsupervised nearest neighbor search (brute-force).
        
        Parameters
        ----------
        n_neighbors : int, optional (default = 5)
            Number of neighbors used for kneighbors (as default).
        ...
        
        """
        
        self.n_neighbors = n_neighbors
        self.float_type = float_type
        self.use_gpu = use_gpu
        self.plat_dev_ids = plat_dev_ids
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __del__(self):
        """ Free external resources if needed
        """
        
        try:
            if self.verbose > 0:
                print("Freeing external resources ...")
            self._get_wrapper_module().free_resources_extern()
        except Exception as e:
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
                "float_type": self.float_type, \
                "use_gpu": self.use_gpu, \
                "plat_dev_ids": self.plat_dev_ids, \
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
        assert self.use_gpu in self.ALLOWED_USE_GPU

        # set float and int type
        self._set_internal_data_types()
                        
        # initialize device
        platform_id = self.plat_dev_ids.keys()[0]
        device_id = self.plat_dev_ids[platform_id][0]
        
        root_path = os.path.dirname(os.path.realpath(__file__))
        kernel_sources_dir = os.path.join(root_path, "../../src/neighbors/brute/kernels/opencl/")
        self._get_wrapper_module().init_extern(self.n_neighbors, self.n_jobs, platform_id, \
                                               device_id, kernel_sources_dir, self.verbose)
        
        # make sure that the array is contiguous
        # (needed for the swig module)
        X = np.ascontiguousarray(X)
                
        # convert input data to correct types (and generate local
        # variable to prevent destruction of external array)
        self.X = X.astype(self.numpy_dtype_float)
        
        if self.X.shape[1] > 30:
            raise Warning(
                """
                The brute-force implementatation is only used for comparison 
                in relatively low-dimensional spaces; the performance is 
                suboptimal for higher dimensional feature spaces (but even 
                superior over other matrix based implementations making use
                e.g., CUBLAS).            
                """)
        
        # fit model
        self._get_wrapper_module().fit_extern(self.X)

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

        # compute distances and indices
        d_mins = np.zeros((X.shape[0], n_neighbors), dtype=self.numpy_dtype_float)
        idx_mins = np.zeros((X.shape[0], n_neighbors), dtype=self.numpy_dtype_int)
        self._get_wrapper_module().neighbors_extern(X, d_mins, idx_mins)

        return np.sqrt(d_mins), idx_mins

    def _get_wrapper_module(self):
        """ Returns the corresponding swig 
        wrapper module.
        
        Returns
        -------
        wrapper : object
            The wrapper object
        """
        
        if self.float_type == "float":
            if self.use_gpu == True:
                return wrapper_gpu_opencl_float
            else:
                return wrapper_cpu_float
        elif self.float_type == "double":
            if self.use_gpu == True:
                return wrapper_gpu_opencl_double
            else:
                return wrapper_cpu_double

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
    
