'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

import os
import math
import threading
import warnings
import numpy as np
import wrapper_cpu_float, wrapper_cpu_double
import wrapper_gpu_opencl_float, wrapper_gpu_opencl_double

class DeviceQueryThread(threading.Thread):
     
    def __init__(self, wrapper_module, params, record, X, d_mins, idx_mins, start_idx, end_idx, verbose=0):
 
        threading.Thread.__init__(self)
        
        self.wrapper_module = wrapper_module 
        self.params = params
        self.record = record
        self.X = X
        self.d_mins = d_mins
        self.idx_mins = idx_mins
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.verbose = verbose
 
    def run(self):
        
        self.wrapper_module.neighbors_extern(self.X[self.start_idx:self.end_idx], \
                                             self.d_mins[self.start_idx:self.end_idx], \
                                             self.idx_mins[self.start_idx:self.end_idx], \
                                             self.record, self.params)
        
        
class FitThread(threading.Thread):
     
    def __init__(self, wrapper_module, params, record, X):
 
        threading.Thread.__init__(self)
        
        self.wrapper_module = wrapper_module 
        self.params = params
        self.record = record
        self.X = X
 
    def run(self):
        
        self.wrapper_module.fit_extern(self.X, self.record, self.params)
            
class BruteNN(object):
    """ Brute-force nearest neighbor computations.
    """

    ALLOWED_FLOAT_TYPES = ['float', 'double']
    ALLOWED_USE_GPU = [True, False]

    OPENCL_ERROR_MAPPINGS = {-1:"ERROR_NO_PLATFORMS",
                             - 2:"ERROR_INVALID_PLATFORMS",
                             - 3:"ERROR_NO_DEVICES",
                             - 4:"ERROR_INVALID_DEVICE"}
    
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

        if self.verbose > 0:
            print("Freeing external resources ...")        
            
        for platform_id in self.plat_dev_ids.keys():
            for device_id in self.plat_dev_ids[platform_id]:
                    
                try:
                    wrapper_params = self.wrapper_instances[platform_id][device_id]['params']
                    wrapper_record = self.wrapper_instances[platform_id][device_id]['record']            
                    self._get_wrapper_module().free_resources_extern(wrapper_record, wrapper_params)
                except Exception as e:
                    if self.verbose > 0:
                        print("Exception occured while freeing external resources: " + unicode(e))
                      
    def get_params(self):
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

        # make sure that the array is contiguous
        # (needed for the swig module)
        X = np.ascontiguousarray(X)
                
        # convert input data to correct types (and generate local
        # variable to prevent destruction of external array)
        self.X = X.astype(self.numpy_dtype_float)
        
        if self.X.shape[1] > 30:
            warnings.warn(
                """
                The brute-force implementatation is only used for comparison 
                in relatively low-dimensional spaces; the performance is 
                suboptimal for higher dimensional feature spaces (but even 
                superior over other matrix based implementations making use
                e.g., CUBLAS).            
                """)
            
        self.wrapper_instances = {}

        root_path = os.path.dirname(os.path.realpath(__file__))
        kernel_sources_dir = os.path.join(root_path, "../../src/neighbors/brute/kernels/opencl/")
                                
        # initialize devices
        for platform_id in self.plat_dev_ids.keys():
            self.wrapper_instances[platform_id] = {}

            for device_id in self.plat_dev_ids[platform_id]:
                
                self._validate_device(platform_id, device_id)
        
                wrapper_params = self._get_wrapper_module().BRUTE_PARAMETERS()
                
                self._get_wrapper_module().init_extern(self.n_neighbors, self.n_jobs, platform_id, \
                                                       device_id, kernel_sources_dir, self.verbose,
                                                       wrapper_params)
                wrapper_record = self._get_wrapper_module().BRUTE_RECORD()
                        
                self.wrapper_instances[platform_id][device_id] = {}
                self.wrapper_instances[platform_id][device_id]['params'] = wrapper_params
                self.wrapper_instances[platform_id][device_id]['record'] = wrapper_record

        threads = []
        
        # fit all models
        for platform_id in self.plat_dev_ids.keys():
            for device_id in self.plat_dev_ids[platform_id]:
                
                wrapper_module = self._get_wrapper_module()
                wrapper_params = self.wrapper_instances[platform_id][device_id]['params']
                wrapper_record = self.wrapper_instances[platform_id][device_id]['record']
                
                thread = FitThread(wrapper_module, wrapper_params, wrapper_record, self.X)
                threads.append(thread)                
            
        # start all threads
        if self.verbose > 0:
            print("Starting all fitting threads ...")
         
        for thread in threads:
            thread.start()
            
        # wait for all threads to be completed
        for thread in threads:
            thread.join()
        if self.verbose > 0:
            print("All fitting threads finished!")                    
        
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

        # split up queries over all devices
        n_total_devices = self._get_n_total_devices()
        n_chunk = int(math.ceil(len(X) / n_total_devices))
        
        if self.verbose > 0:
            print("Splitting queries into %i chunks." % int(n_total_devices))
        
        d_mins = np.zeros((X.shape[0], n_neighbors), dtype=self.numpy_dtype_float)
        idx_mins = np.zeros((X.shape[0], n_neighbors), dtype=self.numpy_dtype_int)
        
        threads = []
        
        chunk_start = 0
        chunk_end = n_chunk
        
        while chunk_end < len(X) + n_chunk / 2:
            for platform_id in self.plat_dev_ids.keys():
                for device_id in self.plat_dev_ids[platform_id]:
                          
                    if chunk_start < len(X):
                        
                        if chunk_end > len(X):
                            chunk_end_cropped = len(X)
                        else:
                            chunk_end_cropped = chunk_end
                        
                        wrapper_module = self._get_wrapper_module()
                        wrapper_params = self.wrapper_instances[platform_id][device_id]['params']
                        wrapper_record = self.wrapper_instances[platform_id][device_id]['record']
                        
                        if self.verbose > 0:
                            print("Initializing device thread for range %i-%i ..." % (chunk_start, chunk_end_cropped))
                        thread = DeviceQueryThread(wrapper_module, wrapper_params, wrapper_record, \
                                             X, d_mins, idx_mins, chunk_start, chunk_end_cropped, \
                                             verbose=self.verbose)
                        threads.append(thread)
    
                    chunk_start += n_chunk
                    chunk_end += n_chunk
                    

        if self.verbose > 0:
            print("Processing all query threads ...")
            
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()         
               
        if self.verbose > 0:
            print("All query threads finished!")  

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

    def _validate_device(self, platform_id, device_id):
        """ Checks if the platform and the devices are valid
        
        platform_id : int
            The platform id
        device_id : int
            The device id
            
        Raises an exception in case any of the 
        platform devices is not valid
        """
         
        err = self._get_wrapper_module().extern_check_platform_device(platform_id, device_id)
        
        if (err < 0):
            raise Exception("Could not retrieve device infos. Error code: %s " % self.OPENCL_ERROR_MAPPINGS[err])
        
    def _get_n_total_devices(self):
        """ Returns the total number of devices
        that can be used for fitting and querying.
        
        Returns
        int : the number of devices
        """
        
        n_devices = 0
        for platform_id in self.plat_dev_ids.keys():
            n_devices += len(self.plat_dev_ids[platform_id])
            
        return n_devices
            
    def __repr__(self):
        """ Reprentation of this object
        """
        
        return str(self.get_params())
    
