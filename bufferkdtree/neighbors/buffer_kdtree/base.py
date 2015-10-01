'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

from __future__ import division

import os
import math
import time
import numpy as np
import threading 
import wrapper_cpu_float, wrapper_cpu_double
import wrapper_gpu_opencl_float, wrapper_gpu_opencl_double
      
class DeviceQueryThread(threading.Thread):
     
    def __init__(self, wrapper_module, tree_params, tree_record, X, d_mins, idx_mins, start_idx, end_idx, verbose=0):
 
        threading.Thread.__init__(self)
        
        self.wrapper_module = wrapper_module 
        self.tree_params = tree_params
        self.tree_record = tree_record
        self.X = X
        self.d_mins = d_mins
        self.idx_mins = idx_mins
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.verbose = verbose
 
    def run(self):

        # perform queries in chunks if needed
        n_test_max = self.wrapper_module.get_max_nXtest_extern(self.tree_record, self.tree_params)
        n_total = self.end_idx - self.start_idx
        
        if n_total <= n_test_max:
            
            self.wrapper_module.neighbors_extern(self.X[self.start_idx:self.end_idx], \
                                                 self.d_mins[self.start_idx:self.end_idx], \
                                                 self.idx_mins[self.start_idx:self.end_idx], \
                                                 self.tree_record, self.tree_params)
        else:
            
            # split up in equal-sized chunks
            n_chunk = int(math.ceil(n_total / n_test_max))
            chunk_size = int(math.ceil(n_total / n_chunk))
        
            chunk_start = self.start_idx
            chunk_end = chunk_start + chunk_size
            
            while chunk_start < self.end_idx:
                
                chunk_end_cropped = chunk_end
                if chunk_end_cropped > self.end_idx:
                    chunk_end_cropped = self.end_idx
                
                if self.verbose > 0:
                    print("Processing query chunk %i -> %i" % (chunk_start, chunk_end_cropped))

                self.wrapper_module.neighbors_extern(self.X[chunk_start:chunk_end_cropped], \
                                                     self.d_mins[chunk_start:chunk_end_cropped], \
                                                     self.idx_mins[chunk_start:chunk_end_cropped], \
                                                     self.tree_record, self.tree_params)
                self.wrapper_module.extern_free_query_buffers(self.tree_record, self.tree_params)
                
                chunk_start += chunk_size
                chunk_end += chunk_size
        
class BuildThread(threading.Thread):
     
    def __init__(self, wrapper_module, tree_params, tree_record, X):
 
        threading.Thread.__init__(self)
        
        self.wrapper_module = wrapper_module 
        self.tree_params = tree_params
        self.tree_record = tree_record
        self.X = X
 
    def run(self):

        self.wrapper_module.build_bufferkdtree(self.X, self.tree_record, self.tree_params)        

class BufferKDTreeNN(object):
    """ Nearest neighbor queries based on buffer k-d trees.
    """
    
    SPLITTING_TYPE_MAPPINGS = {'cyclic': 0, 'longest': 1}
    ALLOWED_FLOAT_TYPES = ['float', 'double']
    ALLOWED_USE_GPU = [True, False]

    OPENCL_ERROR_MAPPINGS = {-1:"ERROR_NO_PLATFORMS",
                             -2:"ERROR_INVALID_PLATFORMS",
                             -3:"ERROR_NO_DEVICES",
                             -4:"ERROR_INVALID_DEVICE"}

    def __init__(self, n_neighbors=5, \
                 leaf_size=2, \
                 tree_depth=None, \
                 float_type="float", \
                 splitting_type="cyclic", \
                 use_gpu=True, \
                 n_train_chunks=1, \
                 plat_dev_ids={0:[0]}, \
                 allowed_train_mem_percent_chunk=0.2, \
                 allowed_test_mem_percent=0.8, \
                 n_jobs=1, \
                 verbose=0):
        """ Model for unsupervised nearest neighbor search (buffer k-d-trees).
        
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
        self.use_gpu = use_gpu
        self.n_train_chunks = n_train_chunks
        self.plat_dev_ids = plat_dev_ids
        self.allowed_train_mem_percent_chunk = allowed_train_mem_percent_chunk
        self.allowed_test_mem_percent = allowed_test_mem_percent
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __del__(self):
        """ Free external resources if needed
        """

        try:
            if self.verbose > 0:
                print("Freeing external resources ...")
            for platform_id in self.plat_dev_ids.keys():
                for device_id in self.plat_dev_ids[platform_id]:     
                    wrapper_tree_params = self.wrapper_instances[platform_id][device_id]['tree_params']
                    wrapper_tree_record = self.wrapper_instances[platform_id][device_id]['wrapper_tree_record']            
                    self._get_wrapper_module().extern_free_resources(wrapper_tree_record, wrapper_tree_params)
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
                "tree_depth": self.tree_depth, \
                "leaf_size": self.leaf_size, \
                "float_type": self.float_type, \
                "splitting_type": self.splitting_type, \
                "use_gpu": self.use_gpu, \
                "n_train_chunks": self.n_train_chunks, \
                "plat_dev_ids": self.plat_dev_ids, \
                "allowed_train_mem_percent_chunk": self.allowed_train_mem_percent_chunk, \
                "allowed_test_mem_percent": self.allowed_test_mem_percent, \
                "n_jobs": self.n_jobs, \
                "verbose": self.verbose
                }
        
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

        final_tree_depth = self._compute_final_tree_depth(len(X), self.leaf_size, self.tree_depth)
        if self.verbose > 0:
            print("Final tree depth: " + unicode(final_tree_depth))
        
        # make sure that the array is contiguous
        # (needed for the swig module)
        X = np.ascontiguousarray(X)
        
        # convert input data to correct types (and generate local
        # variable to prevent destruction of external array)
        self.X = X.astype(self.numpy_dtype_float)
        
        self.wrapper_instances = {}
        if self.use_gpu == True:
            self._fit_gpu(final_tree_depth)
        else:
            self._fit_cpu(final_tree_depth)
            
        return self
    
    def _fit_cpu(self, final_tree_depth):
        
        wrapper_tree_params = self._get_wrapper_module().TREE_PARAMETERS()
        
        root_path = os.path.dirname(os.path.realpath(__file__))
        kernel_sources_dir = os.path.join(root_path, "../../src/neighbors/buffer_kdtree/kernels/")
                
        self._get_wrapper_module().init_extern(self.n_neighbors, final_tree_depth, \
                                               self.n_jobs, self.n_train_chunks, 0, 0, \
                                               self.allowed_train_mem_percent_chunk, \
                                               self.allowed_test_mem_percent, \
                                               self.SPLITTING_TYPE_MAPPINGS[self.splitting_type], \
                                               kernel_sources_dir, self.verbose, wrapper_tree_params)        
        wrapper_tree_record = self._get_wrapper_module().TREE_RECORD()
        
        self._get_wrapper_module().build_bufferkdtree(self.X, wrapper_tree_record, wrapper_tree_params)
        self.wrapper_instances['tree_params'] = wrapper_tree_params
        self.wrapper_instances['wrapper_tree_record'] = wrapper_tree_record    
                
    def _fit_gpu(self, final_tree_depth):        
        
        root_path = os.path.dirname(os.path.realpath(__file__))
        kernel_sources_dir = os.path.join(root_path, "../../src/neighbors/buffer_kdtree/kernels/")
                
        for platform_id in self.plat_dev_ids.keys():
            self.wrapper_instances[platform_id] = {}
            for device_id in self.plat_dev_ids[platform_id]:
                
                self._validate_device(platform_id, device_id)
                    
                wrapper_tree_params = self._get_wrapper_module().TREE_PARAMETERS()                
                self._get_wrapper_module().init_extern(self.n_neighbors, final_tree_depth, \
                                                       self.n_jobs, self.n_train_chunks, 
                                                       platform_id, device_id, \
                                                       self.allowed_train_mem_percent_chunk, \
                                                       self.allowed_test_mem_percent, \
                                                       self.SPLITTING_TYPE_MAPPINGS[self.splitting_type], \
                                                       kernel_sources_dir, self.verbose, wrapper_tree_params)
                wrapper_tree_record = self._get_wrapper_module().TREE_RECORD()
                                
                # store records
                self.wrapper_instances[platform_id][device_id] = {}
                self.wrapper_instances[platform_id][device_id]['tree_params'] = wrapper_tree_params
                self.wrapper_instances[platform_id][device_id]['wrapper_tree_record'] = wrapper_tree_record

        threads = []
        
        for platform_id in self.plat_dev_ids.keys():
            for device_id in self.plat_dev_ids[platform_id]:
                
                wrapper_module = self._get_wrapper_module()
                wrapper_tree_params = self.wrapper_instances[platform_id][device_id]['tree_params']
                wrapper_tree_record = self.wrapper_instances[platform_id][device_id]['wrapper_tree_record']
                
                thread = BuildThread(wrapper_module, wrapper_tree_params, wrapper_tree_record, self.X)
                threads.append(thread)                
            
        # start all threads
        if self.verbose > 0:
            print("Starting all build threads ...")
        for thread in threads:
            thread.start()
            
        # wait for all threads to be completed
        for thread in threads:
            thread.join()
        if self.verbose > 0:
            print("All build threads finished!")          

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
        
        if self.use_gpu == True:
            self._kneighbors_gpu(X, n_neighbors, d_mins, idx_mins)
        else:
            self._kneighbors_cpu(X, n_neighbors, d_mins, idx_mins)
                  
        return d_mins, idx_mins

    def _kneighbors_cpu(self,X, n_neighbors, d_mins, idx_mins):

        wrapper_tree_params = self.wrapper_instances['tree_params']
        wrapper_tree_record = self.wrapper_instances['wrapper_tree_record']

        self._get_wrapper_module().neighbors_extern(X, d_mins, idx_mins, \
                                                    wrapper_tree_record, \
                                                    wrapper_tree_params)
    
    def _kneighbors_gpu(self, X, n_neighbors, d_mins, idx_mins):
        
        # split up queries over all devices
        n_total_devices = self._get_n_total_devices()
        n_chunk = int(math.ceil(len(X) / n_total_devices))
        
        thread_list = []
        
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
                        wrapper_tree_params = self.wrapper_instances[platform_id][device_id]['tree_params']
                        wrapper_tree_record = self.wrapper_instances[platform_id][device_id]['wrapper_tree_record']
                        
                        if self.verbose > 0:
                            print("Initializing device thread for range %i-%i ..." % (chunk_start, chunk_end_cropped))
                        thread = DeviceQueryThread(wrapper_module, wrapper_tree_params, wrapper_tree_record, \
                                             X, d_mins, idx_mins, chunk_start, chunk_end_cropped, verbose=self.verbose)
                        thread_list.append(thread)
    
                    chunk_start += n_chunk
                    chunk_end += n_chunk
                    

        if self.verbose > 0:
            print("Processing all query thread_list ...")
        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()            
        if self.verbose > 0:
            print("All query thread_list finished!")  
    
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
        else:
            raise Exception("Unknown 'float_type'!")

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
            tree_depths = range(2, max_depth - 1)

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
            raise Exception("Warning: Tree depth has to be adapted for buffer k-d trees ...")

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
    
