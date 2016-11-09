.. -*- rst -*-

.. _reference:

Reference
=========

NearestNeighbors
----------------

All neighbor implementations can be invoked via the main ``NearestNeighbors`` class, which exhibits a similar structure as the corresponding `class <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`_ from the `scikit-learn <http://scikit-learn.org>`_ package:

.. autoclass:: bufferkdtree.neighbors.NearestNeighbors
    :members: fit, kneighbors, compute_optimal_tree_depth

Adapting Buffer K-D Trees
-------------------------

.. admonition:: C API

    A separate documentation of this code can be found in bufferkdtree/docs/c_api subdirectory and is also available `online <https://github.com/gieseke/bufferkdtree/tree/master/docs/c_api>`_

If you wish to adapt the buffer k-d tree implementation, then you might want have a look at the C and OpenCL code that is available in the `bufferkdtree/src` directory. 

A good starting point for diving into the details of the underlying implementation is the base.c file in bufferkdtree/src/neighbors/buffer_kdtree directory. 

Within this file, you can find two functions `build_bufferkdtree` and `neighbors_extern`: The first function is called via the Python interface to build a buffer k-d tree given a set of reference/training points. These points are stored in the array `FLOAT_TYPE * Xtrain` (one row per training pattern).

.. admonition:: FLOAT_TYPE

   FLOAT_TYPE is either `float` or `double` depending on the compiler flag `USE_DOUBLE`, which is defined in `bufferkdtree/neighbors/buffer_kdtree/setup.py`.

The corresponding tree is stored in the `tree_record` structure. All other parameters are defined via the params structure, which is initialized beforehand via an outer call of `init_extern`.

The second function `neighbors_extern` can be called via the Python interface to compute the nearest neighbors for a buffer k-d tree already built via the first function: Here, `Xtest` corresponds to the test/query points. The distances and indices that are computed during the execution are stored in `distances` and `indices`, respectively. Similarly to `build_bufferkdtree`, the parameters are stored in `params` and the (already built tree) in `tree`.

The key idea of buffer k-d trees is to speed up the computation of nearest neighbors given many test queries. The two main algorithmic building blocks are two functions, `ProcessAllBuffers` and `FindLeafBatch`, that are called in an alternating fashion until all queries have been processed. For the algorithmic details, please have a look at 

.. admonition:: Details: Buffer K-D Tree 

    Fabian Gieseke, Justin Heinermann, Cosmin Oancea, and Christian Igel. *Buffer k-d Trees: Processing Massive Nearest Neighbor Queries on GPUs*. In: Proceedings of the 31st International Conference on Machine Learning (ICML) 32(1), 2014, 172-180.  [`pdf <http://jmlr.org/proceedings/papers/v32/gieseke14.pdf>`_] [`bibtex <_static/bibtex/GiesekeHOI2014.bib>`_]

For each of these two functions, a corresponding implementation in gpu_opencl.c can be found(`process_all_buffers_gpu` and `find_leaf_idx_batch_gpu`). The underlying workflow is quite complex; please get in touch with us in case you have trouble understanding certain parts of the code. 

If you are just interested in the OpenCL part, then you might want to have a look at `kernels` subdirectory: The most important kernel is given in `brute_all_leaves_nearest_neighbors.cl`, which implements the brute-force processing of nearest neighbors in the leaves of the buffer k-d tree. The kernel taking care of finding the next leaves is given in `find_leaves_idx_batch_float.cl`.




