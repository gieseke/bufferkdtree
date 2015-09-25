.. -*- rst -*-

Examples
========

The following two examples demonstrate the use of the different implementations.

Toy Example
-----------

.. literalinclude:: ../examples/artificial.py
    :start-after: # Licence: GNU GPL (v2)
    :end-before: X = numpy.random.uniform(low=-1, high=1, size=(10000,10))

All implementations are provided via the ``NearestNeighbors`` class, which exhibits a similar layout as the corresponding `class <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`_ from the `scikit-learn <http://scikit-learn.org>`_ package. The parameter ``n_jobs`` determines the number of threads that shall be used by the standard k-d tree implementation (CPU). The parameter ``plat_dev_ids`` determines the OpenCL devices that shall be used by the many-core buffer k-d tree implementation: Each key of the dictionary corresponds to a platform id and for each platform id, a list of associated device ids can be provided.  In this case, we are using platform 0 and the first device.



.. literalinclude:: ../examples/artificial.py
    :start-after: verbose = 0
    :end-before: # (1) apply buffer k-d tree implementation

All approaches are executed on a small artificial toy example; here, ``X`` contains the points (each row corresponds to a point):

.. literalinclude:: ../examples/artificial.py
    :start-after: X = numpy.random.uniform(low=-1, high=1, size=(10000,10))

The parameter ``algorithm`` specifies the method thall shall be used (``brute``, ``kd_tree``, or ``buffer_kd_tree``). The above steps yield the following output::

    Nearest Neighbors
    =================

    This example demonstrates the use of the different 
    implementations given on a small artifical data set.


    buffer_kd_tree output
    [ 0.          1.0035212   1.09866345  1.11734533  1.13440645  1.17730558
      1.1844281   1.20736992  1.2085104   1.21593559]

    brute output
    [ 0.          1.0035212   1.09866357  1.11734521  1.13440645  1.17730546
      1.18442798  1.20736992  1.2085104   1.21593571]

    kd_tree output
    [ 0.          1.0035212   1.09866357  1.11734521  1.13440645  1.17730546
      1.18442798  1.20736992  1.2085104   1.21593571]

Large-Scale Querying
--------------------

The main purpose of the buffer k-d tree implementation is to speed up the querying phase, given a large number of reference points. The next data example is based on data from the `Sloan Digital Sky Survey <http://www.sdss.org>`_ (the data set will be downloaded automatically, see the `copyright notice <http://www.sdss.org/collaboration/citing-sdss>`_):

.. literalinclude:: ../examples/astronomy.py
    :start-after: # Licence: GNU GPL (v2)
    :end-before: def run_algorithm(algorithm="buffer_kd_tree", tree_depth=None, leaf_size=None):

Note that we are now using the OpenCL platform 0 with four devices 0,1,2, and 3. The helper function defined next is used to time the runtimes needed for the training and testing phases of each method:

.. literalinclude:: ../examples/astronomy.py
    :start-after: n_neighbors=10
    :end-before: # get/download data

Note that either ``tree_depth`` or ``leaf_size`` is used to determine the final tree depth of the involved trees (see below). For this example, 2,000,000 training/reference and 10,000,000 testing/querying points are used:

.. literalinclude:: ../examples/astronomy.py
    :start-after: # get/download data
    :end-before: print "----------------------------------------------------------------------"

The output should like this::

    Nearest Neighbors
    =================

    This example demonstrates the use of both tree-based
    implementations on a large-scale data set.

    -------------------------------- DATA --------------------------------
    Number of training patterns: 2000000
    Number of test patterns:	 10000000
    Dimensionality of patterns:	 10
    ----------------------------------------------------------------------

Finally, both implementations are used to compute the neighbors for the loaded data:

.. literalinclude:: ../examples/astronomy.py
    :start-after: print "----------------------------------------------------------------------"

On a Ubuntu 14.04 system with an Intel(R) Core(TM) i7-4790K running at 4.00GHz (4 cores, 8 hardware threads), 32GB RAM, two Geforce Titan Z GPUs (with two devices each), CUDA 6.5 and Nvidia driver version 340.76, the above code yields::

    Running the GPU version ...
    Fitting time: 1.394939
    Testing time: 11.148126

    Running the CPU version ... 
    Fitting time: 0.681938
    Testing time: 314.787735

The parameters ``tree_depth`` and ``leaf_size`` play an important role for the performance. Note that in case ``tree_depth`` is set, then ``leaf_size`` is ignored. Otherwise, ``leaf_size`` is used to automatically determine the associated tree depth. For ``kd_tree``, setting the leaf size to, e.g., 32 is usually a good choice. For ``buffer_kd_tree``, a smaller tree depth is often needed to achieve a good performance (e.g., ``tree_depth=9`` for 1,000,000 training points).

.. admonition:: Performance

    The performance might depend on the particular OpenCL version Nvidia driver. For instance, we observed similar speed-ups (per device) with a weeker Gefore GTX 770 given CUDA 5.5 and Nvidia driver version 319.23. 

.. admonition:: Tree Construction

    Both implementations are based on the standard rule for splitting nodes during the construction (cyclic, median based). Other splitting rules might be beneficial, but are, in general, data set dependent. Other construction schemes will be available in future for all tree-based schemes.




