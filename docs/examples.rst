.. -*- rst -*-

Examples
========

The following two examples sketch the use of the different implementations and can both be found in the *examples* subdirectory of the bufferkdtree package.

Toy Example
-----------

.. literalinclude:: ../examples/artificial.py
    :start-after: # Licence: GNU GPL (v2)
    :end-before: X = numpy.random.uniform(low=-1, high=1, size=(10000,10))

All implementations are provided via the ``NearestNeighbors`` class, which exhibits a similar layout as the corresponding class of the `scikit-learn <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`_ package. The parameter ``n_jobs`` determines the number of threads that shall be used by the standard k-d tree implementation (CPU). The parameter ``plat_dev_ids`` determines the OpenCL devices that shall be used by the buffer k-d tree implementation (OpenCL): Each key of the dictionary corresponds to a OpenCL platform id and for each platform id, a list of associated device ids can be provided. For instance, the first platform (with id 0) and its first device (with id 0) is used for the current example.

Next, a small artificial data set is generated, where ``X`` contains the points, one row per point:

.. literalinclude:: ../examples/artificial.py
    :start-after: verbose = 0
    :end-before: # (1) apply buffer k-d tree implementation

The package provides three implementations (``brute``, ``kd_tree``, or ``buffer_kd_tree``), which can be invoked via the ``algorithm`` keyword of the constructor:

.. literalinclude:: ../examples/artificial.py
    :start-after: X = numpy.random.uniform(low=-1, high=1, size=(10000,10))

For a detailed description of the remaining keywords, see the description of the :ref:`documentation <reference>` of the NearestNeighbors class. The above steps yield the following output::

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


.. admonition:: Brute-Force

    Note that the brute-force implementatation is only used for comparison purposes given data sets in relatively low-dimensional search spaces. Its performance is suboptimal for high-dimensional feature spaces compared to matrix-based implementations that make use of e.g. CUBLAS (but superior to such implementations given low-dimensional search spaces).

Large-Scale Querying
--------------------

The main purpose of the buffer k-d tree implementation is to speed up the querying phase given both a large number of reference and a huge number of query points. The next data example is based on astronomical data from the `Sloan Digital Sky Survey <http://www.sdss.org/collaboration/citing-sdss>`_ (the data set will be downloaded automatically):

.. literalinclude:: ../examples/astronomy.py
    :start-after: # Licence: GNU GPL (v2)
    :end-before: def run_algorithm(algorithm="buffer_kd_tree", tree_depth=None, leaf_size=None):

Note that four devices (with ids 0,1,2,3) of the first platform (with id 0) are used in this case. The helper function defined next is used to time the runtimes needed for the training and testing phases of each method:

.. literalinclude:: ../examples/astronomy.py
    :start-after: n_neighbors=10
    :end-before: print("Parsing data ...")

Note that either ``tree_depth`` or ``leaf_size`` is used to determine the final tree depth, see the :ref:`documentation <reference>`. For this example, large sets of reference (two million) and query points (ten million) are generated: 

.. literalinclude:: ../examples/astronomy.py
    :start-after:     print("Testing time: %f" % (end_time-start_time))
    :end-before: print("----------------------------------------------------------------------")

Loading the data this way should yield an output like::

    Nearest Neighbors
    =================

    This example demonstrates the use of both tree-based
    implementations on a large-scale data set.

    -------------------------------- DATA --------------------------------
    Number of training patterns: 2000000
    Number of test patterns:	 10000000
    Dimensionality of patterns:	 10
    ----------------------------------------------------------------------

Finally, both implementations are invoked to compute the 10 nearest neighbors for each query point:

.. literalinclude:: ../examples/astronomy.py
    :start-after: print("----------------------------------------------------------------------")

The above code yields the folling output on an *Ubuntu 14.04* system (64 bit) with an *Intel(R) Core(TM) i7-4790K* running at 4.00GHz (4 cores, 8 hardware threads), 32GB RAM, two *Geforce Titan Z* GPUs (with two devices each), CUDA 6.5 and Nvidia driver version 340.76::

    Running the GPU version ...
    Fitting time: 1.394939
    Testing time: 11.148126

    Running the CPU version ... 
    Fitting time: 0.681938
    Testing time: 314.787735

The parameters ``tree_depth`` and ``leaf_size`` play an important role: In case ``tree_depth`` is not ``None``, then ``leaf_size`` is ignored. Otherwise, ``leaf_size`` is used to automatically determine the corresponding tree depth (such that at most ``leaf_size`` points are stored in a single leaf). For ``kd_tree``, setting the leaf size to, e.g., 32 is usually a good choice. For ``buffer_kd_tree``, a smaller tree depth is often needed to achieve a good performance (e.g., ``tree_depth=9`` for 1,000,000 reference points).

.. admonition:: Performance

    The performance might depend on the particular OpenCL version Nvidia driver. For instance, we observed similar speed-ups (per device) with a weeker Gefore GTX 770 given CUDA 5.5 and Nvidia driver version 319.23. 

.. admonition:: Tree Construction

    Both implementations are based on the standard rule for splitting nodes during the construction (cyclic, median based). Other splitting rules might be beneficial, but are, in general, data set dependent. Other construction schemes will be available in future for all tree-based schemes.




