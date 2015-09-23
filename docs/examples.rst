.. -*- rst -*-

Examples
========

The following two examples demonstrate the use of the different implementations.

Toy Example
-----------

.. literalinclude:: ../examples/artificial.py
    :start-after: # Licence: GNU GPL (v2)
    :end-before: X = numpy.random.uniform(low=-1, high=1, size=(10000,10))

Here, ``plat_dev_ids`` determines the OpenCL devices that shall be used. Each key of the dictionary corresponds to a platform id and for each platform id, a list of associated device ids can be provided.  In this case, we are using platform 0 and the first device.

.. literalinclude:: ../examples/artificial.py
    :start-after: verbose = 0
    :end-before: # (1) apply buffer k-d tree implementation

Small data set is generated. Next, three different implementations are invoked:

.. literalinclude:: ../examples/artificial.py
    :start-after: X = numpy.random.uniform(low=-1, high=1, size=(10000,10))

The parameter ``algorithm`` determines the method that shall be used. The following output is produced::

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

The next data example is based on data from the Sloan Digital Sky Survey; the data is downloaded automatically.

.. literalinclude:: ../examples/astronomy.py
    :start-after: # Licence: GNU GPL (v2)
    :end-before: def run_algorithm(algorithm="buffer_kd_tree", tree_depth=None, leaf_size=None):

Note that we are now using platform 0 with four devices (0,1,2, and 3). Next, a helper function is defined to time the runtimes needed for the training and testing phases:

.. literalinclude:: ../examples/astronomy.py
    :start-after: n_neighbors=10
    :end-before: # get/download data

Note that either ``tree_depth`` or ``leaf_size`` is used to determine the final tree depth of the involved trees. Next, a bigger data set is downloaded automatically:

.. literalinclude:: ../examples/astronomy.py
    :start-after: # get/download data
    :end-before: print("\n\nRunning the GPU version ...\n")

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

On a Ubuntu 14.04 system with an ``Intel(R) Core(TM) i7-4790K`` running at 4.00GHz (4 cores, 8 hardware threads), 32GB RAM, two ``Geforce Titan Z GPUs`` (with two devices each), CUDA 6.5 and Nvidia driver version 340.76, the output is::

    Running the GPU version ...
    Fitting time: 1.394939
    Testing time: 11.148126

    Running the CPU version ... 
    Fitting time: 0.681938
    Testing time: 314.787735

.. admonition:: Performance

    The performance might depend on the particular OpenCL version Nvidia driver. For instance, we observed similar speed-ups (per device) with a weeker Gefore GTX 770 given CUDA 5.5 and Nvidia driver version 319.23. Also, both implementations are based on the standard rule for splitting nodes during the construction (cyclic, median based). Other splitting rules might be beneficial and are generally data set dependent.



