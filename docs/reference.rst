.. -*- rst -*-

.. _reference:

Reference
=========

NearestNeighbors
----------------

All neighbor implementations can be invoked via the main ``NearestNeighbors`` class, which exhibits a similar structure as the corresponding `class <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`_ from the `scikit-learn <http://scikit-learn.org>`_ package:

.. autoclass:: bufferkdtree.neighbors.NearestNeighbors
    :members: fit, kneighbors

Adapting Buffer K-D Trees
-------------------------

If you wish to adapt the buffer k-d tree implementation, then you might want have a look at the C and OpenCL code that is available in the `bufferkdtree/src` directory. 

.. admonition:: Developer C API

    A documentation of the underlying C code can be found `here <capi/index.html>`_.

A good starting point for diving into the details of the underlying implementation is the base.c file in bufferkdtree/src/neighbors/buffer_kdtree directory. 


