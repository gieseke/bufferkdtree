.. -*- rst -*-

Quick Overview
==============

The main approach provided by the bufferkdtree package is an efficient many-core (e.g., GPU) implementation for processing huge amounts of nearest neighor queries by means of so-called buffer k-d trees. Such trees depict modifications of standard k-d trees that make use of the massive parallelism provided by today's many-core devices (such as GPUs) to process the leaves of the tree. 

Buffer k-d trees aim at scenarios, where you are given both a large reference (e.g., 1,000,000 points) and a huge query set (e.g., 10,000,000 or more points) with an input space of moderate dimensionality (e.g., from 4 to 20 dimensions). 

.. image:: _static/images/bufferkdtree.png
   :width: 500 px
   :align: center
   :alt: map to buried treasure

**Workflow:** In each iteration, the procedure *FindLeafBatch* removes query indices from both queues and distributes them to the buffers (or removes them if no further processing is needed). In case enough work has been gathered, the procedure *ProcessAllBuffers* is invoked, which updates the nearest neighbors and reinserts all query indices into *reinsert*. The process stops as soon as both queues and all buffers are empty.

.. admonition:: Implicit Hardware Caches

   The brute-force step that takes place to empty the leaves via the many-core device makes use of implicit hardware caches. To achieve satisfying speed-ups, this feature has to be supported by the device (see, e.g., the `Kepler GK110 Whitepaper <http://www.nvidia.com/content/PDF/kepler/NVIDIA-Kepler-GK110-Architecture-Whitepaper.pdf>`_)

A detailed description of the techniques used and an experimental evaluation of the implementation using massive astronomical data sets are provided in this `paper <http://jmlr.org/proceedings/papers/v32/gieseke14.pdf>`_.

