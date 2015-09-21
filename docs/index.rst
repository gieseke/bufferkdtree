.. -*- rst -*-

bufferkdtree
============

The bufferkdtree library is a Python library that aims at accelerating nearest neighbor computations using both k-d trees and graphics processing cards (GPUs) using `OpenCL <https://www.khronos.org/opencl/OpenCL>`_. The source code is published under the GNU General Public License (GPLv2).

Buffer k-d trees aim at scenarios, where you are given both a large reference (e.g., 100,000 points) and a huge query set (e.g., 1,000,000 or more points) with an input space of moderate dimensionality (e.g., from 4 to 20 dimensions). A description of the techniques used and an experimental evaluation of the implementation using massive astronomical data sets are provided in this `paper <http://jmlr.org/proceedings/papers/v32/gieseke14.pdf>`_.

Contents
--------
.. toctree::
   :maxdepth: 2

   install
   license
   changes

Index
-----
* :ref:`genindex`

