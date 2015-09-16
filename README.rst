bufferkdtree
============

The bufferkdtree library is a Python library that aims at accelerating nearest neighbor computations using both k-d trees and graphics processing cards (GPUs) using `OpenCL <https://www.khronos.org/opencl/OpenCL>`_. The source code is published under the GNU General Public License (GPLv2).

Buffer k-d trees aim at scenarios, where you are given both a large reference (e.g., 100,000 points) and a huge query set (e.g., 1,000,000 or more points) with an input space of moderate dimensionality (e.g., from 4 to 20 dimensions). A description of the techniques used and an experimental evaluation of the implementation using massive astronomical data sets are provided in this `paper <http://jmlr.org/proceedings/papers/v32/gieseke14.pdf>`_.

Compatibility
=============

The implementation is based on the efficient use of implicit hardware caches. Thus, to obtain good speed-ups, the GPU at hand has to support this feature! Current architectures such as Nvidia's Kepler architecture exhibit such caches, see, e.g., the `Kepler GK110 Whitepaper <http://www.nvidia.com/content/PDF/kepler/NVIDIA-Kepler-GK110-Architecture-Whitepaper.pdf>`_.

Dependencies
============

The bufferkdtree package is tested under Python 2.6, Python 2.7. To build the package, NumPy (>= 1.6.1) a working C/C++ compiler are needed. Further, OpenCL needs to be installed correctly.

Installation
============

The bufferktree package is available on PyPI, but can also be installed from the sources. To install it from PyPI, type::

  sudo pip install bufferkdtree

To install it from a working copy for a single user, type::

  python setup.py install --user

or, if you want to install the package globally for all users on Linux, type::

  python setup.py build
  sudo python setup.py install
  
  
Example: OpenSuse
-----------------

Here are some small installation instructions for OpenSuse using virtualenv and pip::

   # dependences: OpenCL installed
   sudo zypper install python-virtualenv python-devel python-pip swig
   
   mkdir ~/.virtualenvs
   cd ~/.virtualenvs
   mkdir bufferkdtree
   cd bufferkdtree

   # creates a new python environment
   virtualenv bufferkdtree_master

   # activate environment and install numpy
   source bufferkdtree_master/bin/activate
   pip install numpy==1.6.1

   # make the OpenCL header files available, for example with:
   export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH

   # get and install sources
   git clone https://github.com/gieseke/bufferkdtree.git
   cd bufferkdtree
   python setup.py install

   # execute an example
   python examples/neighbors.py

Notes
=====

The performance might depend on the particular OpenCL version (and driver). For instance, the results mentioned above were obtained on Ubuntu 12.04 (64 Bit) with kernel 3.8.0-44-generic, CUDA 5.5, and NVIDIA driver 319.23. The performance might be different on the same system using a different setup (e.g., using Ubuntu 14.04 (64 Bit) with kernel 3.13.0-36-generic, CUDA 6.5, and NVIDIA driver 340.29, the performance drops by about 30%). 

Citations
=========
 
If you wish to cite a paper that describes the techniques and the implementation for buffer k-d trees, please make use of the following work:

Fabian Gieseke, Justin Heinermann, Cosmin Oancea, and Christian Igel. Buffer k-d Trees: Processing Massive Nearest Neighbor Queries on GPUs. In Proceedings of the 31st International Conference on Machine Learning (ICML) 32(1), 2014, 172-180. [`pdf <http://jmlr.org/proceedings/papers/v32/gieseke14.pdf>`_]

Disclaimer
==========

The source code is published under the GNU General Public License (GPLv2). The author is not responsible for any implications that stem from the use of this software.



