============
bufferkdtree
============

The bufferkdtree package is a Python library that aims at accelerating nearest neighbor computations using both k-d trees and modern many-core devices such as graphics processing units (GPUs). The implementation is based on `OpenCL <https://www.khronos.org/opencl/OpenCL>`_. 

The buffer k-d tree technique can be seen as an intermediate version between a standard parallel k-d tree traversal (on multi-core systems) and a massively-parallel brute-force implementation for nearest neighbor search. In particular, it makes use of the top of a standard k-d tree (which induces a spatial subdivision of the space) and resorts to a simple yet efficient brute-force implementation for processing chunks of "big" leaves. The implementation is well-suited for data sets with a large reference set (e.g., 1,000,000 points) and a huge query set (e.g., 10,000,000 points) given a moderate dimensionality of the search space (e.g., from d=5 to d=50).

=============
Documentation
=============

See the `documentation <http://bufferkdtree.readthedocs.org>`_ for details and examples.

============
Dependencies
============

The bufferkdtree package has been tested under Python 2.6/2.7/3.*. The required Python dependencies are:

- NumPy >= 1.11.0

Further, `Swig <http://www.swig.org>`_, `OpenCL <https://www.khronos.org/opencl/OpenCL>`_, `setuptools <https://pypi.python.org/pypi/setuptools>`_, and a working C/C++ compiler need to be available. See the `documentation <http://bufferkdtree.readthedocs.org>`_ for more details.

==========
Quickstart
==========

The package can easily be installed via pip via::

  pip install bufferkdtree

To install the package from the sources, first get the current stable release via::

  git clone https://github.com/gieseke/bufferkdtree.git

Afterwards, on Linux systems, you can install the package locally for the current user via::

  python setup.py install --user

On Debian/Ubuntu systems, the package can be installed globally for all users via::

  python setup.py build
  sudo python setup.py install

==========
Disclaimer
==========

The source code is published under the GNU General Public License (GPLv2). The authors are not responsible for any implications that stem from the use of this software.
