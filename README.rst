============
bufferkdtree
============

The bufferkdtree library is a Python library that aims at accelerating nearest neighbor computations using both k-d trees and many-core devices (e.g., GPUs) via the `OpenCL <https://www.khronos.org/opencl/OpenCL>`_ framework. 

The buffer k-d tree technique can be seen as an intermediate version between a standard parallel k-d tree traversal and massively-parallel brute-force implementations for nearest neigbhor search. The implementation is well-suited for data sets with a large reference set (e.g., 1,000,000 points) and a huge query set (e.g., 10,000,000 points) with a moderate-sized feature space (e.g., from d=5 to d=25).

=============
Documentation
=============

See the `documentation <http://bufferkdtree.readthedocs.org>`_ for details and examples.

==========
Quickstart
==========

The package can be installed via pip via::

  pip install bufferkdtree

To install the package from the sources, get the current version via::

  git clone https://github.com/gieseke/bufferkdtree.git

To install the package locally on a Linux system, use::

  python setup.py install --user

On Debian/Ubuntu systems, the package can be installed globally for all users via::

  python setup.py build
  sudo python setup.py install

To run the tests, type ``nosetests -v bufferkdtree`` from *outside* the source directory.

============
Dependencies
============

The bufferkdtree package is tested under Python 2.6 and Python 2.7. The required Python dependencies are:

- NumPy >= 1.6.1

Further, `Swig <http://www.swig.org>`_, `OpenCL <https://www.khronos.org/opencl/OpenCL>`_, `setuptools <https://pypi.python.org/pypi/setuptools>`_, and a working C/C++ compiler need to be available. See the `documentation <http://bufferkdtree.readthedocs.org>`_ for more details.

==========
Disclaimer
==========

The source code is published under the GNU General Public License (GPLv2). The authors are not responsible for any implications that stem from the use of this software.

