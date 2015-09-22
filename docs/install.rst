.. -*- rst -*-

Installation
============

Dependencies
------------

The bufferkdtree package is tested under Python 2.6, Python 2.7. To build it, the NumPy (>= 1.6.1) package, a working C/C++ compiler, and OpenCL have to be available/installed correctly.

.. admonition:: Compatibility

   The implementation is based on the efficient use of implicit hardware caches. Thus, to obtain good speed-ups, the GPU at hand has to support this feature! Current architectures such as Nvidia's Kepler architecture exhibit such caches, see, e.g., the `Kepler GK110 Whitepaper <http://www.nvidia.com/content/PDF/kepler/NVIDIA-Kepler-GK110-Architecture-Whitepaper.pdf>`_.     

Quick Installation
------------------

The package is available on `PyPI <https://pypi.python.org/pypi>`_, but can also be installed from the sources. For instance, to install the package via PyPI on Linux machines, type::

  $ sudo pip install bufferkdtree

Alternatively, you can resort to the sources::

  $ git clone https://github.com/gieseke/bufferkdtree.git
  $ cd bufferkdtree
  $ python setup.py install --user

If you want to install the package globally for all users (on Linux machines), type::

  $ sudo python setup.py build
  $ sudo python setup.py install

.. warning::

    The authors are not responsible for any implications that stem from the use of this software.

.. rubric:: Example: OpenSuse

Here are some installation instructions for OpenSuse using virtualenv and pip::

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

Cosmin out!
