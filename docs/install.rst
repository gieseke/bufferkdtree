.. -*- rst -*-

Installation
============

Dependencies
------------

The bufferkdtree package has been tested under Ubuntu and OpenSUSE linux distributions. It requires that Python 2.6 or 2.7, a working C/C++ compiler, and OpenCL are already installed. In addition it requires several other packages, such as the development packages of Python (for header files), NumPy, pip, swig, and python-virtualenv. We recommand using the latter for installing NumPy (>=1.6.1).

For Ubuntu, these can be installed with the command::

   $ sudo apt-get install python2.7 python-virtualenv python-dev python-pip swig

For OpenSUSE, these can be installed with the command::

   $ sudo zypper install python python-virtualenv python-devel python-pip swig

Finally, one can create and activate a new python environment in which numpy1.6.1 is installed by::

   $ mkdir ~/.virtualenvs
   $ cd ~/.virtualenvs
   $ mkdir bufferkdtree
   $ cd bufferkdtree

   # creates a new python environment
   $ virtualenv bufferkdtree_master

   # activate environment and install numpy
   $ source bufferkdtree_master/bin/activate
   $ pip install numpy==1.6.1

.. admonition:: Compatibility

   The implementation is based on the efficient use of implicit hardware caches. Thus, to obtain good speed-ups, the GPU at hand has to support this feature! Current architectures such as Nvidia's Kepler architecture exhibit such caches, see, e.g., the `Kepler GK110 Whitepaper <http://www.nvidia.com/content/PDF/kepler/NVIDIA-Kepler-GK110-Architecture-Whitepaper.pdf>`_.     

Quick Installation
------------------

First, make sure that the OpenCL header files are available, for example by setting the C_INCLUDE_PATH environment variable in the .bashrc file::

   # make the OpenCL header files available, for example on a CUDA system
   # PATH_TO_OPENCL_INCLUDE_FOLDER could be /usr/local/cuda/include
   export C_INCLUDE_PATH=PATH_TO_OPENCL_INCLUDE_FOLDER:$C_INCLUDE_PATH

The package is available on `PyPI <https://pypi.python.org/pypi>`_, but can also be installed from the sources. For instance, to install the package via PyPI on Linux machines, type::

  $ sudo pip install bufferkdtree

Alternatively, you can resort to the sources::

  $ git clone https://github.com/gieseke/bufferkdtree.git
  $ cd bufferkdtree
  $ python setup.py install --user

If you want to install the package globally for all users (on Linux machines), type::

  $ sudo python setup.py build
  $ sudo python setup.py install

To run the program, one may enter the examples folder and execute one of the python programs there::

  $ cd examples
  $ python bigastronomy.py

Previous to running the example, one should modify the python program, e.g., bigastronomy.py, in order to set::

   plat_dev_ids={0:[0,1]}

if there are two available GPGPUs (devices 0 and 1), and, at the very end, one should also set n_jobs to the number of parallel threads used for multi-core execution, for example assuming 32 threads are desired::

   run_algorithm(algorithm="kd_tree", leaf_size=32, n_jobs=32)

   

.. warning::

    The authors are not responsible for any implications that stem from the use of this software.
