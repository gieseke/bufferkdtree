.. -*- rst -*-

Installation
============

Dependencies
------------

The bufferkdtree package has been tested under various Linux-based systems such as Ubuntu and OpenSUSE and requires Python 2.6 or 2.7. Below, some installation instructions will be given for Linux-based systems; similar steps have to be conducted on other systems.

To install the package, a working C/C++ compiler, `OpenCL <https://www.khronos.org/opencl/OpenCL>`_, `Swig <http://www.swig.org/>`_, and the Python development package (header files) need to be available. Further, the `NumPy <http://www.numpy.org>`_ package (>=1.6.1) is needed.

On Ubuntu 12.04, for instance, the following command installs all dependencies (except for OpenCL)::

   $ sudo apt-get install python2.7 swig build-essential python-numpy

On an OpenSUSE system, the corresponding commands are::

   $ sudo zypper install python python-devel swig

.. admonition:: Compatibility

   The implementation is based on the efficient use of implicit hardware caches. Thus, to obtain good speed-ups, the GPU at hand has to support this feature! Current architectures such as Nvidia's Kepler architecture exhibit such caches, see, e.g., the `Kepler GK110 Whitepaper <http://www.nvidia.com/content/PDF/kepler/NVIDIA-Kepler-GK110-Architecture-Whitepaper.pdf>`_. 

OpenCL
------

OpenCL needs to be installed correctly. Make sure that the OpenCL header files are available, for example by setting the C_INCLUDE_PATH environment variable in the .bashrc file on Linux systems. For instance, in case CUDA is installed with header files being located in ``/usr/local/cuda/include``, then the following command should update the environment variable::

   export C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/local/cuda/include

Quick Installation
------------------

.. warning::

    The authors are not responsible for any implications that stem from the use of this software.

The package is available on `PyPI <https://pypi.python.org/pypi>`_, but can also be installed from the sources. For instance, to install the package via `PyPI <https://pypi.python.org/pypi>`_ on Linux machines, type::

  $ sudo pip install bufferkdtree

To install the package from the sources, first get the current version via ::

  $ git clone https://github.com/gieseke/bufferkdtree.git

Subsequently, install the package locally via::

  $ cd bufferkdtree
  $ python setup.py install --user

or, globally for all users, via::

  $ sudo python setup.py build
  $ sudo python setup.py install




Virtualenv & Pip
----------------

We recommend to install the package via virtualenv and pip. On Ubuntu 12.04, for instance, the following commands can be used to install virtualenv and pip::

   $ sudo apt-get install python-virtualenv python-pip

Afterwards, create a new virtual environment and install the Numpy package::

   $ mkdir ~/.virtualenvs
   $ cd ~/.virtualenvs
   $ virtualenv bufferkdtree
   $ source bufferkdtree/bin/activate
   $ pip install numpy==1.6.1

Given the activated virtual environment, follow the instructions above to install the bufferkdtree package.

    



