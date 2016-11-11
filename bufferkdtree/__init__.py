#
# Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

"""
bufferkdtree
============
The bufferkdtree package is a Python library that aims at 
accelerating nearest neighbor computations using both 
k-d trees and modern many-core devices such as graphics 
processing units. The implementation is based on OpenCL. 

See the http://bufferkdtree.readthedocs.org for details.
"""

import sys
    
# development branch marker of the form 'X.Y.dev' or 
# 'X.Y.devN' with N being an integer.
__version__ = '1.3'

try:
    __BUFFERKDTREE_SETUP__
except NameError:
    __BUFFERKDTREE_SETUP__ = False

if __BUFFERKDTREE_SETUP__:
    sys.stderr.write("Warning: Incomplete import (installation)\n")
else:
    from .neighbors import NearestNeighbors
    __all__ = ['NearestNeighbors']