"""
bufferkdtree
============
The bufferkdtree package is a Python library that aims at 
accelerating nearest neighbor computations using both 
k-d trees and modern many-core devices such as graphics 
processing units. The implementation is based on OpenCL. 

See the http://bufferkdtree.readthedocs.org for details.
"""

import os

# development branch marker of the form 'X.Y.dev' or 
# 'X.Y.devN' with N being an integer.
__version__ = '1.2'

from .neighbors import NearestNeighbors

__all__ = ['NearestNeighbors']
