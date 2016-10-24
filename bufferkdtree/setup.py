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
from distutils.util import strtobool

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    try:
        kdtree_only = strtobool(os.environ['BUFFERKDTREE_KDTREE_ONLY'])
    except:
        kdtree_only = False
        
    config = Configuration('bufferkdtree', parent_package, top_path)
    config.add_subpackage('neighbors', subpackage_path='neighbors')
    config.add_subpackage('neighbors/kdtree', subpackage_path='neighbors/kdtree')
    if kdtree_only == False:
        config.add_subpackage('neighbors/brute', subpackage_path='neighbors/brute')
        config.add_subpackage('neighbors/buffer_kdtree', subpackage_path='neighbors/buffer_kdtree')
    config.add_subpackage('tests')
    config.add_subpackage('util')
    
    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())