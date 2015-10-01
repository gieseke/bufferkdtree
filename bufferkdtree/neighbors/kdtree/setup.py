'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

import os
import numpy

SOURCES_RELATIVE_PATH = "../../src/"

FILES_TO_BE_COMPILED = ["neighbors/kdtree/base.c", "neighbors/kdtree/util.c", "neighbors/kdtree/kdtree.c", "timing.c", "util.c"]
DIRS_TO_BE_INCLUDED = ["neighbors/kdtree/include"]

# the absolute path to the sources
current_path = os.path.dirname(os.path.abspath(__file__))
sources_abs_path = os.path.abspath(os.path.join(current_path, SOURCES_RELATIVE_PATH))

# all source files
source_files = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in FILES_TO_BE_COMPILED] 
include_paths = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in DIRS_TO_BE_INCLUDED]

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('neighbors/kdtree', parent_package, top_path)

    # CPU + FLOAT
    config.add_extension("_wrapper_cpu_float", \
                                    sources=["swig/cpu_float.i"] + source_files,
                                    swig_opts=['-modern'],
                                    include_dirs=[numpy_include] + [include_paths],
                                    define_macros=[
                                        ('USE_DOUBLE', 0),
                                        ('TIMING', 1)
                                    ],
                                    libraries=['gomp', 'm'],
                                    extra_compile_args=["-fopenmp", '-O3', '-Wall'] + ['-I' + ipath for ipath in include_paths])

    # CPU + DOUBLE
    config.add_extension("_wrapper_cpu_double", \
                                    sources=["swig/cpu_double.i"] + source_files,
                                    swig_opts=['-modern'],
                                    include_dirs=[numpy_include] + [include_paths],
                                    define_macros=[
                                        ('USE_DOUBLE', 1),
                                        ('TIMING', 1)
                                    ],
                                    libraries=['gomp', 'm'],
                                    extra_compile_args=["-fopenmp", '-O3', '-Wall'] + ['-I' + ipath for ipath in include_paths])

    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

