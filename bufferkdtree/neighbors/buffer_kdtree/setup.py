'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

import os
import numpy

TIMING = 1
WORKGROUP_SIZE_BRUTE = 256
WORKGROUP_SIZE_LEAVES = 32
WORKGROUP_SIZE_UPDATE = 16
WORKGROUP_SIZE_COPY_INIT = 32
WORKGROUP_SIZE_COMBINE = 64
WORKGROUP_SIZE_TEST_SUBSET = 32
WORKGROUP_SIZE_COPY_DISTS_INDICES = 32

FILES_TO_BE_COMPILED = ["neighbors/buffer_kdtree/base.c", \
                        "neighbors/buffer_kdtree/cpu.c", \
                        "neighbors/buffer_kdtree/gpu_opencl.c", \
                        "neighbors/buffer_kdtree/util.c", \
                        "neighbors/buffer_kdtree/kdtree.c", \
                        "timing.c", \
                        "util.c", \
                        "opencl.c" \
                       ]
DIRS_TO_BE_INCLUDED = ["neighbors/buffer_kdtree/include"]

# paths
SOURCES_RELATIVE_PATH = "../../src/"
current_path = os.path.dirname(os.path.abspath(__file__))
sources_abs_path = os.path.abspath(os.path.join(current_path, SOURCES_RELATIVE_PATH))

# source files
source_files = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in FILES_TO_BE_COMPILED] 
include_paths = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in DIRS_TO_BE_INCLUDED]

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration
    config = Configuration('neighbors/buffer_kdtree', parent_package, top_path)

    # CPU + FLOAT
    config.add_extension("_wrapper_cpu_float", \
                                    sources=["swig/cpu_float.i"] + source_files,
                                    swig_opts=['-modern', '-threads'],
                                    include_dirs=[numpy_include] + [include_paths],
                                    define_macros=[
                                        ('ABSOLUTE_PATH', os.path.join(sources_abs_path, "neighbors/buffer_kdtree")),
                                        ('USE_DOUBLE', 0),
                                        ('TIMING', TIMING)
                                    ],
                                    libraries=['OpenCL', 'gomp', 'm'],
                                    extra_compile_args=["-fopenmp", '-O3', '-Wall'] + ['-I' + ipath for ipath in include_paths])

    # CPU + DOUBLE
    config.add_extension("_wrapper_cpu_double", \
                                    sources=["swig/cpu_double.i"] + source_files,
                                    swig_opts=['-modern', '-threads'],
                                    include_dirs=[numpy_include] + [include_paths],
                                    define_macros=[
                                        ('ABSOLUTE_PATH', os.path.join(sources_abs_path, "neighbors/buffer_kdtree")),
                                        ('USE_DOUBLE', 1),
                                        ('TIMING', TIMING)
                                    ],
                                    libraries=['OpenCL', 'gomp', 'm'],
                                    extra_compile_args=["-fopenmp", '-O3', '-Wall'] + ['-I' + ipath for ipath in include_paths])

    # GPU + FLOAT
    config.add_extension("_wrapper_gpu_opencl_float", \
                                    sources=["swig/gpu_float.i"] + source_files,
                                    swig_opts=['-modern', '-threads'],
                                    include_dirs=[numpy_include] + [include_paths],
                                    define_macros=[
                                        ('ABSOLUTE_PATH', os.path.join(sources_abs_path, "neighbors/buffer_kdtree")),
                                        ('USE_GPU', 1),
                                        ('USE_DOUBLE', 0),
                                        ('TIMING', TIMING),
                                        ('WORKGROUP_SIZE_BRUTE', WORKGROUP_SIZE_BRUTE),
                                        ('WORKGROUP_SIZE_LEAVES', WORKGROUP_SIZE_LEAVES),
                                        ('WORKGROUP_SIZE_UPDATE', WORKGROUP_SIZE_UPDATE),
                                        ('WORKGROUP_SIZE_COPY_INIT', WORKGROUP_SIZE_COPY_INIT),
                                        ('WORKGROUP_SIZE_COMBINE', WORKGROUP_SIZE_COMBINE),
                                        ('WORKGROUP_SIZE_TEST_SUBSET', WORKGROUP_SIZE_TEST_SUBSET),
                                        ('WORKGROUP_SIZE_COPY_DISTS_INDICES', WORKGROUP_SIZE_COPY_DISTS_INDICES),
                                    ],
                                    libraries=['OpenCL', 'gomp'],
                                    extra_compile_args=["-fopenmp", '-O3', '-Wall'] + ['-I' + ipath for ipath in include_paths])

    # GPU + DOUBLE
    config.add_extension("_wrapper_gpu_opencl_double", \
                                    sources=["swig/gpu_double.i"] + source_files,
                                    swig_opts=['-modern', '-threads'],
                                    include_dirs=[numpy_include] + [include_paths],
                                    define_macros=[
                                        ('ABSOLUTE_PATH', os.path.join(sources_abs_path, "neighbors/buffer_kdtree")),
                                        ('USE_GPU', 1),
                                        ('USE_DOUBLE', 1),
                                        ('TIMING', TIMING),
                                        ('WORKGROUP_SIZE_BRUTE', WORKGROUP_SIZE_BRUTE),
                                        ('WORKGROUP_SIZE_LEAVES', WORKGROUP_SIZE_LEAVES),
                                        ('WORKGROUP_SIZE_UPDATE', WORKGROUP_SIZE_UPDATE),
                                        ('WORKGROUP_SIZE_COPY_INIT', WORKGROUP_SIZE_COPY_INIT),
                                        ('WORKGROUP_SIZE_COMBINE', WORKGROUP_SIZE_COMBINE),
                                        ('WORKGROUP_SIZE_TEST_SUBSET', WORKGROUP_SIZE_TEST_SUBSET),
                                        ('WORKGROUP_SIZE_COPY_DISTS_INDICES', WORKGROUP_SIZE_COPY_DISTS_INDICES),
                                    ],
                                    libraries=['OpenCL', 'gomp'],
                                    extra_compile_args=["-fopenmp", '-O3', '-Wall'] + ['-I' + ipath for ipath in include_paths])

    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

