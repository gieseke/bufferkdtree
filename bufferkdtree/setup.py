'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('bufferkdtree', parent_package, top_path)
    config.add_subpackage('neighbors', subpackage_path='neighbors')
    config.add_subpackage('neighbors/brute', subpackage_path='neighbors/brute')
    config.add_subpackage('neighbors/kdtree', subpackage_path='neighbors/kdtree')
    config.add_subpackage('neighbors/buffer_kdtree', subpackage_path='neighbors/buffer_kdtree')
    config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
