'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

from __future__ import print_function

import os
import urllib2

def download_from_url(url, fname):
    """ Downloads data from a given url.
    
    Parameters
    ----------
    url : str
        The target url from which the data
        shall be downloaded
    fname : str
        The local filename; if the corresponding 
        directory does not exists, it will be created
    """
    
    # create directory if needed
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)

    # open local file
    f = open(fname, 'wb')

    # get data from url; based on 
    # http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python
    u = urllib2.urlopen(url)
    meta = u.info()
    fsize = int(meta.getheaders("Content-Length")[0])
    print("Downloading from %s (%i bytes) ... \n" % (url, fsize))

    fsize_current = 0
    block_size = 8192

    while True:

        buff = u.read(block_size)
        if not buff:
            break

        fsize_current += len(buff)
        f.write(buff)

        stat = r"%10d  [%3.2f%%]" % (fsize_current, fsize_current * 100. / fsize)
        stat = stat + chr(8)*(len(stat)+1)
        print(stat, end='')

    f.close()
