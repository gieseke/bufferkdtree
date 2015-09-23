'''
Created on 15.09.2015

@author: fgieseke
'''

import os
import urllib2

def download_from_url(url, fname):
    """ Downloads data from a given url.
    
    Parameters
    ----------
    url : str
        The target url
    fname : str
        The local filename
    """
    
    # create directory if needed
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)

    # open local file
    f = open(fname, 'wb')

    # get data from url
    u = urllib2.urlopen(url)
    meta = u.info()
    fsize = int(meta.getheaders("Content-Length")[0])
    print("Downloading from %s ... (%i bytes)\n" % (url, fsize))

    fsize_dl = 0
    block_sz = 8192

    while True:

        bu = u.read(block_sz)
        if not bu:
            break

        fsize_dl += len(bu)
        f.write(bu)

        stat = r"%10d  [%3.2f%%]" % (fsize_dl, fsize_dl * 100. / fsize)
        stat = stat + chr(8)*(len(stat)+1)
        print stat,

    f.close()
