#
# Copyright (C) 2013-2019 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import wget

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

    wget.download(url, fname)

