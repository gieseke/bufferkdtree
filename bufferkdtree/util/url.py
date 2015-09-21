import os
import urllib2

def download_data(url, fname):

    # create directory if needed
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)

    f = open(fname, 'wb')

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
