import os
import sys
import numpy
from bufferkdtree.util.url import download_from_url
from bufferkdtree.util.input import ask_question

def psf_model_mag_train(NUM=None):

    fname = os.path.join(os.path.dirname(__file__), 'data', 'photometric_spec_confirmed.npy')
    q = "Additional data need to be downloaded (to the current directory, about 400MB). Do you wish to continue? [y/n] "
    url = "http://www.cs.ru.nl/~fgieseke/data/sdss/photometric_spec_confirmed.npy"

    ack =   """\nFunding for the SDSS and SDSS-II has been provided by the Alfred 
            P. Sloan Foundation, the Participating Institutions, the National 
            Science Foundation, the U.S. Department of Energy, the National 
            Aeronautics and Space Administration, the Japanese Monbukagakusho, 
            the Max Planck Society, and the Higher Education Funding Council 
            for England. The SDSS Web Site is http://www.sdss.org/.\n\nThe 
            SDSS is managed by the Astrophysical Research Consortium for the 
            Participating Institutions. The Participating Institutions are 
            the American Museum of Natural History, Astrophysical Institute 
            Potsdam, University of Basel, University of Cambridge, Case 
            Western Reserve University, University of Chicago, Drexel 
            University, Fermilab, the Institute for Advanced Study, the Japan 
            Participation Group, Johns Hopkins University, the Joint Institute 
            for Nuclear Astrophysics, the Kavli Institute for Particle 
            Astrophysics and Cosmology, the Korean Scientist Group, the 
            Chinese Academy of Sciences (LAMOST), Los Alamos National 
            Laboratory, the Max-Planck-Institute for Astronomy (MPIA), the 
            Max-Planck-Institute for Astrophysics (MPA), New Mexico State 
            University, Ohio State University, University of Pittsburgh, 
            University of Portsmouth, Princeton University, the United States 
            Naval Observatory, and the University of Washington.\n\n
             """

    if not os.path.isfile(fname) or os.path.getsize(fname) != 363691280:
        answer = ask_question(q)
        if answer == True:
            print(ack)
            download_from_url(url, fname)
        else:
            print("Exiting ...")
            sys.exit(0)

    # columns = ['specObjID', 'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z', 'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z',\
    #             'petroMag_u','petroMag_g','petroMag_r','petroMag_i','petroMag_z','z']
    data = numpy.load(fname)
    N = len(data)
    # default
    if NUM == None:
        NUM = 2000000
    X = numpy.zeros((N, 10))
    X[:, 0] = data[:,1]
    X[:, 1] = data[:,2]
    X[:, 2] = data[:,3]
    X[:, 3] = data[:,4]
    X[:, 4] = data[:,5]
    X[:, 5] = data[:,6]
    X[:, 6] = data[:,7]
    X[:, 7] = data[:,8]
    X[:, 8] = data[:,9]
    X[:, 9] = data[:,10]   
    Y = data[:,16]
    Y = Y.astype(numpy.float32)
    # remove invalid entries
    selector = numpy.min(X,axis=1) > -5000
    X = X[selector]
    Y = Y[selector]
    # divide into training and testing data
    X = numpy.array(X, dtype='float32')
    X = numpy.array(X[:NUM])
    Y = Y[:NUM]

    return X, Y

def psf_model_mag_test(NUM=None):

    fname = os.path.join(os.path.dirname(__file__), 'data', 'photometric.npy')
    q = "Additional data need to be downloaded (to the current directory, about 2GB). Do you wish to continue? [y/n] "
    url = "http://www.cs.ru.nl/~fgieseke/data/sdss/photometric.npy"

    ack =   """\nFunding for the SDSS and SDSS-II has been provided by the Alfred 
            P. Sloan Foundation, the Participating Institutions, the National 
            Science Foundation, the U.S. Department of Energy, the National 
            Aeronautics and Space Administration, the Japanese Monbukagakusho, 
            the Max Planck Society, and the Higher Education Funding Council 
            for England. The SDSS Web Site is http://www.sdss.org/.\n\nThe 
            SDSS is managed by the Astrophysical Research Consortium for the 
            Participating Institutions. The Participating Institutions are 
            the American Museum of Natural History, Astrophysical Institute 
            Potsdam, University of Basel, University of Cambridge, Case 
            Western Reserve University, University of Chicago, Drexel 
            University, Fermilab, the Institute for Advanced Study, the Japan 
            Participation Group, Johns Hopkins University, the Joint Institute 
            for Nuclear Astrophysics, the Kavli Institute for Particle 
            Astrophysics and Cosmology, the Korean Scientist Group, the 
            Chinese Academy of Sciences (LAMOST), Los Alamos National 
            Laboratory, the Max-Planck-Institute for Astronomy (MPIA), the 
            Max-Planck-Institute for Astrophysics (MPA), New Mexico State 
            University, Ohio State University, University of Pittsburgh, 
            University of Portsmouth, Princeton University, the United States 
            Naval Observatory, and the University of Washington.\n\n
             """

    if not os.path.isfile(fname) or os.path.getsize(fname) != 1790285720:
        answer = ask_question(q)
        if answer == True:
            print(ack)
            download_from_url(url, fname)
        else:
            print("Exiting ...")
            sys.exit(0)

    #columns = ['objID', 'ra', 'dec', 'type', \
    #        'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z', \
    #        'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', \
    #        'petroMag_u', 'petroMag_g', 'petroMag_r', 'petroMag_i', 'petroMag_z']
    data = numpy.load(fname)
    N = len(data)
    # default
    if NUM == None:
        NUM = 10000000
    X = numpy.zeros((N, 10))
    X[:, 0] = data[:,4]
    X[:, 1] = data[:,5]
    X[:, 2] = data[:,6]
    X[:, 3] = data[:,7]
    X[:, 4] = data[:,8]
    X[:, 5] = data[:,9]
    X[:, 6] = data[:,10]
    X[:, 7] = data[:,11]
    X[:, 8] = data[:,12]
    X[:, 9] = data[:,13]   
    X = numpy.array(X, dtype='float32')
    X = numpy.array(X[:NUM])

    return X

def get_data_set(data_set="psf_model_mag", NUM_TRAIN=None, NUM_TEST=None):

    if data_set == "psf_model_mag":
        Xtrain, Ytrain = psf_model_mag_train(NUM_TRAIN)
        Xtest = psf_model_mag_test(NUM_TEST) 
    else: 
        raise Exception("Unknown data set!")

    return Xtrain, Ytrain, Xtest

