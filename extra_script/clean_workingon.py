#script to erase all the workingon file in the simulations of an object.
#Use it when the optimisation (3c) has crashed


import pycs
import os, glob
import numpy as np

execfile("../config.py")

for a,kn in enumerate(knotstep) :
    for  b, knml in enumerate(mlknotsteps):
        os.chdir(lens_directory + combkw[a, b])
        files = glob.glob('sims*/*.workingon')
        print "files to remove : ", files
        for fil in files :
            os.remove(fil)


