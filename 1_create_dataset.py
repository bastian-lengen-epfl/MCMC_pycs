#!/urs/bin/env python

import pycs
import os,sys

execfile("config.py")

##### import the data

rdbfile = 'data/'+lensname + "_" + dataname+'.rdb'

lcs = []
for i,a in enumerate(lcs_label):
    lcs.append(pycs.gen.lc.rdbimport(rdbfile, a, 'mag_'+a, 'magerr_'+a, dataname))

if display :
    pycs.gen.mrg.colourise(lcs)
    pycs.gen.lc.display(lcs,showdates=True)

pycs.gen.util.writepickle(lcs,'pkl/'+lensname + "_" + dataname+'.pkl')

if not os.path.exists(lens_directory + 'figure/'):
    os.mkdir(lens_directory + 'figure/')

print "I have created a new config file for this object ! You probably want to edit it in %s"%(lens_directory + "config_" + lensname + "_" + dataname + ".py")

