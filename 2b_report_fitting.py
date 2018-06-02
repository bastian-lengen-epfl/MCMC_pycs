import pycs
import os,sys
import numpy as np
import util_func as ut


execfile("config.py")

print "Report will be writen in " + lens_directory +'report_fitting.txt'

f = open(lens_directory+'report_fitting.txt', 'w')
f.write('Measured time shift after fitting the splines : \n')
f.write('------------------------------------------------\n')

for i,kn in enumerate(knotstep) :
    f.write('knotsetp : %i'%kn +'\n')
    f.write('\n')
    for j, knml in enumerate(mlknotsteps):
        lcs, spline = pycs.gen.util.readpickle(lens_directory + '%s/initopt_%s_ks%i_ksml%i.pkl' % (combkw[i,j], dataname, kn,knml), verbose = False)
        delay_pair, delay_name = ut.getdelays(lcs)
        # print 'Micro-lensing knotstep = %i'%knml +"     Delays are " + str(delay_pair) + " for pairs "  + str(delay_name)
        f.write('Micro-lensing knotstep = %i'%knml +"     Delays are " + str(delay_pair) + " for pairs "  + str(delay_name) + '\n')

    f.write('\n')

f.write('------------------------------------------------\n')
f.write('Measured time shift after fitting with regdiff : \n')
lcs = pycs.gen.util.readpickle(lens_directory + 'regdiff_fitting/initopt_regdiff.pkl', verbose = False)
delay_pair, delay_name = ut.getdelays(lcs)
f.write('Regdiff : ' +"     Delays are " + str(delay_pair) + " for pairs "  + str(delay_name) + '\n')
f.write('------------------------------------------------\n')

starting_point = []
for i in range(len(timeshifts)):
    for j in range(len(timeshifts)):
        if i >= j :
			continue
        else :
            starting_point.append(timeshifts[j]-timeshifts[i])

f.write('Starting point used : '+ str(starting_point) + " for pairs "  + str(delay_name) + '\n')


f.close()