#!/urs/bin/env python

import pycs
import os,sys
import numpy as np


execfile("config.py")


def applyshifts(lcs,timeshifts,magshifts):

	if not len(lcs) == len(timeshifts) and len(lcs) == len(magshifts):
		print "Hey, give me arrays of the same lenght !"
		sys.exit()

	for lc, timeshift, magshift in zip(lcs, timeshifts, magshifts):
		lc.resetshifts()
		lc.shiftmag(-np.median(lc.getmags()))
		#lc.shiftmag(magshift)
		lc.shifttime(timeshift)


lcs = pycs.gen.util.readpickle(data)
applyshifts(lcs, timeshifts, magshifts)
name = ['A','B','C','D']
for i,lc in enumerate(lcs):
	print "I will aplly a initail shift of : %2.4f days, %2.4f mag for %s" %(timeshifts[i],magshifts[i],name[i])



for i,kn in enumerate(knotstep) :
	for j, knml in enumerate(mlknotsteps):
		if knml != 0 :
			attachml(lcs, knml) # add microlensing
		spline = optfct(lcs, kn)
		# add the polyml shift as a magshift to the iniopt
		if 0:  # FOR polyml only, put 0 if you use splml
			for lc in lcs:
				pass
				print lc.ml.longinfo()
				magshift = lc.ml.getfreeparams()[0]
				lc.rmml()
				lc.shiftmag(magshift)

		rls = pycs.gen.stat.subtract(lcs, spline)
		if display :
			pycs.gen.lc.display(lcs, [spline], showlegend=True, showdelays=True, filename="screen")
			pycs.gen.stat.plotresiduals([rls])
		else :
			pycs.gen.lc.display(lcs, [spline], showlegend=True, showdelays=True,
								filename=figure_directory + "spline_fit_ks%i_ksml%i.png"%(kn,knml))
			pycs.gen.stat.plotresiduals([rls], filename=figure_directory + "residual_fit_ks%i_ksml%i.png"%(kn,knml))



		# and write data, again
		if not os.path.isdir(lens_directory + combkw[i,j]):
			os.mkdir(lens_directory + combkw[i,j])

		pycs.gen.util.writepickle((lcs, spline), lens_directory + '%s/initopt_%s_ks%i_ksml%i.pkl' % (combkw[i,j], dataname, kn,knml))



#DO the optimisation with regdiff as well !
import pycs.regdiff
for ind, l in enumerate(lcs):
	l.shiftmag(ind*0.1)

myrslcs = [pycs.regdiff.rslc.factory(l, pd=pointdensity, covkernel=covkernel, pow=pow, amp=amp, scale=scale, errscale=errscale) for l in lcs]

if display :
	pycs.gen.lc.display(lcs, myrslcs)
pycs.gen.lc.display(lcs, myrslcs, showdelays=True, filename = figure_directory + "regdiff_fit.png")

for ind, l in enumerate(lcs):
	l.shiftmag(-ind*0.1)

map(regdiff, [lcs])

if display :
	pycs.gen.lc.display(lcs, showlegend=False, showdelays=True)
pycs.gen.lc.display(lcs, showlegend=False, showdelays=True, filename = figure_directory + "regdiff_optimized_fit.png")
if not os.path.isdir(lens_directory + 'regdiff_fitting'):
	os.mkdir(lens_directory + 'regdiff_fitting')
pycs.gen.util.writepickle(lcs, lens_directory + 'regdiff_fitting/initopt_regdiff.pkl')