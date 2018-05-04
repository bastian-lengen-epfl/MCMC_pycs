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


if optfct==regdiff: #regdiff just to display the fit...

	for ind, l in enumerate(lcs):
		l.shiftmag(ind*0.1)
	myrslcs = [pycs.regdiff.rslc.factory(l, pd=pointdensity, covkernel=covkernel, pow=pow, amp=amp, scale=scale, errscale=errscale) for l in lcs]
	pycs.gen.lc.display(lcs, myrslcs)
	for ind, l in enumerate(lcs):
		l.shiftmag(-ind*0.1)

	attachml(lcs)
	map(optfct, [lcs])

	pycs.gen.lc.display(lcs, showlegend=False, showdelays=True)
	sys.exit()


else:
	for i,kn in enumerate(knotstep) :
		attachml(lcs) # add microlensing
		spline = optfct(lcs, kn)
		# add the polyml shift as a magshift to the iniopt
		if 0:  # FOR polyml only, put 0 if you use splml
			for lc in lcs:
				pass
				print lc.ml.longinfo()
				magshift = lc.ml.getfreeparams()[0]
				lc.rmml()
				lc.shiftmag(magshift)

		if display :
			pycs.gen.lc.display(lcs, [spline], showlegend=True, showdelays=True)

		# and write data, again
		print lens_directory
		if not os.path.isdir(lens_directory + combkw[i]):
			os.mkdir(lens_directory + combkw[i])

		pycs.gen.util.writepickle((lcs, spline), lens_directory + '%s/initopt_%s_ks%i.pkl' % (combkw[i], dataname, kn))
