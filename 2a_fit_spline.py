#!/urs/bin/env python

import pycs
import os,sys
import numpy as np
from module import util_func as ut

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

figure_directory = figure_directory + "spline_and_residuals_plots/"
if not os.path.isdir(figure_directory):
	os.mkdir(figure_directory)

lcs = pycs.gen.util.readpickle(data)
applyshifts(lcs, timeshifts, magshifts)
name = ['A','B','C','D']
for i,lc in enumerate(lcs):
	print "I will aplly a initail shift of : %2.4f days, %2.4f mag for %s" %(timeshifts[i],magshifts[i],name[i])



for i,kn in enumerate(knotstep) :
	for j, knml in enumerate(mlknotsteps):
		if knml != 0 :
			attachml(lcs, knml) # add microlensing

		spline = spl1(lcs, kn = kn)
		rls = pycs.gen.stat.subtract(lcs, spline)
		pycs.gen.mrg.colourise(lcs)
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



#DO the optimisation with regdiff as well, just to have an idea, this the first point of the grid !
import pycs.regdiff
for ind, l in enumerate(lcs):
	l.shiftmag(ind*0.1)

kwargs_optimiser_simoptfct = ut.get_keyword_regdiff(pointdensity, covkernel, pow, amp, scale, errscale)
regdiff_param_kw = ut.generate_regdiff_regdiffparamskw(pointdensity, covkernel, pow, amp, scale, errscale)
for i,k in enumerate(kwargs_optimiser_simoptfct):

	myrslcs = [pycs.regdiff.rslc.factory(l, pd=k['pointdensity'], covkernel=k['covkernel'],
										 pow=k['pow'], amp=k['amp'], scale=k['scale'], errscale=k['errscale']) for l in lcs]

	if display :
		pycs.gen.lc.display(lcs, myrslcs)
	pycs.gen.lc.display(lcs, myrslcs, showdelays=True, filename = figure_directory + "regdiff_fit%s.png"%regdiff_param_kw[i])

	for ind, l in enumerate(lcs):
		l.shiftmag(-ind*0.1)

	# map(regdiff, [lcs], **kwargs_optimiser_simoptfct[0])
	regdiff(lcs, **kwargs_optimiser_simoptfct[i])

	if display :
		pycs.gen.lc.display(lcs, showlegend=False, showdelays=True)
	pycs.gen.lc.display(lcs, showlegend=False, showdelays=True, filename = figure_directory + "regdiff_optimized_fit%s.png"%regdiff_param_kw[i])
	if not os.path.isdir(lens_directory + 'regdiff_fitting'):
		os.mkdir(lens_directory + 'regdiff_fitting')
	pycs.gen.util.writepickle(lcs, lens_directory + 'regdiff_fitting/initopt_regdiff%s.pkl'%regdiff_param_kw[i])