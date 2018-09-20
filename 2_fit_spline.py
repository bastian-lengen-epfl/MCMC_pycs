#!/urs/bin/env python

import pycs
import os,sys
import numpy as np
import pickle as pkl
from module import util_func as ut

execfile("config.py")

#TODO : code something to give a delay and not a time shift
def applyshifts(lcs,timeshifts,magshifts):

	if not len(lcs) == len(timeshifts) and len(lcs) == len(magshifts):
		print "Hey, give me arrays of the same lenght !"
		sys.exit()

	for lc, timeshift, magshift in zip(lcs, timeshifts, magshifts):
		lc.resetshifts()
		lc.shiftmag(-np.median(lc.getmags()))
		#lc.shiftmag(magshift)
		lc.shifttime(timeshift)

def compute_chi2(rls):
	#return the chi2 given a rls object
	chi2 = 0.0
	count = 0.0
	for rl in rls :
		meanmag = np.mean(rl.getmags())
		chi2_c = np.mean(((meanmag - rl.getmags())**2) / rl.getmagerrs()**2)
		print "Chi2 for light curve %s : %2.2f"%(rl.object, chi2_c)
		chi2 += chi2_c
		count +=1.0

	chi2 =chi2 / count
	print "Final chi2 : %2.2f"%chi2
	return chi2


figure_directory = figure_directory + "spline_and_residuals_plots/"
if not os.path.isdir(figure_directory):
	os.mkdir(figure_directory)

lcs = pycs.gen.util.readpickle(data)
name = ['A','B','C','D']
for i,lc in enumerate(lcs):
	print "I will aplly a initial shift of : %2.4f days, %2.4f mag for %s" %(timeshifts[i],magshifts[i],name[i])


# Do the optimisation with the splines
chi2 = np.zeros((len(knotstep),len(mlknotsteps)))
for i,kn in enumerate(knotstep) :
	for j, knml in enumerate(mlknotsteps):
		lcs = pycs.gen.util.readpickle(data)
		applyshifts(lcs, timeshifts, magshifts)
		if knml != 0 :
			attachml(lcs, knml) # add microlensing

		spline = spl1(lcs, kn = kn)
		pycs.gen.mrg.colourise(lcs)
		rls = pycs.gen.stat.subtract(lcs, spline)
		chi2[i,j] = compute_chi2(rls)

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
lcs = pycs.gen.util.readpickle(data)
pycs.gen.mrg.colourise(lcs)
applyshifts(lcs, timeshifts, magshifts)

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

#Write the report :
print "Report will be writen in " + lens_directory +'report/report_fitting.txt'

f = open(lens_directory+'report/report_fitting.txt', 'w')
f.write('Measured time shift after fitting the splines : \n')
f.write('------------------------------------------------\n')

for i,kn in enumerate(knotstep) :
    f.write('knotsetp : %i'%kn +'\n')
    f.write('\n')
    for j, knml in enumerate(mlknotsteps):
        lcs, spline = pycs.gen.util.readpickle(lens_directory + '%s/initopt_%s_ks%i_ksml%i.pkl' % (combkw[i,j], dataname, kn,knml), verbose = False)
        delay_pair, delay_name = ut.getdelays(lcs)
        f.write('Micro-lensing knotstep = %i'%knml +"     Delays are " + str(delay_pair) + " for pairs "  + str(delay_name) + '. Chi2 : %2.2f\n'%chi2[i,j])

    f.write('\n')

f.write('------------------------------------------------\n')
f.write('Measured time shift after fitting with regdiff : \n')
f.write('\n')
regdiff_param_kw = ut.generate_regdiff_regdiffparamskw(pointdensity, covkernel, pow, amp, scale, errscale)
kwargs_optimiser_simoptfct = ut.get_keyword_regdiff(pointdensity, covkernel, pow, amp, scale, errscale)
for i,k in enumerate(kwargs_optimiser_simoptfct):
    lcs = pycs.gen.util.readpickle(lens_directory + 'regdiff_fitting/initopt_regdiff%s.pkl'%regdiff_param_kw[i], verbose = False)
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