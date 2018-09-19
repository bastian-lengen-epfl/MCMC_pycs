import pycs
from module import util_func as ut

execfile("config.py")

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
        # print 'Micro-lensing knotstep = %i'%knml +"     Delays are " + str(delay_pair) + " for pairs "  + str(delay_name)
        f.write('Micro-lensing knotstep = %i'%knml +"     Delays are " + str(delay_pair) + " for pairs "  + str(delay_name) + '\n')

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
