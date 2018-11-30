#!/urs/bin/env python

import os,sys
import numpy as np
import argparse as ap
from module import util_func as ut

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

def compute_dof_spline(rls, kn,knml):
    '''compute the degree of freedom for the spline optimiser'''
    n_curve = len(rls)
    a = rls[0].jds[0]
    b = rls[0].jds[-1]
    nkn = int(float(b - a) / float(kn) - 2) #number of internal knot
    nknml = int(float(b - a) / float(knml) - 2) #number of internal ml knot
    return (2*nkn + n_curve) + n_curve*(2*nknml + n_curve) + n_curve

def compute_chi2(rls, kn, knml):
    #return the chi2 given a rls object
    chi2 = 0.0
    for rl in rls :
        chi2_c = np.mean((rl.getmags()**2) / rl.getmagerrs()**2)
        print "Chi2 for light curve %s : %2.2f"%(rl.object, chi2_c)
        chi2 += chi2_c

    # chi2 =chi2 / count
    chi2_red = chi2 / compute_dof_spline(rls, kn,knml)
    print "DoF :", compute_dof_spline(rls, kn,knml)
    print "Final chi2 reduced: %2.5f"%chi2_red
    return chi2_red

def main(lensname,dataname,work_dir='./'):
    import importlib
    import pycs
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    figure_directory = config.figure_directory + "spline_and_residuals_plots/"
    if not os.path.isdir(figure_directory):
        os.mkdir(figure_directory)

    lcs = pycs.gen.util.readpickle(config.data)
    for i,lc in enumerate(config.lcs_label):
        print "I will aplly a initial shift of : %2.4f days, %2.4f mag for %s" %(config.timeshifts[i],config.magshifts[i],config.lcs_label[i])


    # Do the optimisation with the splines
    chi2 = np.zeros((len(config.knotstep),len(config.mlknotsteps)))
    dof = np.zeros((len(config.knotstep),len(config.mlknotsteps)))
    for i,kn in enumerate(config.knotstep) :
        for j, knml in enumerate(config.mlknotsteps):
            lcs = pycs.gen.util.readpickle(config.data)
            applyshifts(lcs, config.timeshifts, config.magshifts)
            if knml != 0 :
                config.attachml(lcs, knml) # add microlensing

            spline = config.spl1(lcs, kn = kn)
            pycs.gen.mrg.colourise(lcs)
            rls = pycs.gen.stat.subtract(lcs, spline)
            chi2[i,j] = compute_chi2(rls, kn, knml)
            dof[i,j] = compute_dof_spline(rls, kn, knml)

            if config.display :
                pycs.gen.lc.display(lcs, [spline], showlegend=True, showdelays=True, filename="screen")
                pycs.gen.stat.plotresiduals([rls])
            else :
                pycs.gen.lc.display(lcs, [spline], showlegend=True, showdelays=True,
                                    filename=figure_directory + "spline_fit_ks%i_ksml%i.png"%(kn,knml))
                pycs.gen.stat.plotresiduals([rls], filename=figure_directory + "residual_fit_ks%i_ksml%i.png"%(kn,knml))



            # and write data, again
            if not os.path.isdir(config.lens_directory + config.combkw[i,j]):
                os.mkdir(config.lens_directory + config.combkw[i,j])

            pycs.gen.util.writepickle((lcs, spline), config.lens_directory + '%s/initopt_%s_ks%i_ksml%i.pkl' % (config.combkw[i,j], dataname, kn,knml))



    #DO the optimisation with regdiff as well, just to have an idea, this the first point of the grid !
    lcs = pycs.gen.util.readpickle(config.data)
    pycs.gen.mrg.colourise(lcs)
    applyshifts(lcs, config.timeshifts, config.magshifts)

    import pycs.regdiff
    for ind, l in enumerate(lcs):
        l.shiftmag(ind*0.1)

    if config.use_preselected_regdiff:
        kwargs_optimiser_simoptfct = ut.get_keyword_regdiff_from_file(config.preselection_file)
        regdiff_param_kw = ut.read_preselected_regdiffparamskw(config.preselection_file)
    else:
        kwargs_optimiser_simoptfct = ut.get_keyword_regdiff(config.pointdensity, config.covkernel, config.pow, config.amp, config.scale, config.errscale)
        regdiff_param_kw = ut.generate_regdiffparamskw(config.pointdensity, config.covkernel, config.pow, config.amp, config.scale, config.errscale)

    for i,k in enumerate(kwargs_optimiser_simoptfct):
        myrslcs = []
        myrslcs = [pycs.regdiff.rslc.factory(l, pd=k['pointdensity'], covkernel=k['covkernel'],
                                             pow=k['pow'], amp=k['amp'], scale=k['scale'], errscale=k['errscale']) for l in lcs]

        if config.display :
            pycs.gen.lc.display(lcs, myrslcs)
        pycs.gen.lc.display(lcs, myrslcs, showdelays=True, filename = figure_directory + "regdiff_fit%s.png"%regdiff_param_kw[i])

        for ind, l in enumerate(lcs):
            l.shiftmag(-ind*0.1)

        config.regdiff(lcs, **kwargs_optimiser_simoptfct[i])

        if config.display :
            pycs.gen.lc.display(lcs, showlegend=False, showdelays=True)
        pycs.gen.lc.display(lcs, showlegend=False, showdelays=True, filename = figure_directory + "regdiff_optimized_fit%s.png"%regdiff_param_kw[i])
        if not os.path.isdir(config.lens_directory + 'regdiff_fitting'):
            os.mkdir(config.lens_directory + 'regdiff_fitting')
        pycs.gen.util.writepickle(lcs, config.lens_directory + 'regdiff_fitting/initopt_regdiff%s.pkl'%regdiff_param_kw[i])

    #Write the report :
    print "Report will be writen in " + config.lens_directory +'report/report_fitting.txt'

    f = open(config.lens_directory+'report/report_fitting.txt', 'w')
    f.write('Measured time shift after fitting the splines : \n')
    f.write('------------------------------------------------\n')

    for i,kn in enumerate(config.knotstep) :
        f.write('knotsetp : %i'%kn +'\n')
        f.write('\n')
        for j, knml in enumerate(config.mlknotsteps):
            lcs, spline = pycs.gen.util.readpickle(config.lens_directory + '%s/initopt_%s_ks%i_ksml%i.pkl' % (config.combkw[i,j], dataname, kn,knml), verbose = False)
            delay_pair, delay_name = ut.getdelays(lcs)
            f.write('Micro-lensing knotstep = %i'%knml +"     Delays are " + str(delay_pair) + " for pairs "  +
                    str(delay_name) + '. Chi2 Red : %2.5f\n'%chi2[i,j] + ' DoF : %i'%dof[i,j])

        f.write('\n')

    f.write('------------------------------------------------\n')
    f.write('Measured time shift after fitting with regdiff : \n')
    f.write('\n')

    for i,k in enumerate(kwargs_optimiser_simoptfct):
        lcs = pycs.gen.util.readpickle(config.lens_directory + 'regdiff_fitting/initopt_regdiff%s.pkl'%regdiff_param_kw[i], verbose = False)
        delay_pair, delay_name = ut.getdelays(lcs)
        f.write('Regdiff : ' +"     Delays are " + str(delay_pair) + " for pairs "  + str(delay_name) + '\n')
        f.write('------------------------------------------------\n')

    starting_point = []
    for i in range(len(config.timeshifts)):
        for j in range(len(config.timeshifts)):
            if i >= j :
                continue
            else :
                starting_point.append(config.timeshifts[j]-config.timeshifts[i])

    f.write('Starting point used : '+ str(starting_point) + " for pairs "  + str(delay_name) + '\n')
    f.close()

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Fit spline and regdiff on the data.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_lensname = "name of the lens to process"
    help_dataname = "name of the data set to process (Euler, SMARTS, ... )"
    help_work_dir = "name of the working directory"
    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument(dest='dataname', type=str,
                        metavar='dataname', action='store',
                        help=help_dataname)
    parser.add_argument('--dir', dest='work_dir', type=str,
                            metavar='', action='store', default='./',
                            help=help_work_dir)
    args = parser.parse_args()
    main(args.lensname,args.dataname, work_dir=args.work_dir)
