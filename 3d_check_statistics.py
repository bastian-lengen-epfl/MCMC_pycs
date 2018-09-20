#This script simply check that the optimised mocks light curves have the same statistics than the real one in term of zruns and sigmas.
#Plots are created in your figure directory.

import pycs
import os,sys, glob

execfile("config.py")

check_stat_plot_dir = figure_directory + 'check_stat_plots/'

if not os.path.isdir(check_stat_plot_dir):
	os.mkdir(check_stat_plot_dir)

for i,kn in enumerate(knotstep) :
    for j, knml in enumerate(mlknotsteps):
        simset_available = glob.glob(lens_directory + combkw[i, j] + '/sims_mocks_*')
        lcs, spline = pycs.gen.util.readpickle(lens_directory + combkw[i, j] + '/initopt_%s_ks%i_ksml%i.pkl' % (dataname, kn, knml))

        for a in simset_available :
            a = a.split('/')[-1]
            if "_opt_" in a : # take only the optimised sub-folders
                sset = a.split('_opt_')[0]
                sset = sset[5:]
                ooset = a.split('_opt_')[1]
                print ooset[0:7]
                if ooset[0:7] == 'regdiff' :
                    continue #it makes no sens to use this function for regdiff
                else :
                    pycs.gen.stat.anaoptdrawn(lcs, spline, simset=sset, optset=ooset, showplot=False, nplots= 1,
                                              directory= lens_directory + combkw[i, j] + '/')
                    #move the figure to the correct directory :
                    os.system("mv "  + "fig_anaoptdrawn_%s_%s_resi_1.png " % (sset, ooset) + check_stat_plot_dir
                              +  "%s_fig_anaoptdrawn_%s_%s_resi_1.png" % (combkw[i, j], sset, ooset))
                    os.system("mv "+ "fig_anaoptdrawn_%s_%s_resihists.png " % (sset, ooset) + check_stat_plot_dir
                              + "%s_fig_anaoptdrawn_%s_%s_resihists.png" % (combkw[i, j], sset, ooset) )

#TODO : confirm that it has no sens to use this script for regdiff
