
import pycs
import os
import numpy as np
import matplotlib.pyplot as plt

execfile("config.py")

figure_directory = figure_directory + "final_results/"
if not os.path.isdir(figure_directory):
	os.mkdir(figure_directory)

for a,kn in enumerate(knotstep) :
    for  b, knml in enumerate(mlknotsteps):

        if run_on_copies:

            copiesres = [pycs.sim.run.collect(lens_directory + combkw[a, b] + '/sims_%s_opt_%s' %(simset_copy, optset), 'blue', dataname + "_" + combkw[a, b])]
            pycs.sim.plot.hists(copiesres, r=50.0, nbins=100, dataout=True, filename=figure_directory+'delay_hist_%i-%i_sims_%s_opt_%s.png'%(kn,knml,simset_copy, optset),
                                outdir = lens_directory + combkw[a, b] + '/sims_%s_opt_%s/' %(simset_copy, optset))
            if display:
                plt.show()

        if run_on_sims:
            simres = [pycs.sim.run.collect(lens_directory + combkw[a, b] +'/sims_%s_opt_%s' % (simset_mock, optset), 'blue', dataname + "_" + combkw[a, b])]
            pycs.sim.plot.measvstrue(simres, r=30.0, nbins=10, plotpoints=True, ploterrorbars=True, sidebyside=True,
                                     errorrange=20., binclip=True, binclipr=20.0, dataout=True, filename=figure_directory+'deviation_hist_%i-%i_sims_%s_opt_%s.png'%(kn,knml,simset_copy, optset),
                                     outdir =lens_directory + combkw[a, b] + '/sims_%s_opt_%s/' %(simset_copy, optset) )
            if display:
                plt.show()

        toplot = []
        spl = (pycs.gen.util.readpickle(lens_directory + combkw[a, b] + '/sims_%s_opt_%s/' %(simset_copy, optset)+'sims_%s_opt_%s_delays.pkl' % (simset_copy, optset)),
               pycs.gen.util.readpickle(lens_directory + combkw[a, b] + '/sims_%s_opt_%s/' %(simset_copy, optset)+'sims_%s_opt_%s_errorbars.pkl' % (simset_mock, optset)))

        toplot.append(spl)

        text = [(0.43, 0.85, r"$\mathrm{" + full_lensname + "}$", {"fontsize": 18}),
                (0.7, 0.5, 'Warning, spline delays and errors \n are preliminary', {"fontsize": 10})]

        pycs.sim.plot.newdelayplot(toplot, rplot=5.0, displaytext=True, text=text,
                                   filename=figure_directory+"fig_delays__%i-%i_%s_%s.png" % (kn,knml,simset_mock, optset))

        if display:
            plt.show()

