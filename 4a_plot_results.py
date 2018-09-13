
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
        for o, opt in enumerate(optset):

            #Copies :
            copiesres = [pycs.sim.run.collect(lens_directory + combkw[a, b] + '/sims_%s_opt_%s' %(simset_copy, opt), 'blue',
                                              dataname + "_" + combkw[a, b])]
            pycs.sim.plot.hists(copiesres, r=50.0, nbins=100, dataout=True,
                                filename=figure_directory+'delay_hist_%i-%i_sims_%s_opt_%s.png'%(kn,knml,simset_copy, opt),
                                outdir = lens_directory + combkw[a, b] + '/sims_%s_opt_%s/' %(simset_copy, opt))
            if display:
                plt.show()

            #simulations
            simres = [pycs.sim.run.collect(lens_directory + combkw[a, b] +'/sims_%s_opt_%s' % (simset_mock, opt),
                                           'blue', dataname + "_" + combkw[a, b])]
            pycs.sim.plot.measvstrue(simres, r=2*truetsr, nbins=10, plotpoints=True, ploterrorbars=True, sidebyside=True,
                                     errorrange=5., binclip=True, binclipr=20.0, dataout=True, figsize = (12,8),
                                     filename=figure_directory+'deviation_hist_%i-%i_sims_%s_opt_%s.png'%(kn,knml,simset_copy, opt),
                                     outdir =lens_directory + combkw[a, b] + '/sims_%s_opt_%s/' %(simset_copy, opt))
            if display:
                plt.show()

            # pycs.sim.plot.hists(simres,  r=2*truetsr, nbins=50, dataout=True,
            #                     filename=figure_directory+'delay_hists_mocks_%i-%i_sims_%s_opt_%s.png'%(kn,knml,simset_mock, opt),
            #                     outdir = lens_directory + combkw[a, b] + '/sims_%s_opt_%s/' %(simset_copy, opt))

            toplot = []
            spl = (pycs.gen.util.readpickle(lens_directory + combkw[a, b] +
                                            '/sims_%s_opt_%s/' %(simset_copy, opt)+'sims_%s_opt_%s_delays.pkl' % (simset_copy, opt)),
                   pycs.gen.util.readpickle(lens_directory + combkw[a, b] +
                                            '/sims_%s_opt_%s/' %(simset_copy, opt)+'sims_%s_opt_%s_errorbars.pkl' % (simset_mock, opt)))

            toplot.append(spl)

            text = [(0.43, 0.85, r"$\mathrm{" + full_lensname + "}$", {"fontsize": 18})]

            pycs.sim.plot.newdelayplot(toplot, rplot=5.0, displaytext=True, text=text,
                                       filename=figure_directory+"fig_delays__%i-%i_%s_%s.png" % (kn,knml,simset_mock, opt))

            if display:
                plt.show()

