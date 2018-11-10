import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pycs, sys
import os, importlib
import argparse as ap


def main(lensname,dataname,work_dir='./'):
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    figure_directory = config.figure_directory + "final_results/"
    if not os.path.isdir(figure_directory):
        os.mkdir(figure_directory)

    for a,kn in enumerate(config.knotstep) :
        for  b, knml in enumerate(config.mlknotsteps):
            for o, opt in enumerate(config.optset):

                #Copies :
                copiesres = [pycs.sim.run.collect(config.lens_directory + config.combkw[a, b] + '/sims_%s_opt_%s' %(config.simset_copy, opt), 'blue',
                                                  dataname + "_" + config.combkw[a, b])]
                pycs.sim.plot.hists(copiesres, r=50.0, nbins=100, dataout=True,
                                    filename=figure_directory+'delay_hist_%i-%i_sims_%s_opt_%s.png'%(kn,knml,config.simset_copy, opt),
                                    outdir = config.lens_directory + config.combkw[a, b] + '/sims_%s_opt_%s/' %(config.simset_copy, opt))
                if config.display:
                    plt.show()

                #simulations
                simres = [pycs.sim.run.collect(config.lens_directory + config.combkw[a, b] +'/sims_%s_opt_%s' % (config.simset_mock, opt),
                                               'blue', dataname + "_" + config.combkw[a, b])]
                pycs.sim.plot.measvstrue(simres, r=2*config.truetsr, nbins=10, plotpoints=True, ploterrorbars=True, sidebyside=True,
                                         errorrange=5., binclip=True, binclipr=20.0, dataout=True, figsize = (12,8),
                                         filename=figure_directory+'deviation_hist_%i-%i_sims_%s_opt_%s.png'%(kn,knml,config.simset_copy, opt),
                                         outdir =config.lens_directory + config.combkw[a, b] + '/sims_%s_opt_%s/' %(config.simset_copy, opt))
                if config.display:
                    plt.show()

                # pycs.sim.plot.hists(simres,  r=2*truetsr, nbins=50, dataout=True,
                #                     filename=figure_directory+'delay_hists_mocks_%i-%i_sims_%s_opt_%s.png'%(kn,knml,simset_mock, opt),
                #                     outdir = lens_directory + combkw[a, b] + '/sims_%s_opt_%s/' %(simset_copy, opt))

                toplot = []
                spl = (pycs.gen.util.readpickle(config.lens_directory + config.combkw[a, b] +
                                                '/sims_%s_opt_%s/' %(config.simset_copy, opt)+'sims_%s_opt_%s_delays.pkl' % (config.simset_copy, opt)),
                       pycs.gen.util.readpickle(config.lens_directory + config.combkw[a, b] +
                                                '/sims_%s_opt_%s/' %(config.simset_copy, opt)+'sims_%s_opt_%s_errorbars.pkl' % (config.simset_mock, opt)))

                toplot.append(spl)

                text = [(0.43, 0.85, r"$\mathrm{" + config.full_lensname + "}$", {"fontsize": 18})]

                pycs.sim.plot.newdelayplot(toplot, rplot=5.0, displaytext=True, text=text,
                                           filename=figure_directory+"fig_delays__%i-%i_%s_%s.png" % (kn,knml,config.simset_mock, opt))

                if config.display:
                    plt.show()

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Plot the final results.",
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