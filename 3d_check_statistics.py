#This script simply check that the optimised mocks light curves have the same statistics than the real one in term of zruns and sigmas.
#Plots are created in your figure directory.

import pycs
import os,sys, glob, importlib
import argparse as ap


def main(lensname,dataname,work_dir='./'):
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    check_stat_plot_dir = config.figure_directory + 'check_stat_plots/'

    if not os.path.isdir(check_stat_plot_dir):
        os.mkdir(check_stat_plot_dir)

    for i,kn in enumerate(config.knotstep) :
        for j, knml in enumerate(config.mlknotsteps):
            simset_available = glob.glob(config.lens_directory + config.combkw[i, j] + '/sims_mocks_*')
            lcs, spline = pycs.gen.util.readpickle(config.lens_directory + config.combkw[i, j] + '/initopt_%s_ks%i_ksml%i.pkl' % (dataname, kn, knml))

            for a in simset_available :
                a = a.split('/')[-1]
                if "_opt_" in a : # take only the optimised sub-folders
                    sset = a.split('_opt_')[0]
                    sset = sset[5:]
                    ooset = a.split('_opt_')[1]
                    if ooset[0:7] == 'regdiff' :
                        continue #it makes no sens to use this function for regdiff
                    else :
                        pycs.gen.stat.anaoptdrawn(lcs, spline, simset=sset, optset=ooset, showplot=False, nplots= 1,
                                                  directory= config.lens_directory + config.combkw[i, j] + '/')
                        #move the figure to the correct directory :
                        os.system("mv "  + "fig_anaoptdrawn_%s_%s_resi_1.png " % (sset, ooset) + check_stat_plot_dir
                                  +  "%s_fig_anaoptdrawn_%s_%s_resi_1.png" % (config.combkw[i, j], sset, ooset))
                        os.system("mv "+ "fig_anaoptdrawn_%s_%s_resihists.png " % (sset, ooset) + check_stat_plot_dir
                                  + "%s_fig_anaoptdrawn_%s_%s_resihists.png" % (config.combkw[i, j], sset, ooset) )

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Check the noise statistics of the mock light curves.",
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
>>>>>>> f82a52d2dd5305d9ac91e9cbcd5b0a1d49047c27
