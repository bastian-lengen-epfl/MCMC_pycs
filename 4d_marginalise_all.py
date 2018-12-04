import pycs, sys
import os, copy
import numpy as np
import pickle as pkl
import importlib
import argparse as ap
from module import util_func as ut


def main(lensname,dataname,work_dir='./'):
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)
    marginalisation_plot_dir = config.figure_directory + 'marginalisation_plots/'

    if not os.path.isdir(marginalisation_plot_dir):
        os.mkdir(marginalisation_plot_dir)

    indiv_marg_dir = marginalisation_plot_dir + config.new_name_marg + '/'
    if not os.path.isdir(indiv_marg_dir):
        os.mkdir(indiv_marg_dir)

    marginalisation_dir = config.lens_directory + config.new_name_marg + '/'
    if not os.path.isdir(marginalisation_dir):
        os.mkdir(marginalisation_dir)

    colors = ["royalblue", "crimson", "seagreen", "darkorchid", "darkorange", 'indianred', 'purple', 'brown', 'black', 'violet', 'paleturquoise', 'palevioletred', 'olive',
              'indianred', 'salmon','lightcoral', 'chocolate', 'indigo', 'steelblue' , 'cyan', 'gold']

    if len(config.name_marg_list) != len(config.sigmathresh_list):
        print "Error : name_marg_list and sigmathresh_list must haev the same size ! "
        exit()

    path_list = [config.lens_directory + marg + '/' + marg +'_sigma_%2.2f'%sig + '_combined.pkl' for marg,sig in zip(config.name_marg_list,config.sigmathresh_list)]
    name_list = [d for d in config.display_name]
    group_list, combined = ut.group_estimate(path_list, name_list, config.delay_labels, colors, config.sigmathresh_final, config.new_name_marg
                                             , testmode = config.testmode, object_name= config.lcs_label)

    #plot the results :

    text = [
        (0.85, 0.90, r"$\mathrm{"+config.full_lensname+"}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
         {"fontsize": 26, "horizontalalignment": "center"})]

    radius = (combined.errors_down[0] + combined.errors_up[0]) / 2.0 * 2.5
    ncurve = len(config.lcs_label)

    if ncurve > 2:
        auto_radius = True
    else:
        auto_radius = False

    if config.display :
        pycs.mltd.plot.delayplot(group_list+[combined], rplot=radius, refgroup=combined, text=text,
                                 hidedetails=True, showbias=False, showran=False, showlegend=True,tick_step_auto= True,
                                 figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False, auto_radius=auto_radius)

    pycs.mltd.plot.delayplot(group_list+[combined], rplot=radius, refgroup=combined, text=text,
                             hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10),auto_radius=auto_radius, tick_step_auto= True,
                             horizontaldisplay=False, legendfromrefgroup=False, filename = indiv_marg_dir + config.name_marg_spline +"_sigma_%2.2f.png"%config.sigmathresh_final)

    pkl.dump(group_list, open(marginalisation_dir + config.new_name_marg +"_sigma_%2.2f"%config.sigmathresh_final +'_goups.pkl', 'wb'))
    pkl.dump(combined, open(marginalisation_dir + config.new_name_marg + "_sigma_%2.2f"%config.sigmathresh_final + '_combined.pkl', 'wb'))

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Even higher level of marginalisation, marginalise over previous marginalisation",
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