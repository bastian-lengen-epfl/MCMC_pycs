import argparse as ap
import os, sys, pycs
import pickle as pkl
import importlib
from shutil import copyfile
from module import util_func as ut

def main(lensname,work_dir='./'):

    combi_dir = os.path.join(os.path.join(work_dir,'Combination'),lensname)
    plot_dir = os.path.join(combi_dir, 'plots')
    if not os.path.exists(combi_dir):
        ut.mkdir_recursive(combi_dir)
    if not os.path.exists(plot_dir):
        ut.mkdir_recursive(plot_dir)

    config_file = work_dir + "config/config_combination_" + lensname +'.py'

    if os.path.isfile(config_file):
        print "Combination config file already exists."
        print "Using :", config_file
    else :
        print "Combination config file do not exist yet. I will create it."
        copyfile("config_combination_default.py", config_file)
        print "Please edit : ", config_file
        exit()


    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_combination_" + lensname)

    print "Working on :", lensname
    print "Combining the following data sets :", config.datasets

    ####ACCSSING THE GROUP FILE ####

    path_list = [config.lens_directory + marg + '/' + marg +'_sigma_%2.2f'%sig + '_combined.pkl' for marg,sig in zip(config.marg_to_combine,config.sigma_to_combine)]
    path_list_spline = [config.lens_directory + marg + '/' + marg +'_sigma_%2.2f'%sig + '_combined.pkl'
                        for marg,sig in zip(config.marg_to_combine_spline,config.sigma_to_combine_spline)]
    path_list_regdiff = [config.lens_directory + marg + '/' + marg +'_sigma_%2.2f'%sig + '_combined.pkl'
                         for marg,sig in zip(config.marg_to_combine_regdiff,config.sigma_to_combine_regdiff)]
    name_list = [d for d in config.datasets]

    combs = []
    combs_spline = []
    combs_regdiff = []
    for i,p in enumerate(path_list) :
        comb = pkl.load(p)
        comb.name = name_list[i]
        combs.append(comb)

    for i,p in enumerate(path_list_spline) :
        comb = pkl.load(p)
        comb.name = "Spline " + name_list[i]
        combs_spline.append(comb)

    for i,p in enumerate(path_list_regdiff) :
        comb = pkl.load(p)
        comb.name = "Regdiff " + name_list[i]
        combs_spline.append(comb)


    mult = pycs.mltd.comb.mult_estimates(combs)
    mult.name = "Mult"
    mult.plotcolor = "gray"

    sum = pycs.mltd.comb.combine_estimates(combs, sigmathresh=0.0, testmode=config.testmode)
    sum.name = "Sum"
    sum.plotcolor = "black"

    radius = (sum.errors_down[0] + sum.errors_up[0]) / 2.0 * 2.5
    ncurve = len(config.lcs_label)
    if ncurve > 2:
        auto_radius = True
    else:
        auto_radius = False

    text = [
        (0.85, 0.90, r"$\mathrm{"+config.full_lensname+"}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
         {"fontsize": 26, "horizontalalignment": "center"})]

    toplot = combs + [sum]
    if config.display :
        pycs.mltd.plot.delayplot(toplot, rplot=radius, refgroup=mult, text=text,
                                 hidedetails=True, showbias=False, showran=False, showlegend=True,tick_step_auto= True,
                                 figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False, auto_radius=auto_radius)

    pycs.mltd.plot.delayplot(toplot, rplot=radius, refgroup=mult, text=text,
                             hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), auto_radius=auto_radius, tick_step_auto= True,
                             horizontaldisplay=False, legendfromrefgroup=False, filename = plot_dir + "combined_estimated_"+config.combi_name + ".png")

    pkl.dump(combs, open(os.path.join(combi_dir, config.combi_name + '_goups.pkl'), 'wb'))
    pkl.dump([sum,mult], open(os.path.join(combi_dir, "sum-mult_"+config.combi_name + '.pkl'), 'wb'))

    ### Plot with the spline only ####
    sum_spline = pycs.mltd.comb.combine_estimates(combs_spline, sigmathresh=0.0, testmode=config.testmode)
    sum_spline.name = "Sum"
    sum_spline.plotcolor = "black"
    text = [
        (0.85, 0.90, r"$\mathrm{"+config.full_lensname+"}$" + "\n" + r"$\mathrm{PyCS\ Free-knot Spline}$",
         {"fontsize": 26, "horizontalalignment": "center"})]
    if config.display :
        pycs.mltd.plot.delayplot(combs_spline + [sum_spline], rplot=radius, refgroup=sum_spline, text=text,
                                 hidedetails=True, showbias=False, showran=False, showlegend=True,tick_step_auto= True,
                                 figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False, auto_radius=auto_radius)

    pycs.mltd.plot.delayplot(combs_spline + [sum_spline], rplot=radius, refgroup=sum_spline, text=text,
                             hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), auto_radius=auto_radius, tick_step_auto= True,
                             horizontaldisplay=False, legendfromrefgroup=False, filename = plot_dir + "combined_estimated_spline"+config.combi_name + ".png")

    ### Plot with regdiff only ####
    sum_regdiff = pycs.mltd.comb.combine_estimates(combs_regdiff, sigmathresh=0.0, testmode=config.testmode)
    sum_regdiff.name = "Sum"
    sum_regdiff.plotcolor = "black"
    text = [
        (0.85, 0.90, r"$\mathrm{"+config.full_lensname+"}$" + "\n" + r"$\mathrm{PyCS\ Regression Difference}$",
         {"fontsize": 26, "horizontalalignment": "center"})]
    if config.display :
        pycs.mltd.plot.delayplot(combs_regdiff + [sum_regdiff], rplot=radius, refgroup=sum_regdiff, text=text,
                                 hidedetails=True, showbias=False, showran=False, showlegend=True,tick_step_auto= True,
                                 figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False, auto_radius=auto_radius)

    pycs.mltd.plot.delayplot(combs_regdiff + [sum_regdiff], rplot=radius, refgroup=sum_regdiff, text=text,
                             hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), auto_radius=auto_radius, tick_step_auto= True,
                             horizontaldisplay=False, legendfromrefgroup=False, filename = plot_dir + "combined_estimated_regdiff"+config.combi_name + ".png")

    ### Plot with regdiff and spline together ####
    sum_all = pycs.mltd.comb.combine_estimates([combs_regdiff]+[combs_spline], sigmathresh=0.0, testmode=config.testmode)
    sum_all.name = "Sum"
    sum_all.plotcolor = "black"
    text = [
        (0.85, 0.90, r"$\mathrm{"+config.full_lensname+"}$" + "\n" + r"$\mathrm{PyCS\ Regression Difference and Spline}$",
         {"fontsize": 26, "horizontalalignment": "center"})]
    if config.display :
        pycs.mltd.plot.delayplot([combs_regdiff]+[combs_spline] + [sum_all], rplot=radius, refgroup=sum_all, text=text,
                                 hidedetails=True, showbias=False, showran=False, showlegend=True,tick_step_auto= True,
                                 figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False, auto_radius=auto_radius)

    pycs.mltd.plot.delayplot([combs_regdiff]+[combs_spline] + [sum_all], rplot=radius, refgroup=sum_all, text=text,
                             hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), auto_radius=auto_radius, tick_step_auto= True,
                             horizontaldisplay=False, legendfromrefgroup=False, filename = plot_dir + "combined_estimated_regdiff-spline"+config.combi_name + ".png")




if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Combine the data sets.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_lensname = "name of the lens to process"
    help_work_dir = "name of the working directory. default : ./"

    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument('--dir', dest='work_dir', type=str,
                        metavar='', action='store', default='./',
                        help=help_work_dir)


    args = parser.parse_args()

    main(args.lensname, work_dir=args.work_dir)