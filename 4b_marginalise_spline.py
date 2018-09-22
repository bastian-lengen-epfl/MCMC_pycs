import pycs.mltd
import os, copy, sys
import numpy as np
import pickle as pkl
import importlib
import argparse as ap

def main(lensname,dataname,work_dir='./'):
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)
    marginalisation_plot_dir = config.figure_directory + 'marginalisation_plots/'

    if not os.path.isdir(marginalisation_plot_dir):
        os.mkdir(marginalisation_plot_dir)

    indiv_marg_dir = marginalisation_plot_dir + config.name_marg_spline + '/'
    if not os.path.isdir(indiv_marg_dir):
        os.mkdir(indiv_marg_dir)

    marginalisation_dir = config.lens_directory + config.name_marg_spline + '/'
    if not os.path.isdir(marginalisation_dir):
        os.mkdir(marginalisation_dir)

    f = open(marginalisation_dir + 'report_%s_sigma%2.1f.txt'%(config.name_marg_regdiff,config.sigmathresh), 'w')

    if config.testmode:
        nbins = 500
    else:
        nbins = 5000

    colors = ["royalblue", "crimson", "seagreen", "darkorchid", "darkorange", 'indianred', 'purple', 'brown', 'violet', 'paleturquoise', 'palevioletred', 'olive',
              'indianred', 'salmon','lightcoral', 'chocolate', 'indigo', 'steelblue' , 'cyan', 'gold']
    color_id = 0

    group_list = []
    medians_list = []
    errors_up_list = []
    errors_down_list = []
    simset_mock_ava = ["mocks_n%it%i_%s" % (int(config.nsim * config.nsimpkls), config.truetsr,i) for i in config.tweakml_name_marg_spline]
    opt = 'spl1'

    for a,kn in enumerate(config.knotstep_marg):
        for b, knml in enumerate(config.mlknotsteps_marg):
            for n, noise in enumerate(config.tweakml_name_marg_spline):
                result_file_delay = config.lens_directory + config.combkw[a, b] + '/sims_%s_opt_%st%i/' % (config.simset_copy, opt, int(config.tsrand)) \
                                    + 'sims_%s_opt_%s' % (config.simset_copy, opt) + 't%i_delays.pkl'%int(config.tsrand)
                result_file_errorbars = config.lens_directory + config.combkw[a, b] + '/sims_%s_opt_%st%i/' % (config.simset_copy, opt, int(config.tsrand))\
                                        + 'sims_%s_opt_%s' % (simset_mock_ava[n], opt) + 't%i_errorbars.pkl'%int(config.tsrand)
                if not os.path.isfile(result_file_delay) or not os.path.isfile(result_file_errorbars):
                    print 'Error I cannot find the files %s or %s. ' \
                          'Did you run the 3c and 4a?' % (result_file_delay, result_file_errorbars)
                    f.write('Error I cannot find the files %s or %s. \n' % (result_file_delay, result_file_errorbars))
                    continue

                name = "%s_ks%i_%s_%s" % (dataname,kn,knml, noise)
                group_list.append(pycs.mltd.comb.getresults(
                    pycs.mltd.comb.CScontainer(data=dataname, knots=kn, ml=knml, name=name,
                                               drawopt=config.optfctkw, runopt=opt, ncopy=config.ncopy*config.ncopypkls, nmocks=config.nsim*config.nsimpkls, truetsr=config.truetsr,
                                               colour=colors[color_id], result_file_delays= result_file_delay, result_file_errorbars = result_file_errorbars)))
                medians_list.append(group_list[-1].medians)
                errors_up_list.append(group_list[-1].errors_up)
                errors_down_list.append(group_list[-1].errors_down)
                color_id +=1
                if color_id >= len(colors):
                    print "Warning : I don't have enough colors in my list, I'll restart from the beginning."
                    color_id = 0 #reset the color form the beginning

                f.write('Set %s, knotstep : %2.2f, mlknotstep : %2.2f \n' %(name, kn, knml))
                f.write('Tweak ml name : %s \n' % noise)
                f.write('------------------------------------------------ \n')

    #build the bin list :
    medians_list = np.asarray(medians_list)
    errors_down_list = np.asarray(errors_down_list)
    errors_up_list = np.asarray(errors_up_list)
    binslist = []

    for i, lab in enumerate(config.delay_labels):
        bins = np.linspace(min(medians_list[:,i]) - 10 *min(errors_down_list[:,i]), max(medians_list[:,i]) + 10*max(errors_up_list[:,i]), nbins)
        binslist.append(bins)


    color_id = 0
    for g,group in enumerate(group_list):
        group.binslist = binslist
        group.plotcolor = colors[color_id]
        group.linearize(testmode=config.testmode)
        color_id += 1
        if color_id >= len(colors):
            print "Warning : I don't have enough colors in my list, I'll restart from the beginning."
            color_id = 0  # reset the color form the beginning

    combined = copy.deepcopy(pycs.mltd.comb.combine_estimates(group_list, sigmathresh=config.sigmathresh, testmode=config.testmode))
    combined.linearize(testmode=config.testmode)
    combined.name = 'combined $\sigma = %2.2f$'%config.sigmathresh
    combined.plotcolor = 'black'

    print "Final combination for marginalisation ", name_marg_spline
    combined.niceprint()

    #plot the results :

    text = [
        (0.10, 0.90, r"$\mathrm{"+config.full_lensname+"}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
         {"fontsize": 24, "horizontalalignment": "left"})]

    radius = (np.max(errors_up_list) + np.max(errors_down_list))/2.0 *1.5
    if config.display :
        pycs.mltd.plot.delayplot(group_list+[combined], rplot=radius, refgroup=combined, text=text, hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False)

    pycs.mltd.plot.delayplot(group_list+[combined], rplot=radius, refgroup=combined, text=text, hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False, filename = indiv_marg_dir + config.name_marg_spline +"_sigma_%2.2f.png"%config.sigmathresh)

    pkl.dump(group_list, open(marginalisation_dir + config.name_marg_spline +"_sigma_%2.2f"%config.sigmathresh +'_goups.pkl', 'wb'))
    pkl.dump(combined, open(marginalisation_dir + config.name_marg_spline + "_sigma_%2.2f"%config.sigmathresh + '_combined.pkl', 'wb'))

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Marginalise over the spline optimiser parameters.",
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

#TODO : add execption if there is no error bars measured
>>>>>>> f82a52d2dd5305d9ac91e9cbcd5b0a1d49047c27
