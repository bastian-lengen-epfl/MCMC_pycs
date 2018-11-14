import sys
import pycs, copy
import os
import numpy as np
import pickle as pkl
import importlib
import argparse as ap
from module import util_func as ut


def main(lensname,dataname,work_dir='./'):
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)
    marginalisation_plot_dir = config.figure_directory + 'marginalisation_plots/'
    regdiff_dir = os.path.join(config.lens_directory, "regdiff_outputs/")
    regdiff_copie_dir = os.path.join(regdiff_dir, "copies/")

    if not os.path.isdir(marginalisation_plot_dir):
        os.mkdir(marginalisation_plot_dir)

    indiv_marg_dir = marginalisation_plot_dir + config.name_marg_regdiff + '/'
    if not os.path.isdir(indiv_marg_dir):
        os.mkdir(indiv_marg_dir)

    marginalisation_dir = config.lens_directory + config.name_marg_regdiff + '/'
    if not os.path.isdir(marginalisation_dir):
        os.mkdir(marginalisation_dir)

    if config.testmode:
        nbins = 500
    else:
        nbins = 5000

    colors = ["royalblue", "crimson", "seagreen", "darkorchid", "darkorange", 'indianred', 'purple', 'brown', 'black',
              'violet', 'dodgerblue', 'palevioletred', 'olive',
              'brown', 'salmon', 'chocolate', 'indigo', 'steelblue', 'cyan', 'gold' , 'lightcoral']

    f = open(marginalisation_dir + 'report_%s_sigma%2.1f.txt' % (config.name_marg_regdiff, config.sigmathresh), 'w')
    path_list = []
    name_list = []
    combkw_marg = [["%s_ks%i_%s_ksml_%i" % (config.optfctkw, config.knotstep_marg[i], config.mlname, config.mlknotsteps_marg[j])
               for j in range(len(config.mlknotsteps_marg))] for i in range(len(config.knotstep_marg))]
    combkw_marg = np.asarray(combkw_marg)

    kw_list = ut.read_preselected_regdiffparamskw(config.preselection_file)
    kw_dic = ut.get_keyword_regdiff_from_file(config.preselection_file)
    for s,set in enumerate(kw_dic):
        if not 'name' in set.keys() :
            set['name']='Set %i'%s

    for paramskw, dickw in zip(kw_list, kw_dic):
        for n, noise in enumerate(config.tweakml_name_marg_regdiff):

            count = 0
            color_id = 0

            group_list = []
            medians_list = []
            errors_up_list = []
            errors_down_list = []
            simset_mock_ava = ["mocks_n%it%i_%s" % (int(config.nsim * config.nsimpkls), config.truetsr,twk) for twk in config.tweakml_name_marg_regdiff]
            opt = 'regdiff'

            if config.auto_marginalisation:
                if config.use_preselected_regdiff == False :
                    raise RuntimeError("Turn the use_preselected_regdiff to True and set your preselection_file before using auto_marginalisation.")

                for a, kn in enumerate(config.knotstep_marg_regdiff):
                    for b, knml in enumerate(config.mlknotsteps_marg_regdiff):

                        regdiff_mocks_dir = os.path.join(regdiff_dir, "mocks_knt%i_mlknt%i/" %(kn, knml))

                        result_file_delay = regdiff_copie_dir + 'sims_%s_opt_regdiff%s' % (config.simset_copy, paramskw) \
                                            + 't%i_delays.pkl' % int(config.tsrand)
                        result_file_errorbars = regdiff_mocks_dir \
                                                + 'sims_%s_opt_regdiff%s' % (simset_mock_ava[n], paramskw) + \
                                                't%i_errorbars.pkl' % int(config.tsrand)

                        if not os.path.isfile(result_file_delay) or not os.path.isfile(result_file_errorbars):
                            print 'Error I cannot find the files %s or %s. ' \
                                  'Did you run the 3c and 4a?' % (result_file_delay, result_file_errorbars)
                            f.write('Error I cannot find the files %s or %s. \n' % (
                            result_file_delay, result_file_errorbars))
                            continue

                        group_list.append(pycs.mltd.comb.getresults(
                            pycs.mltd.comb.CScontainer(data=dataname, knots=kn, ml=knml,
                                                       name="knstp %i mlknstp %i"%(kn,knml),
                                                       drawopt=config.optfctkw, runopt=opt,
                                                       ncopy=config.ncopy * config.ncopypkls,
                                                       nmocks=config.nsim * config.nsimpkls, truetsr=config.truetsr,
                                                       colour=colors[color_id],
                                                       result_file_delays=result_file_delay,
                                                       result_file_errorbars=result_file_errorbars)))
                        medians_list.append(group_list[-1].medians)
                        errors_up_list.append(group_list[-1].errors_up)
                        errors_down_list.append(group_list[-1].errors_down)

                        if np.isnan(medians_list[-1]).any() or np.isnan(errors_up_list[-1]).any() or np.isnan(errors_down_list[-1]).any():
                            print "There is some Nan value in %s, for noise %s, kn %i, knml%i"%(dickw['name'], noise, kn,knml)
                            print "I could erase this entry and continue the marginalisation. "
                            ut.proquest(True)
                            medians_list = medians_list[:-1]
                            errors_down_list = errors_down_list[:-1]
                            errors_up_list = errors_down_list[:-1]
                            group_list = group_list[:-1]
                            continue
                        color_id += 1
                        count +=1
                        if color_id >= len(colors):
                            print "Warning : I don't have enough colors in my list, I'll restart from the beginning."
                            color_id = 0  # reset the color form the beginning

                        f.write('Set %i, knotstep : %2.2f, mlknotstep : %2.2f \n' % (count, kn, knml))
                        f.write('covkernel : %s, point density: %2.2f, pow : %2.2f, amp : %2.2f, '
                                'scale:%2.2f, errscale:%2.2f \n' % (dickw["covkernel"], dickw["pointdensity"],
                                                                    dickw["pow"],dickw["amp"],dickw["scale"],
                                                                    dickw["errscale"]))
                        f.write('Tweak ml name : %s \n' % noise)
                        f.write('------------------------------------------------ \n')


            else :
                print "This is deprecated sorry... Use auto_marginalisation in your config file ! " #TODO : supress this option !
                exit()
                # count = 0
                # for c in config.covkernel_marg:
                #     for pts in config.pointdensity_marg:
                #         for p in config.pow_marg:
                #             for am in config.amp_marg:
                #                 for s in config.scale_marg:
                #                     for e in config.errscale_marg:
                #                         for n, noise in enumerate(config.tweakml_name_marg_regdiff):
                #                             count +=1
                #                             if c == 'gaussian' :
                #                                 paramskw = "regdiff_pd%i_ck%s_amp%.1f_sc%i_errsc%i_" % (pts, c, am, s, e)
                #                             else :
                #                                 paramskw = "regdiff_pd%i_ck%s_pow%.1f_amp%.1f_sc%i_errsc%i_" % (pts, c, p, am, s, e)
                #                             print count
                #                             result_file_delay = config.lens_directory + combkw_marg[a, b] + '/sims_%s_opt_%st%i/' \
                #                                                 %(config.simset_copy, paramskw, int(config.tsrand)) \
                #                                                 + 'sims_%s_opt_%s' % (config.simset_copy, paramskw) + 't%i_delays.pkl'%int(config.tsrand)
                #                             result_file_errorbars = config.lens_directory + combkw_marg[a, b] + '/sims_%s_opt_%st%i/' \
                #                                                     % (config.simset_copy, paramskw, int(config.tsrand))\
                #                                                     + 'sims_%s_opt_%s' % (simset_mock_ava[n], paramskw) + \
                #                                                     't%i_errorbars.pkl'%int(config.tsrand)
                #
                #                             if not os.path.isfile(result_file_delay) or not os.path.isfile(result_file_errorbars):
                #                                 print 'Error I cannot find the files %s or %s. ' \
                #                                       'Did you run the 3c and 4a?'%(result_file_delay, result_file_errorbars)
                #                                 f.write('Error I cannot find the files %s or %s. \n'%(result_file_delay, result_file_errorbars))
                #                                 continue
                #
                #                             group_list.append(pycs.mltd.comb.getresults(
                #                                 pycs.mltd.comb.CScontainer(data=dataname, knots=kn, ml=knml,
                #                                                            name="set %i" %(count+1),
                #                                                            drawopt=config.optfctkw, runopt=opt, ncopy=config.ncopy*config.ncopypkls,
                #                                                            nmocks=config.nsim*config.nsimpkls, truetsr=config.truetsr,
                #                                                            colour=colors[color_id],
                #                                                            result_file_delays= result_file_delay,
                #                                                            result_file_errorbars = result_file_errorbars)))
                #                             medians_list.append(group_list[-1].medians)
                #                             errors_up_list.append(group_list[-1].errors_up)
                #                             errors_down_list.append(group_list[-1].errors_down)
                #                             color_id +=1
                #                             if color_id >= len(colors):
                #                                 print "Warning : I don't have enough colors in my list, I'll restart from the beginning."
                #                                 color_id = 0 #reset the color form the beginning
                #
                #                             f.write('Set %i, knotstep : %2.2f, mlknotstep : %2.2f \n'%(count,kn,knml))
                #                             f.write('covkernel : %s, point density: %2.2f, pow : %2.2f, amp : %2.2f, '
                #                                     'scale:%2.2f, errscale:%2.2f \n'%(c,pts,p,am,s,e))
                #                             f.write('Tweak ml name : %s \n'%noise)
                #                             f.write('------------------------------------------------ \n')

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

            combined= copy.deepcopy(pycs.mltd.comb.combine_estimates(group_list, sigmathresh=1000.0, testmode=config.testmode))
            combined.linearize(testmode=config.testmode)
            combined.name = 'Most precise'

            print "%s : Taking the best of all spline parameters for regdiff parameters set %s"%(config.name_marg_regdiff, dickw['name'])
            combined.niceprint()

            #plot the results :

            text = [
                (0.80, 0.90, r"$\mathrm{"+config.full_lensname+"}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
                 {"fontsize": 26, "horizontalalignment": "center"})]

            if config.display :
                pycs.mltd.plot.delayplot(group_list+[combined], rplot=25.0, refgroup=combined,
                                         text=text, hidedetails=True, showbias=False, showran=False,
                                         showlegend=True, figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False)

            pycs.mltd.plot.delayplot(group_list+[combined], rplot=25.0, refgroup=combined, text=text, hidedetails=True,
                                     showbias=False, showran=False, showlegend=True, figsize=(15, 10), horizontaldisplay=False,
                                     legendfromrefgroup=False, filename = indiv_marg_dir + config.name_marg_regdiff + "_%s_%s.png"%(dickw['name'],noise))

            pkl.dump(group_list, open(marginalisation_dir + config.name_marg_regdiff +"_%s_%s"%(dickw['name'],noise) +'_goups.pkl', 'wb'))
            pkl.dump(combined, open(marginalisation_dir + config.name_marg_regdiff + "_%s_%s"%(dickw['name'],noise) + '_combined.pkl', 'wb'))
            path_list.append(marginalisation_dir + config.name_marg_regdiff + "_%s_%s"%(dickw['name'],noise) + '_combined.pkl')
            name_list.append('%s, Noise : %s '%(dickw['name'], noise))


    ###################  MAKE THE FINAL REGDIFF ESTIMATE ####################
    final_groups, final_combined = ut.group_estimate(path_list, name_list, config.delay_labels, colors, config.sigmathresh, config.name_marg_regdiff, testmode = config.testmode)
    text = [
        (0.80, 0.90, r"$\mathrm{" + config.full_lensname + "}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
         {"fontsize": 26, "horizontalalignment": "center"})]

    if config.display:
        pycs.mltd.plot.delayplot(final_groups + [final_combined], rplot=25.0, refgroup=final_combined,
                                 text=text, hidedetails=True, showbias=False, showran=False,
                                 showlegend=True, figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False)

    pycs.mltd.plot.delayplot(final_groups + [final_combined], rplot=25.0, refgroup=final_combined, text=text, hidedetails=True,
                             showbias=False, showran=False, showlegend=True, figsize=(15, 10), horizontaldisplay=False,
                             legendfromrefgroup=False,
                             filename=indiv_marg_dir + config.name_marg_regdiff + "_final_sigma_%2.2f.png" % (config.sigmathresh))

    pkl.dump(group_list, open(marginalisation_dir + config.name_marg_regdiff + "_final_sigma_%2.2f" % (config.sigmathresh) + '_goups.pkl', 'wb'))
    pkl.dump(combined, open(marginalisation_dir + config.name_marg_regdiff + "_final_sigma_%2.2f" % (config.sigmathresh) + '_combined.pkl', 'wb'))


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Marginalise over the regdiff optimiser parameters.",
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
