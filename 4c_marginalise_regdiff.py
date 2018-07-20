import sys
import pycs
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

execfile("config.py")
marginalisation_plot_dir = figure_directory + 'marginalisation_plots/'

if not os.path.isdir(marginalisation_plot_dir):
    os.mkdir(marginalisation_plot_dir)

indiv_marg_dir = marginalisation_plot_dir + name_marg_regdiff + '/'
if not os.path.isdir(indiv_marg_dir):
    os.mkdir(indiv_marg_dir)

marginalisation_dir = lens_directory + name_marg_regdiff + '/'
if not os.path.isdir(marginalisation_dir):
    os.mkdir(marginalisation_dir)

f = open(marginalisation_dir + 'report_%s_sigma%2.1f.txt'%(name_marg_regdiff,sigmathresh), 'w')

if testmode:
    nbins = 500
else:
    nbins = 5000

colors = ["royalblue", "crimson", "seagreen", "darkorchid", "darkorange", 'indianred', 'purple', 'brown', 'black', 'violet', 'paleturquoise', 'palevioletred', 'olive',
          'indianred', 'salmon','lightcoral', 'chocolate', 'indigo', 'steelblue' , 'cyan', 'gold']
color_id = 0

group_list = []
medians_list = []
errors_up_list = []
errors_down_list = []
simset_mock_ava = ["mocks_n%it%i_%s" % (int(nsim * nsimpkls), truetsr,i) for i in tweakml_name_marg_regdiff]
opt = 'regdiff'

if auto_all :
    pass
#TODO write this : true marginalisation over regdiff params + take the most precise estimate over the knotstep...

else :
    count = 0
    for a,kn in enumerate(knotstep_marg_regdiff):
        for b, knml in enumerate(mlknotsteps_marg_regdiff):
            for c in covkernel_marg:
                for pts in pointdensity_marg:
                    for p in pow_marg:
                        for am in amp_marg:
                            for s in scale_marg:
                                for e in errscale_marg:
                                    for n, noise in enumerate(tweakml_name_marg_regdiff):
                                        count +=1
                                        if c == 'gaussian' :
                                            paramskw = "regdiff_pd%i_ck%s_amp%.1f_sc%i_errsc%i_" % (pts, c, am, s, e)
                                        else :
                                            paramskw = "regdiff_pd%i_ck%s_pow%.1f_amp%.1f_sc%i_errsc%i_" % (pts, c, p, am, s, e)
                                        print count
                                        result_file_delay = lens_directory + combkw[a, b] + '/sims_%s_opt_%st%i/' \
                                                            %(simset_copy, paramskw, int(tsrand)) \
                                                            + 'sims_%s_opt_%s' % (simset_copy, paramskw) + 't%i_delays.pkl'%int(tsrand)
                                        result_file_errorbars = lens_directory + combkw[a, b] + '/sims_%s_opt_%st%i/' \
                                                                % (simset_copy, paramskw, int(tsrand))\
                                                                + 'sims_%s_opt_%s' % (simset_mock_ava[n], paramskw) + \
                                                                't%i_errorbars.pkl'%int(tsrand)

                                        if not os.path.isfile(result_file_delay) or not os.path.isfile(result_file_errorbars):
                                            print 'Error I cannot find the files %s or %s. ' \
                                                  'Did you run the 3c and 4a?'%(result_file_delay, result_file_errorbars)
                                            f.write('Error I cannot find the files %s or %s. \n'%(result_file_delay, result_file_errorbars))
                                            continue

                                        group_list.append(pycs.mltd.comb.getresults(
                                            pycs.mltd.comb.CScontainer(data=dataname, knots=kn, ml=knml,
                                                                       name="set %i" %count,
                                                                       drawopt=optfctkw, runopt=opt, ncopy=ncopy*ncopypkls,
                                                                       nmocks=nsim*nsimpkls, truetsr=truetsr,
                                                                       colour=colors[color_id],
                                                                       result_file_delays= result_file_delay,
                                                                       result_file_errorbars = result_file_errorbars)))
                                        medians_list.append(group_list[-1].medians)
                                        errors_up_list.append(group_list[-1].errors_up)
                                        errors_down_list.append(group_list[-1].errors_down)
                                        color_id +=1
                                        if color_id >= len(colors):
                                            print "Warning : I don't have enough colors in my list, I'll restart from the beginning."
                                            color_id = 0 #reset the color form the beginning

                                        f.write('Set %i, knotstep : %2.2f, mlknotstep : %2.2f \n'%(count,kn,knml))
                                        f.write('covkernel : %s, point density: %2.2f, pow : %2.2f, amp : %2.2f, '
                                                'scale:%2.2f, errscale:%2.2f \n'%(c,pts,p,am,s,e))
                                        f.write('Tweak ml name : %s \n'%noise)
                                        f.write('------------------------------------------------ \n')



#build the bin list :
medians_list = np.asarray(medians_list)
errors_down_list = np.asarray(errors_down_list)
errors_up_list = np.asarray(errors_up_list)
binslist = []
for i, lab in enumerate(delay_labels):
    bins = np.linspace(min(medians_list[:,i]) - 10 *min(errors_down_list[:,i]), max(medians_list[:,i]) + 10*max(errors_up_list[:,i]), nbins)
    binslist.append(bins)


color_id = 0
for g,group in enumerate(group_list):
    group.binslist = binslist
    group.plotcolor = colors[color_id]
    group.linearize(testmode=testmode)
    color_id += 1
    if color_id >= len(colors):
        print "Warning : I don't have enough colors in my list, I'll restart from the beginning."
        color_id = 0  # reset the color form the beginning

combined= pycs.mltd.comb.combine_estimates(group_list, sigmathresh=sigmathresh, testmode=testmode)
combined.linearize(testmode=testmode)
combined.name = 'combined $\sigma = %2.2f$'%sigmathresh

print "Final combination for marginalisation ", name_marg_regdiff
combined.niceprint()

#plot the results :

text = [
	(0.75, 0.92, r"$\mathrm{"+full_lensname+"}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
	 {"fontsize": 26, "horizontalalignment": "center"})]

if display :
    pycs.mltd.plot.delayplot(group_list+[combined], rplot=8.0, refgroup=combined, text=text, hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False)

pycs.mltd.plot.delayplot(group_list+[combined], rplot=8.0, refgroup=combined, text=text, hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False, filename = indiv_marg_dir + name_marg_regdiff +"_sigma_%2.2f.png"%sigmathresh)

pkl.dump(group_list, open(marginalisation_dir + name_marg_regdiff +"_sigma_%2.2f"%sigmathresh +'_goups.pkl', 'wb'))
pkl.dump(combined, open(marginalisation_dir + name_marg_regdiff + "_sigma_%2.2f"%sigmathresh + '_combined.pkl', 'wb'))


