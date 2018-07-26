import pycs
import os, copy
import numpy as np
import pickle as pkl

execfile("config.py")
marginalisation_plot_dir = figure_directory + 'marginalisation_plots/'

if not os.path.isdir(marginalisation_plot_dir):
    os.mkdir(marginalisation_plot_dir)

indiv_marg_dir = marginalisation_plot_dir + new_name_marg + '/'
if not os.path.isdir(indiv_marg_dir):
    os.mkdir(indiv_marg_dir)

marginalisation_dir = lens_directory + new_name_marg + '/'
if not os.path.isdir(marginalisation_dir):
    os.mkdir(marginalisation_dir)

colors = ["royalblue", "crimson", "seagreen", "darkorchid", "darkorange", 'indianred', 'purple', 'brown', 'black', 'violet', 'paleturquoise', 'palevioletred', 'olive',
          'indianred', 'salmon','lightcoral', 'chocolate', 'indigo', 'steelblue' , 'cyan', 'gold']

if len(name_marg_list) != len(sigmathresh_list):
    print "Error : name_marg_list and sigmathresh_list must haev the same size ! "
    exit()

if testmode:
    nbins = 500
else:
    nbins = 5000

group_list = []
medians_list = []
errors_up_list = []
errors_down_list = []

for i,marg in enumerate(name_marg_list):
    path = lens_directory + marg + '/' + marg +'_sigma_%2.2f'%sigmathresh_list[i] + '_combined.pkl'
    if not os.path.isfile(path):
        print "Warning : I cannot find %s. I will skip this one. Be careful !" %path
        continue

    group = pkl.load(open(path, 'rb'))
    group.name = marg
    group_list.append(group)
    medians_list.append(group.medians)
    errors_up_list.append(group.errors_up)
    errors_down_list.append(group.errors_down)

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
    group.plotcolor = colors[color_id]
    group.binslist = binslist
    group.linearize(testmode=testmode)
    color_id += 1
    if color_id >= len(colors):
        print "Warning : I don't have enough colors in my list, I'll restart from the beginning."
        color_id = 0  # reset the color form the beginning


combined = copy.deepcopy(pycs.mltd.comb.combine_estimates(group_list, sigmathresh=sigmathresh, testmode=testmode))
combined.linearize(testmode=testmode)
combined.name = 'combined $\sigma = %2.2f$'%sigmathresh
combined.plotcolor = 'black'
print "Final combination for marginalisation ", new_name_marg
combined.niceprint()

#plot the results :

text = [
	(0.75, 0.92, r"$\mathrm{"+full_lensname+"}$" + "\n" + r"$\mathrm{PyCS\ estimates}$",
	 {"fontsize": 26, "horizontalalignment": "center"})]

if display :
    pycs.mltd.plot.delayplot(group_list+[combined], rplot=8.0, refgroup=combined, text=text, hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False)

pycs.mltd.plot.delayplot(group_list+[combined], rplot=8.0, refgroup=combined, text=text, hidedetails=True, showbias=False, showran=False, showlegend=True, figsize=(15, 10), horizontaldisplay=False, legendfromrefgroup=False, filename = indiv_marg_dir + name_marg_spline +"_sigma_%2.2f.png"%sigmathresh)

pkl.dump(group_list, open(marginalisation_dir + new_name_marg +"_sigma_%2.2f"%sigmathresh +'_goups.pkl', 'wb'))
pkl.dump(combined, open(marginalisation_dir + new_name_marg + "_sigma_%2.2f"%sigmathresh + '_combined.pkl', 'wb'))