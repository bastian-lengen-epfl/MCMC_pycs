#This script show how you can use the POwer Spectrum of the residuals to tweak your ml spline.
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from module import tweakml_PS_from_data as twk
import pycs, os
from module.optimisation import Optimiser as mcmc
import numpy as np
import pickle as pkl

source ="pickle"
object = "HE0435b_Euler"

test_dir = './test/'
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

kntstp = 35
# kntstp = 40
ml_kntstep =150
# ml_kntstep =360
picklepath = "./"+ object +"/spl1_ks"+str(kntstp)+"_splml_ksml_"+str(ml_kntstep) + "/"
picklename = "initopt_Euler_ks%i_ksml%i.pkl"%(kntstp, ml_kntstep)

ncurve = 50
shotnoise = 'magerrs'
suf = '_magerrs_new_'

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)
curves = len(lcs)
rls = pycs.gen.stat.subtract(lcs, spline)
target = pycs.gen.stat.mapresistats(rls)
print target
pycs.sim.draw.saveresiduals(lcs, spline)

execfile(picklepath + "tweakml_colored_noise_magerrs.py")
# execfile(picklepath + "tweakml_PS_noise.py")
# tweak_type = 'PS_from_residuals'
tweak_type = 'colored_noise'

mocklcs = []
optmock_list = []
resi_list = []
stat_list = []
zruns_list = []
sigmas_list=[]

# theta_vec = [[0.3,0.94], [0.8,0.91], [0.75,0.91], [0.35,0.95]]
theta_vec = [[-2.95,0.001], [-0.5,0.511], [-0.1,0.51], [-2.95,0.001]]
# theta_vec = [[-2.95,0.001], [-2.95,0.001], [-2.95,0.001], [-2.95,0.001]]

# for (lc,theta) in zip(lcs,theta_vec) :
#     optim = mcmc.Optimiser(lc, [0,0], spline, knotstep=kntstp,
#                          savedirectory="./", recompute_spline=True, max_core = 8, theta_init = [-2.0 ,0.1],
#                         n_curve_stat = 32, shotnoise = None, tweakml_type = 'PS_from_residuals', display = False, verbose = False,
#                      tweakml_name = '', correction_PS_residuals = True)
#
#     optim.A_correction = theta[1]
#     print theta[0], optim.A_correction
#     [mz,mstd], [sz, sstd], zruns, sigmas = optim.make_mocks_para(theta)
#     zruns_list.append(zruns)
#     sigmas_list.append(sigmas)

optim = mcmc.Optimiser(lcs, target, spline, knotstep=kntstp,
                     savedirectory="./", recompute_spline=True, max_core = 8, theta_init = theta_vec,
                    n_curve_stat = ncurve, shotnoise = shotnoise, tweakml_type = tweak_type, display = False, verbose = False,
                 tweakml_name = '', correction_PS_residuals = True)

[mz,mstd], [sz, sstd], zruns, sigmas = optim.make_mocks_para(theta_vec)

print 'Shape :',np.shape(zruns), np.shape(sigmas)
zruns = np.asarray(zruns)
sigmas = np.asarray(sigmas)

fig1 = plt.figure(figsize=(3 * len(lcs), 4))
plt.subplots_adjust(left=0.02, bottom=0.12, right=0.98, top=0.98, wspace=0.08, hspace=0.37)
for i in range(curves):
    # print (1, len(curves), i+1)
    plt.subplot(2, curves, i + 1)
    plt.hist(sigmas[:,i], 50, facecolor='black',
             alpha=0.4, histtype="stepfilled")
    plt.axvline(target[i]["std"], color="green", linewidth=2.0, alpha=0.7)
    plt.axvline(np.mean(sigmas[:,i]), color="red", linewidth=2.0, alpha=0.7)
    plt.xlabel("Spline fit residuals [mag]")

    # print plt.gca().get_ylim()
    # plt.xlim(-r, r)
    # plt.gca().get_yaxis().set_ticks([])

# zruns histos :

    # print (1, len(curves), i+1)
    plt.subplot(2, curves, curves + i + 1)

    plt.hist(zruns[:,i], 20, facecolor="black", alpha=0.4,
             histtype="stepfilled")
    plt.axvline(target[i]["zruns"], color="green", linewidth=2.0, alpha=0.7)
    plt.axvline(np.mean(zruns[:,i]), color="red", linewidth=2.0, alpha=0.7)

    plt.xlabel(r"$z_{\mathrm{r}}$", fontsize=18)
    # plt.xlim(-5.0, 5.0)

    # plt.text(-9.0, 0.85*plt.gca().get_ylim()[1], curve["optorigrlc"].object, fontsize=20)
    plt.gca().get_yaxis().set_ticks([])

    print "Curve ", i
    print 'Mean zruns (simu): ', np.mean(zruns[:,i]), '+/-', np.std(zruns[:,i])
    print 'Mean sigmas (simu): ', np.mean(sigmas[:,i]), '+/-', np.std(sigmas[:,i])

fig1.savefig(test_dir + 'histplot'+suf+'.png')
plt.show()
exit()





for i in range(ncurve) :
    mocklcs.append(pycs.sim.draw.draw(lcs, spline, tweakml=tweakml_list
                            , shotnoise=shotnoise, keeptweakedml=False))

for i, m in enumerate(mocklcs):
    print "optimisation curves ", i
    optmock = pycs.spl.topopt.opt_fine(m, nit=5, knotstep=kntstp, verbose=False)
    resi = pycs.gen.stat.subtract(m, optmock)
    stats_mock = pycs.gen.stat.mapresistats(resi)
    print stats_mock

    resi_list.append(resi)
    optmock_list.append(optmock)
    stat_list.append(stats_mock)

stat_list = np.asarray(stat_list)
print np.shape(stat_list)
print stat_list[:,0]

pkl.dump(optmock_list, open(test_dir + 'optmock'+suf+'.pkl', 'wb'))
pkl.dump(mocklcs, open(test_dir + 'mock'+suf+'.pkl', 'wb'))
pkl.dump(sigmas_list, open(test_dir + 'sigma_optim'+suf+'.pkl', 'wb'))
pkl.dump(zruns_list, open(test_dir + 'zruns_optim'+suf+'.pkl', 'wb'))

fig2 = plt.figure(figsize=(3 * len(lcs), 4))
plt.subplots_adjust(left=0.02, bottom=0.12, right=0.98, top=0.98, wspace=0.08, hspace=0.37)

# # Resi histos :
for i in range(curves):
    stat = stat_list[:, i]
    zruns = [stat[j]['zruns'] for j in range(len(stat))]
    std = [stat[j]['std'] for j in range(len(stat))]
    # print (1, len(curves), i+1)
    plt.subplot(2, curves, i + 1)
    plt.hist(std, 50, facecolor='black',
             alpha=0.4, histtype="stepfilled")
    plt.axvline(target[i]["std"], color="green", linewidth=2.0, alpha=0.7)
    plt.axvline(np.mean(std), color="red", linewidth=2.0, alpha=0.7)
    plt.xlabel("Spline fit residuals [mag]")

    # print plt.gca().get_ylim()
    # plt.xlim(-r, r)
    # plt.gca().get_yaxis().set_ticks([])

# zruns histos :

    # print (1, len(curves), i+1)
    plt.subplot(2, curves, curves + i + 1)

    plt.hist(zruns, 20, facecolor="black", alpha=0.4,
             histtype="stepfilled")
    plt.axvline(target[i]["zruns"], color="green", linewidth=2.0, alpha=0.7)
    plt.axvline(np.mean(zruns), color="red", linewidth=2.0, alpha=0.7)

    plt.xlabel(r"$z_{\mathrm{r}}$", fontsize=18)
    # plt.xlim(-5.0, 5.0)

    # plt.text(-9.0, 0.85*plt.gca().get_ylim()[1], curve["optorigrlc"].object, fontsize=20)
    plt.gca().get_yaxis().set_ticks([])

    print "Curve ", i
    print 'Mean zruns (simu): ', np.mean(zruns), '+/-', np.std(zruns)
    print 'Mean sigmas (simu): ', np.mean(std), '+/-', np.std(std)

fig2.savefig(test_dir + "histdist2"+suf+".png")
plt.show()


