from module import tweakml_PS_from_data as twk
from module.optimisation import grid_search_PS as grid
import pycs
import numpy as np
from module.optimisation import mcmc_function as mcmc
import matplotlib.pyplot as plt

source ="pickle"
object = "UM673_Euler"

picklepath = "./"+object+"/save/"

kntstp = 60
# kntstp = 40
ml_kntstep =500
# ml_kntstep =360
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
curve =0

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

A = 1.0 #this is the maximum frequency you want to add (in unit of the Nymquist frequency of your signal)
B = 1.0 #this is the scaling of the power spectrum

#EXEMPLE :
# lcs[curve].ml.spline.display()
# print "Nb coefficient before tweak :", len(lcs[curve].ml.spline.c)
# pycs.sim.twk.tweakml([lcs[curve]], beta=-1.0, sigma=1.0, fmin=1.0/50.0, fmax=0.2, psplot=False)
# lcs[curve].ml.spline.display()
# print "Nb coefficient after tweak :", len(lcs[curve].ml.spline.c)
# #
# #
# # #To check what does the microlensing curve looks like
rls = pycs.gen.stat.subtract([lcs[curve]], spline)
# lcs[curve].ml.spline.display()
# print "Nb coefficient before tweak :", len(lcs[curve].ml.spline.c)
# print "before tweak :", pycs.gen.stat.resistats(rls[0])
# twk.tweakml_PS([lcs[curve]],spline, B,f_min = 1/300.0,save_figure_folder=None,  psplot=True, verbose =True, interpolation = 'linear')
# lcs[curve].ml.spline.display()
# print "Nb coefficient after tweak :", len(lcs[curve].ml.spline.c)


target = pycs.gen.stat.mapresistats(rls)[0]
B_vec = np.linspace(0.1,2,50)
pycs.sim.draw.saveresiduals(lcs, spline)
success, B_min, [zruns,sigma], [zruns_std,sigma_std], chi2,min_ind = grid.grid_search_PS(lcs[curve],spline,B_vec, target,  max_core=8, n_curve_stat=16, verbose = True, shotnoise = None, knotstep= kntstp)

#
mocklc_besdt_B = pycs.sim.draw.draw([lcs[curve]], spline, tweakml=lambda x: twk.tweakml_PS(x, spline, B_min, f_min=1 / 300.0,
                                                                           psplot=False, save_figure_folder=None,
                                                                           verbose=False,
                                                                           interpolation='linear')
                            , shotnoise=None, keeptweakedml=False)


plt.figure(1)
plt.errorbar(B_vec,zruns, yerr= zruns_std)
plt.hlines(target['zruns'], B_vec[0], B_vec[-1], colors='r', linestyles='solid', label='target')
plt.xlabel('B in unit of Nymquist frequency)')
plt.legend()
plt.ylabel('zruns')
plt.figure(2)
plt.errorbar(B_vec,sigma, yerr= sigma_std)
plt.hlines(target['std'], B_vec[0], B_vec[-1], colors='r', linestyles='solid', label='target')
plt.xlabel('B in unit of Nymquist frequency)')
plt.legend()
plt.ylabel('sigma')
plt.show()


#for comparison, plot the mock curves and the real one.
spline_on_mock = pycs.spl.topopt.opt_fine(mocklc_besdt_B, nit=5, knotstep=kntstp, verbose=False)
mockrls = pycs.gen.stat.subtract(mocklc_besdt_B, spline_on_mock)
print "Mock residuals : ",pycs.gen.stat.mapresistats(mockrls)
print "Mock original :", target

pycs.gen.lc.display([lcs[curve]], rls, showdelays=True)
pycs.gen.lc.display(mocklc_besdt_B, mockrls, showdelays=True)

pycs.gen.stat.plotresiduals([mockrls],filename='resinoise.png')
pycs.gen.stat.plotresiduals([rls],filename='resinoise.png')


#todo : REFINE the estimation of A after refitting the spline...