#This script show how you can use the POwer Spectrum of the residuals to tweak your ml spline.

from module import tweakml_PS_from_data as twk
import pycs
from module.optimisation import Optimiser as mcmc
import matplotlib.pyplot as plt

source ="pickle"
object = "HE0435b_Euler"



kntstp = 35
# kntstp = 40
ml_kntstep =150
# ml_kntstep =360
picklepath = "./"+ object +"/spl1_ks"+str(kntstp)+"_splml_ksml_"+str(ml_kntstep) + "/"
picklename = "initopt_Euler_ks%i_ksml%i.pkl"%(kntstp, ml_kntstep)
curve =1

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

A = 1.0 #this is the maximum frequency you want to add (in unit of the Nymquist frequency of your signal)
B = 1.0 #this is the scaling of the power spectrum

rls = pycs.gen.stat.subtract([lcs[curve]], spline)
target = pycs.gen.stat.mapresistats(rls)[0]
# B_vec = np.linspace(0.1,2,2)
B_vec = [1.0]
pycs.sim.draw.saveresiduals(lcs, spline)

fit_vector = mcmc.get_fit_vector(lcs[curve], spline)
print "I will try to find the parameter for lightcurve :", lcs[curve].object

grid_opt = mcmc.Grid_Optimiser(lcs[curve], fit_vector, spline, knotstep=kntstp,
                               savedirectory="./", recompute_spline=True, max_core=8,
                               n_curve_stat=8, shotnoise=None, tweakml_type='PS_from_residuals',
                               tweakml_name='test',
                               display=False, verbose=False, grid=B_vec, correction_PS_residuals= False)

grid_opt.optimise()

chi2, B_min = grid_opt.get_best_param()

mocklc_besdt_B = pycs.sim.draw.draw([lcs[curve]], spline, tweakml=lambda x: twk.tweakml_PS(x, spline, B_min, f_min=1 / 300.0,
                                                                           psplot=False, save_figure_folder=None,
                                                                           verbose=False,
                                                                           interpolation='linear')
                            , shotnoise=None, keeptweakedml=False)

grid_opt.display = True
grid_opt.analyse_plot_results()


#for comparison, plot the mock curves and the real one.
spline_on_mock = pycs.spl.topopt.opt_fine(mocklc_besdt_B, nit=5, knotstep=kntstp, verbose=False)
mockrls = pycs.gen.stat.subtract(mocklc_besdt_B, spline_on_mock)
stats_mock = pycs.gen.stat.mapresistats(mockrls)[0]
print "Mock residuals : ",stats_mock
print "Original residuals : ", target

print "Correction factor to match sigma : ", stats_mock['std'] / target ['std']

# pycs.gen.lc.display([lcs[curve]], rls, showdelays=True)
# pycs.gen.lc.display(mocklc_besdt_B, mockrls, showdelays=True)
#
# pycs.gen.stat.plotresiduals([mockrls],filename='resinoise.png')
# pycs.gen.stat.plotresiduals([rls],filename='resinoise.png')
plt.show()


# grid_opt = mcmc.Grid_Optimiser(lcs[curve], fit_vector, spline, knotstep=kntstp,
#                                savedirectory="./", recompute_spline=True, max_core=1,
#                                n_curve_stat=8, shotnoise=None, tweakml_type='PS_from_residuals',
#                                tweakml_name='test',
#                                display=False, verbose=False, grid=B_vec, correction_PS_residuals= True)
#
# grid_opt.optimise()
#
# chi2, B_min = grid_opt.get_best_param()
#
# mocklc_besdt_B = pycs.sim.draw.draw([lcs[curve]], spline, tweakml=lambda x: twk.tweakml_PS(x, spline, B_min, f_min=1 / 300.0,
#                                                                            psplot=False, save_figure_folder=None,
#                                                                            verbose=False,
#                                                                            interpolation='linear',
#                                                                             A_correction= grid_opt.A_correction)
#                             , shotnoise=None, keeptweakedml=False)
#
# grid_opt.display = True
# grid_opt.analyse_plot_results()
#
#
# #for comparison, plot the mock curves and the real one.
# spline_on_mock = pycs.spl.topopt.opt_fine(mocklc_besdt_B, nit=5, knotstep=kntstp, verbose=False)
# mockrls = pycs.gen.stat.subtract(mocklc_besdt_B, spline_on_mock)
# stats_mock = pycs.gen.stat.mapresistats(mockrls)[0]
# print "Mock residuals with correction : ",stats_mock
# print "Mock original : ", target