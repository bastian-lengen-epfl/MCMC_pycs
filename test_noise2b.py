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

rls = pycs.gen.stat.subtract([lcs[curve]], spline)
target = pycs.gen.stat.mapresistats(rls)[0]
B_vec = np.linspace(0.1,2,2)
pycs.sim.draw.saveresiduals(lcs, spline)

fit_vector = mcmc.get_fit_vector(lcs[0], spline)
print "I will try to find the parameter for lightcurve :", lcs[0].object

grid_opt = mcmc.Grid_Optimiser(lcs[0], fit_vector, spline, knotstep=kntstp,
                               savedirectory="./", recompute_spline=True, max_core=8,
                               n_curve_stat=2, shotnoise=None, tweakml_type='PS_from_residuals',
                               tweakml_name='test',
                               display=False, verbose=True, grid=B_vec)

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
print "Mock residuals : ",pycs.gen.stat.mapresistats(mockrls)
print "Mock original :", target

pycs.gen.lc.display([lcs[curve]], rls, showdelays=True)
pycs.gen.lc.display(mocklc_besdt_B, mockrls, showdelays=True)

pycs.gen.stat.plotresiduals([mockrls],filename='resinoise.png')
pycs.gen.stat.plotresiduals([rls],filename='resinoise.png')
plt.show()

#todo : REFINE the estimation of A after refitting the spline...