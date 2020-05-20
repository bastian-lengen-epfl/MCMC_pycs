import sys
sys.path.append("..")
from module import tweakml_PS_from_data as twk
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pycs
import numpy as np
from module.optimisation import Optimiser as mcmc

source ="pickle"
object = "WG0214"
dataname="WFI"

kntstp = 15
ml_kntstep =50

picklepath = "../Simulation/" + object +'_'+dataname + '/' + 'spl1_ks%i_splml_ksml_%i/'%(kntstp, ml_kntstep)
picklename ="initopt_%s_ks%i_ksml%i.pkl"%(dataname, kntstp, ml_kntstep)

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)
fit_vector = mcmc.get_fit_vector(lcs, spline)
pycs.sim.draw.saveresiduals(lcs, spline)

dic_opt = mcmc.Dic_Optimiser(lcs, fit_vector, spline, knotstep=kntstp,
                             savedirectory='./test_DIC/',
                             recompute_spline=True, max_core=8,
                             n_curve_stat=8,
                             shotnoise=None, tweakml_type='PS_from_residuals',
                             tweakml_name='PS', display=False, verbose=False,
                             correction_PS_residuals=True, max_iter=4)

chain = dic_opt.optimise()
dic_opt.analyse_plot_results()
chi2, B_best = dic_opt.get_best_param()
A = dic_opt.A_correction
dic_opt.reset_report()
dic_opt.report()