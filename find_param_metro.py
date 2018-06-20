#Find micro lensing Power spectrum using MCMC algo

import pycs.regdiff
import pickle
import time
from module.optimisation import mcmc_function as mcmc
import os

start_time = time.time()

object = "HE0435"
kntstp = 40
ml_kntstep =360
picklepath = "./"+object+"/save/"
sim_path = "./"+object+"/simulation_log2/"
plot_path = sim_path + "figure/"
shotnoise = "mcres" #'magerrs' or "mcres"
if not os.path.exists(sim_path):
    os.mkdir(sim_path)
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
niter = 3
nburn = 0
nlcs = 3 #numero de la courbe a traiter
rdm_walk = 'log'
n_curve_stat = 2 #number of curve to optimise to compute the statistic.
max_process = 1
stopping_condition =True

open(sim_path + 'rt_file_' + object +"_"+ picklename[:-4] + "_" + str(niter)+"_"+rdm_walk +"_"+str(nlcs)+'.txt', 'w').close() # to clear the file
rt_file = open(sim_path + 'rt_file_' + object +"_"+ picklename[:-4]  + "_" + str(niter)+"_"+rdm_walk +"_"+str(nlcs)+'.txt','a')

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

#Control the residuals :
rls = pycs.gen.stat.subtract(lcs, spline)
print 'Residuals from the fit : '
print pycs.gen.stat.mapresistats(rls)
fit_sigma = pycs.gen.stat.mapresistats(rls)[nlcs]["std"]
fit_zruns = pycs.gen.stat.mapresistats(rls)[nlcs]["zruns"]
fit_nruns = pycs.gen.stat.mapresistats(rls)[nlcs]["nruns"]
fit_vector = [fit_zruns,fit_sigma]
sigma_step = [0.22,0.005] # standard deviation for gaussian step
pycs.sim.draw.saveresiduals(lcs, spline)

initial_position = [-1.9,0.1]
MH_opt = mcmc.Metropolis_Hasting_Optimiser(lcs[nlcs], fit_vector,spline, gaussian_step = sigma_step,
                                 niter = niter, burntime = nburn, savedirectory = sim_path, recompute_spline= True,
                                knotstep = kntstp, rdm_walk=rdm_walk, n_curve_stat=n_curve_stat,
                                max_core = max_process, stopping_condition=stopping_condition, shotnoise = shotnoise,
                                           tweakml_type = 'colored_noise',theta_init = initial_position)

theta_walk, chi2_walk, sz_walk, errorsz_walk = MH_opt.optimise()
best_chi2, best_param = MH_opt.get_best_param()
MH_opt.analyse_plot_results()
MH_opt.dump_results()
MH_opt.reset_report()
MH_opt.report()

print("--- %s seconds ---" % (time.time() - start_time))
#
# pickle.dump(theta_walk, open(sim_path+"theta_walk_" + object +"_"+ picklename[:-4] + "_" + str(niter)+"_"+rdm_walk + "_"+str(nlcs)+".pkl", "wb" ))
# pickle.dump(chi2_walk, open(sim_path+"chi2_walk_" + object +"_"+ picklename[:-4] + "_" + str(niter) +"_"+rdm_walk + "_"+str(nlcs)+".pkl", "wb" ))
# pickle.dump(sz_walk, open(sim_path+"sz_walk_" + object +"_"+ picklename[:-4] + "_" + str(niter) +"_"+rdm_walk + "_"+str(nlcs)+".pkl", "wb" ))
# pickle.dump(errorsz_walk, open(sim_path+"errorsz_walk_" + object +"_"+ picklename[:-4] + "_" + str(niter) +"_"+rdm_walk + "_"+str(nlcs)+".pkl", "wb" ))
#
#
#
#
