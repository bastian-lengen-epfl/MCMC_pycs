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
sim_path = "./"+object+"/simulation_PSO/"
plot_path = sim_path + "figure/"
shotnoise = "mcres" #'magerrs' or "mcres"
if not os.path.exists(sim_path):
    os.mkdir(sim_path)
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

n_particles = 5
n_iterations = 2
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
nlcs = 3 #numero de la courbe a traiter
n_curve_stat = 2 #number of curve to optimise to compute the statistic.
max_process = 8
mpi = False

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)
open(sim_path + 'rt_file_PSO_' + object +"_"+ picklename[:-4] + "_i"
                             + str(n_iterations)+"_p"+str(n_particles)+ "_" +str(nlcs)+".txt", "w").close() # to clear the file
rt_file = sim_path + 'rt_file_PSO_' + object +"_"+ picklename[:-4] + "_i" + str(n_iterations)+"_p"+str(n_particles)+ "_" +str(nlcs)+".txt"

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
lowerLimit = [-8., 0.]
upperLimit = [-1.0, 0.5]

PSO_opt = mcmc.PSO_Optimiser(lcs[nlcs], fit_vector, spline, savedirectory= sim_path,
                              knotstep=kntstp, max_core = max_process, shotnoise = shotnoise,
                              recompute_spline = True, n_curve_stat= n_curve_stat, theta_init= initial_position,
                             n_particles=n_particles, n_iter=n_iterations, lower_limit = lowerLimit, upper_limit = upperLimit, mpi = False)

chain_list = PSO_opt.optimise()
best_chi2, best_param = PSO_opt.get_best_param()
PSO_opt.analyse_plot_results()
PSO_opt.reset_report()
PSO_opt.report()
PSO_opt.dump_results()

# pickle.dump(chain_list, open(sim_path+"chain_PSO_" + object +"_"+ picklename[:-4] + "_i"
#                              + str(n_iterations)+"_p"+str(n_particles)+ "_" +str(nlcs)+".pkl", "wb" ))
# pickle.dump(PSO_opt, open(sim_path+"PSO_opt_" + object +"_"+ picklename[:-4] + "_i"
#                              + str(n_iterations)+"_p"+str(n_particles)+ "_" +str(nlcs)+".pkl", "wb" ))


print("--- %s seconds ---" % (time.time() - start_time))




