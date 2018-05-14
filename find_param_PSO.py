#Find micro lensing Power spectrum using MCMC algo

import pycs
import pycs.regdiff
import pickle
import time
import mcmc_function as mcmc
import os
import numpy as np
from cosmoHammer import MpiParticleSwarmOptimizer
from cosmoHammer import ParticleSwarmOptimizer

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

n_particles = 50
n_iterations = 20
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
nlcs = 3 #numero de la courbe a traiter
n_curve_stat = 1 #number of curve to optimise to compute the statistic.
max_process = 8
stopping_condition =True
mpi = False

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)
open(sim_path + 'rt_file_' + object +"_"+ picklename[:-4] + "_PSO_"+str(nlcs)+'.txt', 'w').close() # to clear the file
rt_file = open(sim_path + 'rt_file_' + object +"_"+ picklename[:-4]  + "_PSO_"+str(nlcs)+'.txt','a')

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

chain = mcmc.LikelihoodModule(lcs, fit_vector, spline, rt_file, nlcs,
                              kntstp, max_core = max_process, shotnoise = shotnoise,
                              recompute_spline = True, n_curve_stat= n_curve_stat, para = False)

if mpi is True:
    pso = MpiParticleSwarmOptimizer(chain, lowerLimit, upperLimit, n_particles, threads=1)
else:
    pso = ParticleSwarmOptimizer(chain, lowerLimit, upperLimit, n_particles, threads=1)

X2_list = []
vel_list = []
pos_list = []
num_iter = 0
for swarm in pso.sample(n_iterations):
    X2_list.append(pso.gbest.fitness * 2)
    vel_list.append(pso.gbest.velocity)
    pos_list.append(pso.gbest.position)
    num_iter += 1
    if pso.isMaster():
        if num_iter % 10 == 0:
            print(num_iter)

print("--- %s seconds ---" % (time.time() - start_time))




