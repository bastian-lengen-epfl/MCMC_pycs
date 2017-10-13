import pycs
import pycs.regdiff
import pickle
import time
import mcmc_function as mcmc

start_time = time.time()

object = "LCJ0806b"
kntstp = 80
ml_kntstep =350
picklepath = "./"+object+"/save/"
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
niter = 10
nburn = 0
nlcs = 0 #numero de la courbe a traiter

open('./MCMC_test/rt_file' + object +"_"+ picklename + "_" + str(niter) +'.txt', 'w').close() # to clear the file
rt_file = open('./MCMC_test/rt_file' + object +"_"+ picklename + "_" + str(niter) +'.txt','a')

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)
pycs.sim.draw.saveresiduals(lcs, spline)
theta = [-1.5,0.1]

res,err = mcmc.make_mocks(theta, lcs, spline, recompute_spline=True, kntstep=kntstp, verbose=True, display=True)

