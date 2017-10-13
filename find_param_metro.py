#Find micro lensing Power spectrum using MCMC algo

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
niter = 100
nburn = 10
nlcs = 0 #numero de la courbe a traiter

open('./MCMC_test/rt_file' + object +"_"+ picklename + "_" + str(niter) +'.txt', 'w').close() # to clear the file
rt_file = open('./MCMC_test/rt_file' + object +"_"+ picklename + "_" + str(niter) +'.txt','a')

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

#Control the residuals :
rls = pycs.gen.stat.subtract(lcs, spline)
print 'Residuals from the fit : '
print pycs.gen.stat.mapresistats(rls)
fit_sigma = pycs.gen.stat.mapresistats(rls)[nlcs]["std"]
fit_zruns = pycs.gen.stat.mapresistats(rls)[nlcs]["zruns"]
fit_nruns = pycs.gen.stat.mapresistats(rls)[nlcs]["nruns"]
fit_vector = [fit_zruns,fit_sigma]
fit_error = [0.05,0.02]
pycs.sim.draw.saveresiduals(lcs, spline)

initial_position = [-1.5,0.1]
theta_walk, chi2_walk = mcmc.mcmc_metropolis(initial_position, lcs, fit_vector,fit_error, niter = niter, burntime = nburn, savefile = rt_file, nlcs=0)

print("--- %s seconds ---" % (time.time() - start_time))

pickle.dump(theta_walk, open("./MCMC_test/theta_walk_" + object +"_"+ picklename + "_" + str(niter) +".pkl", "wb" ))
pickle.dump(chi2_walk, open("./MCMC_test/chi2_walk_" + object +"_"+ picklename + "_" + str(niter) +".pkl", "wb" ))




#TODO : check how many curve you need to have a good statistic on zruns and sigma
#TODO : check if it changes something to use draw with one lightcurve or with all lightcurve


