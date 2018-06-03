import pycs.regdiff
import numpy as np
from module.optimisation import mcmc_function as mcmc

object = "LCJ0806b"
kntstp = 80
ml_kntstep =350
picklepath = "./"+object+"/save/"
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
niter = 44
nburn = 0
nlcs = 0 #numero de la courbe a traiter
rt_filename = './MCMC_test/rt_file' + object +"_"+ picklename + "_100.txt"

rt_file = open(rt_filename,'a')
vec = np.loadtxt(rt_filename, delimiter=',')
initial = vec[-1][0:2]
fit_error = [0.05,0.02]


(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)
#Control the residuals :
rls = pycs.gen.stat.subtract(lcs, spline)
print 'Residuals from the fit : '
print pycs.gen.stat.mapresistats(rls)
fit_sigma = pycs.gen.stat.mapresistats(rls)[nlcs]["std"]
fit_zruns = pycs.gen.stat.mapresistats(rls)[nlcs]["zruns"]
fit_nruns = pycs.gen.stat.mapresistats(rls)[nlcs]["nruns"]
fit_vector = [fit_zruns,fit_sigma]
pycs.sim.draw.saveresiduals(lcs, spline)

theta_walk, chi2_walk = mcmc.mcmc_metropolis(initial, lcs, fit_vector,fit_error,spline,
                                 niter = niter, burntime = nburn, savefile = rt_file, nlcs=0, recompute_spline= True, para= True, knotstep = kntstp)
