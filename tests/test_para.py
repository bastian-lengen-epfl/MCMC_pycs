import pycs.regdiff
from module.optimisation import mcmc_function as mcmc
import time


object = "LCJ0806b"
kntstp = 80
ml_kntstep =350
picklepath = "./"+object+"/save/"
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
niter = 100
nburn = 10
nlcs = 0 #numero de la courbe a traiter

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)
pycs.sim.draw.saveresiduals(lcs, spline)
theta = [-1.5,0.1]

t = time.time()
res,err = mcmc.make_mocks_para(theta, lcs, spline, recompute_spline=True, knotstep=kntstp, verbose=True, display=False, ncurve=30)

print res, err
print("--- %s seconds ---" % (time.time() - t))


