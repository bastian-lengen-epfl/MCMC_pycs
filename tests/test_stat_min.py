import pycs.regdiff
import time
from module.optimisation import mcmc_function as mcmc
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

start_time = time.time()

object = "LCJ0806b"
kntstp = 80
ml_kntstep =350
picklepath = "./"+object+"/save/"
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
niter = 100
nburn = 10
nlcs = 0 #numero de la courbe a traiter

open('./MCMC_test/rt_file_' + object +"_"+ picklename + "_" + str(niter) +'.txt', 'w').close() # to clear the file
rt_file = open('./MCMC_test/rt_file_' + object +"_"+ picklename + "_" + str(niter) +'.txt','a')

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)
pycs.sim.draw.saveresiduals(lcs, spline)
theta = [-1.5,0.1]
x = np.linspace(2,200,30)
res_save = []
err_save = []
print x

for i,j in enumerate(x):
    x[i] = int(j)
    print i
    res,err = mcmc.make_mocks(theta, lcs, spline, recompute_spline=True, kntstep=kntstp, verbose=True, ncurve = int(j))
    res_save.append(res)
    err_save.append(err)

res_save = np.asarray(res_save)
err_save = np.asarray(err_save)

fig1, axe = plt.subplots(2, 2, sharex=True)
axe[0,0].plot(x, res_save[:,0], 'r')
axe[1,0].plot(x, res_save[:,1], 'g')
axe[0,1].plot(x, err_save[:,0], 'r')
axe[1,1].plot(x, err_save[:,1], 'g')
axe[1,0].set_xlabel('N', fontdict={"fontsize": 16})
axe[1,1].set_xlabel('N', fontdict={"fontsize": 16})
axe[0,0].set_ylabel('Mean zruns', fontdict={"fontsize": 16})
axe[1,0].set_ylabel('Mean sigma', fontdict={"fontsize": 16})
axe[0,1].set_ylabel('Std zruns', fontdict={"fontsize": 16})
axe[1,1].set_ylabel('Std sigma', fontdict={"fontsize": 16})
plt.show()



