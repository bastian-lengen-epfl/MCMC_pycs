import pickle
import corner
import matplotlib.pyplot as plt
import numpy as np
import pycs
import mcmc_function as fmcmc


makeplot = True
source ="pickle"
object = "LCJ0806b"
picklepath = "./LCJ0806b/save/"
picklename ="opt_spl_ml_80-350knt.pkl"
burntime = 10
knotstep = 80
niter = 15000
nlcs = 0 #numero de la courbe a traiter

if source == "pickle":
    theta = pickle.load(open("./MCMC_test/theta_walk_"+ object + "_" + picklename + "_" + str(niter) + ".pkl", "rb"))
    chi2 = pickle.load(open("./MCMC_test/chi2_walk_"+ object + "_" + picklename + "_" + str(niter) + ".pkl", "rb"))

if source == "rt_file":
    rt_filename = './MCMC_test/rt_file' + object + "_" + picklename + "_" + str(niter) + '.txt'
    vec = np.loadtxt(rt_filename, delimiter=',')
    vec = np.asarray(vec)
    theta = vec[burntime:,0:2]
    chi2 = vec[burntime:,2]

theta = np.asarray(theta)
chi2 = np.asarray(chi2)

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

#Control the residuals :
rls = pycs.gen.stat.subtract(lcs, spline)
pycs.sim.draw.saveresiduals(lcs, spline)

print 'Residuals from the fit : '
print pycs.gen.stat.mapresistats(rls)[nlcs]
fit_sigma = pycs.gen.stat.mapresistats(rls)[nlcs]["std"]
fit_zruns = pycs.gen.stat.mapresistats(rls)[nlcs]["zruns"]
fit_nruns = pycs.gen.stat.mapresistats(rls)[nlcs]["nruns"]
fit_vector = [fit_zruns,fit_sigma]

min_chi2 = np.min(chi2)
N_min = np.argmin(chi2)
min_theta = theta[N_min,:]

print "min Chi2 : ", min_chi2
print "min theta :", min_theta
mean_mini,sigma_mini = fmcmc.make_mocks_para(min_theta,lcs,spline,ncurve=100, recompute_spline= True, knotstep=knotstep, nlcs=nlcs, verbose=True)
print "compared to sigma, nruns, zruns : "+ str(fit_sigma) + ', ' + str(fit_nruns) + ', ' + str(fit_zruns)
print "For minimum Chi2, we are standing at " + str(np.abs(mean_mini[0]-fit_zruns)/sigma_mini[0]) + " sigma [zruns]" 
print "For minimum Chi2, we are standing at " + str(np.abs(mean_mini[1]-fit_sigma)/sigma_mini[1]) + " sigma [sigma]"

if makeplot :
    fig1 = corner.corner(theta, labels=["$beta$", "$\sigma$"])

    plt.figure(2)
    x = np.arange(len(chi2))
    plt.xlabel('N', fontdict={"fontsize" : 16})
    plt.ylabel('$\chi^2$', fontdict={"fontsize" : 16})
    plt.plot(x,chi2)
    plt.savefig('./MCMC_test/chi2-random'+ picklename + "_" + str(niter) + '.png')

    fig3, axe = plt.subplots(2,1,sharex=True)
    axe[0].plot(x,theta[:,0],'r')
    axe[1].plot(x,theta[:,1],'g')
    plt.xlabel('N', fontdict={"fontsize" : 16})
    axe[0].set_ylabel('beta', fontdict={"fontsize" : 16})
    axe[1].set_ylabel('$\sigma$', fontdict={"fontsize" : 16})
    plt.savefig('./MCMC_test/beta-sigma-random'+ picklename + "_" + str(niter) + '.png')
    plt.show()

    fig1.savefig("./MCMC_test/cornerplot_" + picklename + "_" + str(niter) + '.png')
