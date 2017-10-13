import pickle
import corner
import matplotlib.pyplot as plt
import numpy as np
import pycs
import mcmc_function as fmcmc

theta = pickle.load(open("./MCMC_test/theta_walk_10000.pkl", "rb"))
chi2 = pickle.load(open("./MCMC_test/chi2_walk_10000.pkl", "rb"))
makeplot = True

theta = np.asarray(theta)
chi2 = np.asarray(chi2)

picklepath = "./LCJ0806b/save/"
picklename ="opt_spl_ml_80-350knt.pkl"
nlcs = 0 #numero de la courbe a traiter

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

#Control the residuals :
rls = pycs.gen.stat.subtract(lcs, spline)
pycs.sim.draw.saveresiduals(lcs, spline)

print 'Residuals from the fit : '
print pycs.gen.stat.mapresistats(rls)
fit_sigma = pycs.gen.stat.mapresistats(rls)[nlcs]["std"]
fit_zruns = pycs.gen.stat.mapresistats(rls)[nlcs]["zruns"]
fit_nruns = pycs.gen.stat.mapresistats(rls)[nlcs]["nruns"]
fit_vector = [fit_zruns,fit_sigma]

min_chi2 = np.min(chi2)
N_min = np.argmin(chi2)
min_theta = theta[N_min,:]

print "min Chi2 : ", min_chi2
print "min theta :", min_theta
fmcmc.make_mocks(min_theta,lcs[nlcs],spline,ncurve=100, verbose=True)
print "compared to sigma, nruns, zruns : "+ str(fit_sigma) + ', ' + str(fit_nruns) + ', ' + str(fit_zruns)

if makeplot :
    fig1 = corner.corner(theta, labels=["$beta$", "$\sigma$"])

    plt.figure(2)
    x = np.arange(len(chi2))
    plt.xlabel('N', fontdict={"fontsize" : 16})
    plt.ylabel('$\chi^2$', fontdict={"fontsize" : 16})
    plt.plot(x,chi2)
    plt.savefig('./MCMC_test/chi2-random.png')

    fig3, axe = plt.subplots(2,1,sharex=True)
    axe[0].plot(x,theta[:,0],'r')
    axe[1].plot(x,theta[:,1],'g')
    plt.xlabel('N', fontdict={"fontsize" : 16})
    axe[0].set_ylabel('beta', fontdict={"fontsize" : 16})
    axe[1].set_ylabel('$\sigma$', fontdict={"fontsize" : 16})
    plt.savefig('./MCMC_test/beta-sigma-random.png')
    plt.show()

    fig1.savefig("./MCMC_test/plot.png")