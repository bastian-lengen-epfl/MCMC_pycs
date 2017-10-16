#Find micro lensing Power spectrum using MCMC algo

import pycs
import pycs.regdiff
import numpy as np
import pickle
import corner

import scipy.optimize as op

def ln_likelihood(theta, lcs, spline, fit_vector):
    return -0.5*compute_chi2(theta, lcs, spline, fit_vector)

def lnprior(theta):
    if -1.0 < theta[0] < -2.0 and 0 < theta[1] < 1.0:
        return 0.0
    return -np.inf

def lnprob(theta, lcs, spline, fit_vector):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(theta, lcs, spline, fit_vector)


def make_mocks(theta, lcs, spline, ncurve = 20):
    mocklcs = []
    mockrls = []
    stat = []
    zruns = []
    sigmas = []
    nruns = []

    for i in range(ncurve) :
        mocklcs.append(pycs.sim.draw.draw([lcs], spline, tweakml= lambda x : pycs.sim.twk.tweakml(x, beta=theta[0],
                                            sigma=theta[1], fmin=1/300.0, fmax=None, psplot=False),
                                          shotnoise="magerrs", keeptweakedml=False))
        mockrls.append(pycs.gen.stat.subtract(mocklcs[i], spline))

        stat.append(pycs.gen.stat.mapresistats(mockrls[i]))
        for j in range(len(stat[i])):
            zruns.append(stat[i][j]['zruns'])
            sigmas.append(stat[i][j]['std'])
            nruns.append(stat[i][j]['nruns'])


    print 'Mean zruns (simu): ', np.mean(zruns), '+/-', np.std(zruns)
    print 'Mean sigmas (simu): ', np.mean(sigmas), '+/-', np.std(sigmas)
    print 'Mean nruns (simu): ', np.mean(nruns), '+/-', np.std(nruns)

    return [np.mean(zruns),np.mean(sigmas)], [np.std(zruns),np.std(sigmas)]

def compute_chi2(theta, lcs, spline, fit_vector):

    chi2 =0.0
    out, error = make_mocks(theta, lcs, spline)
    for i in range(len(out)):
        chi2 += (fit_vector[i]-out[i])**2 / error[i]**2
    return chi2


# picklepath = "./LCJ0806/save/"
# picklename ="optspl_oneml_80bknt.pkl"
picklepath = "./LCJ0806b/save/"
picklename ="opt_spl_ml_80-350knt.pkl"
ncurve = 100
nlcs = 0 #numero de la courbe a traiter

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

initial_position = [-1.8,0.0]
bounds = ((-2.0, -1.0), (0, 0.5))

nll = lambda *args: -ln_likelihood(*args)
result = op.minimize(nll, initial_position, args=(lcs[nlcs],spline,fit_vector), method="Powell", options={"disp" : True}, bounds= bounds)
pickle.dump(result, open("./MCMC_test/result_minimization.pkl", "wb" ))
print result["x"]
print result["success"]

print "RESULT : "
chi2 = compute_chi2(result["x"], lcs[nlcs],spline , fit_vector)
print "compared to sigma, nruns, zruns : "+ str(fit_sigma) + ', ' + str(fit_nruns) + ', ' + str(fit_zruns)
print "chi2 :", chi2

exit()


#MCMC priors (noise parameters) :
# sig = pymc.Uniform('sig', 0.0, 0.6, value=0.1)
# beta = pymc.Uniform('beta', -2.0, -1.0, value= -1.5)
#
# M = pymc.MCMC(set([sig,beta]))
# M.sample(10000)
# print M.trace('sig')[:]

# ndim, nwalkers = 2, 1000
# burntime =100
# pos = [initial_position + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
# print pos
#
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(lcs[nlcs],spline,fit_vector))
# sampler.run_mcmc(pos, 1000)
#
# samples = sampler.chain[:, burntime:, :].reshape((-1, ndim))
#
# fig = corner.corner(samples, labels=["$beta$", "$\sigma$"],
#                       truths=fit_vector)
# fig.savefig("plot.png")



