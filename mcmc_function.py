import numpy as np
import pycs
import pycs.regdiff
import copy

def mcmc_metropolis(theta, lcs, fit_vector, fit_error, spline, knstep = None, niter=1000, burntime=100, savefile = None, nlcs = 0, recompute_spline = False):
    theta_save = []
    chi2_save = []
    chi2_current = compute_chi2(theta, lcs, fit_vector, spline, knstep= knstep, nlcs = nlcs, recompute_spline= recompute_spline)

    for i in range(niter):
        print i
        theta_new = theta + fit_error*np.random.randn(2)
        if not prior(theta_new):
            continue

        print theta_new
        chi2_new = compute_chi2(theta_new, lcs, fit_vector, spline, knstep= knstep, nlcs = nlcs, recompute_spline = recompute_spline)
        ratio = np.exp((-chi2_new + chi2_current)/2.0 );

        if np.random.rand() < ratio :
            theta = copy.deepcopy(theta_new)
            chi2_current = copy.deepcopy(chi2_new)

        if i > burntime :
            theta_save.append(theta)
            chi2_save.append(chi2_current)

        if savefile != None :
            data = np.asarray([theta[0], theta[1], chi2_current])
            data =np.reshape(data, (1,3))
            np.savetxt(savefile, data, delimiter=',')



    return theta_save, chi2_save

def prior(theta):
    if -2.0<theta[0]<-1.0 and 0<theta[1]< 0.5:
        return True
    else:
        return False

def make_mocks(theta, lcs, spline, ncurve = 50, verbose = False, kntstep = None, recompute_spline = True, nlcs = 0, display = False):
    mocklcs = []
    mockrls = []
    stat = []
    zruns = []
    sigmas = []
    nruns = []

    for i in range(ncurve) :

        mocklcs.append(pycs.sim.draw.draw([lcs[nlcs]], spline, tweakml= lambda x : pycs.sim.twk.tweakml(x, beta=theta[0],
                                            sigma=theta[1], fmin=1/300.0, fmax=None, psplot=False),
                                          shotnoise="magerrs", keeptweakedml=False))

        if recompute_spline :
            if kntstep == None :
                print "Error : you must give a knotstep to recompute the spline"
            spline_on_mock = pycs.spl.topopt.opt_fine(mocklcs[i], nit=5, knotstep=kntstep, verbose=False)
            mockrls.append(pycs.gen.stat.subtract(mocklcs[i], spline_on_mock))
        else:
            mockrls.append(pycs.gen.stat.subtract(mocklcs[i], spline))

        if recompute_spline and display :
            pycs.gen.lc.display([lcs[nlcs]], [spline_on_mock], showdelays=True)
            pycs.gen.stat.plotresiduals([mockrls[i]])

        stat.append(pycs.gen.stat.mapresistats(mockrls[i]))
        zruns.append(stat[i][nlcs]['zruns'])
        sigmas.append(stat[i][nlcs]['std'])
        nruns.append(stat[i][nlcs]['nruns'])

    if verbose:
        print 'Mean zruns (simu): ', np.mean(zruns), '+/-', np.std(zruns)
        print 'Mean sigmas (simu): ', np.mean(sigmas), '+/-', np.std(sigmas)
        print 'Mean nruns (simu): ', np.mean(nruns), '+/-', np.std(nruns)

    return [np.mean(zruns),np.mean(sigmas)], [np.std(zruns),np.std(sigmas)]

def compute_chi2(theta, lcs, fit_vector, spline, nlcs = 0, knstep = 40, recompute_spline = False ):

    chi2 =0.0
    out, error = make_mocks(theta, lcs, spline,  nlcs=nlcs, recompute_spline=recompute_spline, kntstep=knstep)

    for i in range(len(out)):
        chi2 += (fit_vector[i]-out[i])**2 / error[i]**2
    return chi2