import numpy as np
import pycs
import pycs.regdiff
import copy
import time
import multiprocessing
from module import tweakml_PS_from_data as twk


def mcmc_metropolis(theta, lc, fit_vector, spline, gaussian_step=[0.05, 0.02], knotstep=None, niter=1000,
                    burntime=100, savefile=None, recompute_spline=False, para=True, rdm_walk='gaussian', max_core = 16,
                    n_curve_stat = 32, stopping_condition = True, shotnoise = "magerrs"):
    theta_save = []
    chi2_save = []
    sz_save = []
    errorsz_save = []
    global hundred_last
    hundred_last = 100
    chi2_current, sz_current, errorsz_current = compute_chi2(theta, lc, fit_vector, spline, knotstep=knotstep,
                                recompute_spline=recompute_spline, max_core = max_core, n_curve_stat = n_curve_stat, shotnoise = shotnoise)
    t = time.time()

    for i in range(niter):
        t_now = time.time() - t
        print "time : ", t_now

        if rdm_walk == 'gaussian':
            theta_new = make_random_step_gaussian(theta, gaussian_step)
        elif rdm_walk == 'exp':
            theta_new = make_random_step_exp(theta, gaussian_step)
        elif rdm_walk == 'log':
            theta_new = make_random_step_log(theta, gaussian_step)

        if not prior(theta_new):
            continue

        chi2_new, sz_new, errorsz_new = compute_chi2(theta_new, lc, fit_vector, spline, knotstep=knotstep,
                                recompute_spline=recompute_spline, para=para, max_core = max_core,n_curve_stat = n_curve_stat, shotnoise = shotnoise)
        ratio = np.exp((-chi2_new + chi2_current) / 2.0);
        print "Iteration, Theta, Chi2, sz, errorsz :", i, theta_new, chi2_new, sz_new, errorsz_new

        if np.random.rand() < ratio:
            theta = copy.deepcopy(theta_new)
            chi2_current = copy.deepcopy(chi2_new)
            sz_current = copy.deepcopy(sz_new)
            errorsz_current = copy.deepcopy(errorsz_new)


            theta_save.append(theta)
            chi2_save.append(chi2_current)
            sz_save.append(sz_current)
            errorsz_save.append(errorsz_current)

            if savefile != None:
                data = np.asarray([theta[0], theta[1], chi2_current, sz_current[0],sz_current[1], errorsz_current[0], errorsz_current[1]])
                data = np.reshape(data, (1, 7))
                np.savetxt(savefile, data, delimiter=',')

        if stopping_condition == True:
            if check_if_stop(fit_vector, sz_current, errorsz_current):
                break


    return theta_save, chi2_save, sz_save, errorsz_save


def prior(theta):
    if -8.0 < theta[0] < -1.0 and 0 < theta[1] < 0.5:
        return True
    else:
        return False


def make_random_step_gaussian(theta, sigma_step):
    return theta + sigma_step * np.random.randn(2)

def make_random_step_log(theta, sigma_step):
    s = theta[1]
    s = np.log10(s) + sigma_step[1] * np.random.randn()
    s = 10.0**s
    return [theta[0] + sigma_step[0] * np.random.randn(), s]


def make_random_step_exp(theta, sigma_step):
    sign = np.random.random()
    print sign
    if sign > 0.5:
        print "step proposed : ", np.asarray(theta) - [theta[0] + sigma_step[0] * np.random.randn(),theta[1] + np.random.exponential(scale=sigma_step[1])]
        return [theta[0] + sigma_step[0] * np.random.randn(), theta[1] + np.random.exponential(scale=sigma_step[1])]
    else:
        print "step proposed : ",np.asarray(theta) - [theta[0] + sigma_step[0] * np.random.randn(), theta[1] - np.random.exponential(scale=sigma_step[1])]
        return [theta[0] + sigma_step[0] * np.random.randn(), theta[1] - np.random.exponential(scale=sigma_step[1])]


def make_mocks(lc, spline, theta = [-2.0,0.2], tweakml_type='colored_noise', n_curve_stat=32, verbose=False, knotstep=None, recompute_spline=True,
               display=False, shotnoise = "magerrs"):
    
    #by default this function is using colored noise, but you can change tweak_ml type to make it work for PS_from_residuals. In this case, theta has length 1. 
    mocklc = []
    mockrls = []
    stat = []
    zruns = []
    sigmas = []
    nruns = []

    for i in range(n_curve_stat):
        
        if tweakml_type == 'colored_noise' :
            mocklc.append(pycs.sim.draw.draw([lc], spline, tweakml=lambda x: pycs.sim.twk.tweakml(x, beta=theta[0],
                                                                                                      sigma=theta[1],
                                                                                                      fmin=1 / 300.0,
                                                                                                      fmax=None,
                                                                                                      psplot=False),
                                          shotnoise=shotnoise, keeptweakedml=False))

        elif tweakml_type == 'PS_from_residuals':
            mocklc.append(pycs.sim.draw.draw([lc], spline, tweakml=twk.tweakml_PS([lc], spline, theta[0], f_min = 1/300.0,
                                                                              psplot=False, save_figure_folder = None,  verbose = False,
                                                                              interpolation = 'linear')
                                             , shotnoise=shotnoise, keeptweakedml=False))
            

        if recompute_spline:
            if knotstep == None:
                print "Error : you must give a knotstep to recompute the spline"
            spline_on_mock = pycs.spl.topopt.opt_fine(mocklc[i], nit=5, knotstep=knotstep, verbose=False)
            mockrls.append(pycs.gen.stat.subtract(mocklc[i], spline_on_mock))
        else:
            mockrls.append(pycs.gen.stat.subtract(mocklc[i], spline))

        if recompute_spline and display:
            pycs.gen.lc.display([lc], [spline_on_mock], showdelays=True)
            pycs.gen.stat.plotresiduals([mockrls[i]])

        stat.append(pycs.gen.stat.mapresistats(mockrls[i]))
        zruns.append(stat[i][0]['zruns'])
        sigmas.append(stat[i][0]['std'])
        nruns.append(stat[i][0]['nruns'])

    if verbose:
        print 'Mean zruns (simu): ', np.mean(zruns), '+/-', np.std(zruns)
        print 'Mean sigmas (simu): ', np.mean(sigmas), '+/-', np.std(sigmas)
        print 'Mean nruns (simu): ', np.mean(nruns), '+/-', np.std(nruns)

    return [np.mean(zruns), np.mean(sigmas)], [np.std(zruns), np.std(sigmas)]


def make_mocks_para(lc, spline, theta = [-2.0,0.2], tweakml_type='colored_noise', knotstep=None, recompute_spline=True,
                    display=False, max_core = 16, n_curve_stat = 32, shotnoise = "magerrs", verbose = False):
    stat = []
    zruns = []
    sigmas = []
    nruns = []

    pool = multiprocessing.Pool(processes = max_core)
    job_kwarg = {'knotstep': knotstep, 'recompute_spline': recompute_spline, 'shotnoise' : shotnoise, 'theta' : theta, 'tweakml_type' : tweakml_type}
    job_args = [(lc, spline, job_kwarg) for j in range(n_curve_stat)]

    stat_out = pool.map(fct_para_aux, job_args)
    pool.close()
    pool.join()

    for i in range(len(stat_out)):
        zruns.append(stat_out[i][0]['zruns'])
        sigmas.append(stat_out[i][0]['std'])
        nruns.append(stat_out[i][0]['nruns'])

    if verbose:
        print 'Mean zruns (simu): ', np.mean(zruns), '+/-', np.std(zruns)
        print 'Mean sigmas (simu): ', np.mean(sigmas), '+/-', np.std(sigmas)
        print 'Mean nruns (simu): ', np.mean(nruns), '+/-', np.std(nruns)

    return [np.mean(zruns), np.mean(sigmas)], [np.std(zruns), np.std(sigmas)]


def compute_chi2(theta, lc, fit_vector, spline, knotstep=40, recompute_spline=False,
                 para=True, max_core = 16, n_curve_stat = 32, shotnoise = "magerrs",tweakml_type = 'colored_noise'):
    chi2 = 0.0
    if n_curve_stat ==1 :
        print "Warning : I cannot compute statistics with one single curves !!"

    if para:
        out, error = make_mocks_para(lc, spline, theta=theta, tweakml_type=tweakml_type, recompute_spline=recompute_spline,
                                     knotstep=knotstep, max_core = max_core, n_curve_stat= n_curve_stat, shotnoise=shotnoise)
    else:
        out, error = make_mocks(lc, spline, theta=theta, tweakml_type=tweakml_type, recompute_spline=recompute_spline,
                                knotstep=knotstep, n_curve_stat = n_curve_stat, shotnoise=shotnoise)

    # for i in range(len(out)):
    #     chi2 += (fit_vector[i] - out[i]) ** 2 / error[i] ** 2

    chi2 = (fit_vector[0] - out[0]) ** 2 / error[0] ** 2
    chi2 += (fit_vector[1] - out[1]) ** 2 / (2*error[1] ** 2)

    return chi2, out, error


def fct_para(lc, spline, theta='theta', knotstep=None, recompute_spline=True, shotnoise = "magerrs", tweakml_type = 'colored_noise'):
    if tweakml_type == 'colored_noise':
        mocklc = pycs.sim.draw.draw([lc], spline, tweakml=lambda x: pycs.sim.twk.tweakml(x, beta=theta[0],
                                                                                              sigma=theta[1],
                                                                                              fmin=1 / 300.0,
                                                                                              fmax=None,
                                                                                              psplot=False),
                                         shotnoise=shotnoise, keeptweakedml=False)

    elif tweakml_type == 'PS_from_residuals':
        mocklc = pycs.sim.draw.draw([lc], spline, tweakml=twk.tweakml_PS([lc], spline, theta[0], f_min = 1/300.0,
                                                                              psplot=False, save_figure_folder = None,  verbose = False,
                                                                              interpolation = 'linear')
                                    , shotnoise=shotnoise, keeptweakedml=False)

    if recompute_spline:
        if knotstep == None:
            print "Error : you must give a knotstep to recompute the spline"
        spline_on_mock = pycs.spl.topopt.opt_fine(mocklc, nit=5, knotstep=knotstep, verbose=False)
        mockrls = pycs.gen.stat.subtract(mocklc, spline_on_mock)
    else:
        mockrls = pycs.gen.stat.subtract(mocklc, spline)

    stat = pycs.gen.stat.mapresistats(mockrls)
    return stat


def fct_para_aux(args):
    kwargs = args[-1]
    args = args[0:-1]
    return fct_para(*args, **kwargs)

def check_if_stop(fitvector, sz, sz_error):
    global hundred_last
    if hundred_last != 100 :
        hundred_last -= 1# check if we already
        print "I have already matched the stopping condition, I will do %i more steps." %hundred_last

    elif np.abs(fitvector[0] - sz[0]) < 0.75*sz_error[0] and np.abs(fitvector[1] - sz[1]) < 0.75*sz_error[1]:
        hundred_last -= 1
        print "I'm matching the stopping condition at this iteration, I will do %i more steps."%hundred_last
    else :
        print "Stopping condition not reached."

    if hundred_last == 0 :
        return True
    else :
        return False

class LikelihoodModule :
    def __init__(self, lc, fit_vector, spline, rt_file, knotstep,max_core = 8, shotnoise = 'magerrs',
                 recompute_spline = True, n_curve_stat= 32, para = True):
        self.savefile = rt_file
        self.recompute_spline = recompute_spline
        self.para = para
        self.knotstep = knotstep
        self.n_curve_stat = n_curve_stat
        self.max_core = max_core,
        self.shotnoise = shotnoise
        self.lc = lc
        self.fit_vector = fit_vector
        self.spline = spline

    def __call__(self, theta):
        return self.likelihood(theta)

    def likelihood(self, theta):
        chi2, out ,error  = compute_chi2(theta, self.lc, self.fit_vector, self.spline, knotstep=self.knotstep,
                    recompute_spline=self.recompute_spline, max_core=self.max_core, n_curve_stat=self.n_curve_stat,
                     shotnoise=self.shotnoise, para= self.para)


        return [-0.5*chi2]


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real