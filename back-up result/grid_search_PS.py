import numpy as np
from module.optimisation import mcmc_function as mcmc

def grid_search_PS(lc,spline,B_vec, target,  max_core=None, n_curve_stat=32, verbose = False, shotnoise = None, knotstep = None):


    sigma = []
    zruns = []
    sigma_std = []
    zruns_std = []
    chi2 = []
    zruns_target = target['zruns']
    sigma_target = target['std']

    for i,B in enumerate(B_vec):
        [[zruns_c,sigma_c],[zruns_std_c,sigma_std_c]] = mcmc.make_mocks_para(lc, spline, theta=[B], tweakml_type='PS_from_residuals', knotstep=knotstep,
                            recompute_spline=True,max_core=max_core,
                            display=False, n_curve_stat=n_curve_stat, shotnoise= shotnoise, verbose=verbose)
        # [[zruns_c,sigma_c],[zruns_std_c,sigma_std_c]] = mcmc.make_mocks(lc, spline, theta=[B], tweakml_type='PS_from_residuals', knotstep=knotstep,
        #                     recompute_spline=True,
        #                     display=False, n_curve_stat=n_curve_stat, shotnoise= shotnoise, verbose=verbose)

        chi2.append((zruns_c-zruns_target)**2 / zruns_std_c**2 + (sigma_c-sigma_target)**2 / sigma_std_c**2)

        sigma.append(sigma_c)
        zruns.append(zruns_c)
        sigma_std.append(sigma_std_c)
        zruns_std.append(zruns_std_c)


    min_ind = np.argmin(chi2)
    error_sig = np.abs(sigma[min_ind] - sigma_target) / sigma_std[min_ind]
    error_zruns= np.abs(zruns[min_ind] - zruns_target) / zruns_std[min_ind]

    if verbose :
        print "target :", target
        print "Best parameter from grid search :", B_vec[min_ind]
        print "Associated chi2 : ", chi2[min_ind]
        print "Zruns : %2.6f +/- %2.6f (%2.4f sigma from target)" %(zruns[min_ind], zruns_std[min_ind], error_zruns)
        print "Sigma : %2.6f +/- %2.6f (%2.4f sigma from target)" %(sigma[min_ind], sigma_std[min_ind], error_sig)

    if error_sig < 0.5 and error_zruns <0.5 :
        success =True
    else :
        success = False

    return success, B_vec[min_ind], [zruns,sigma], [zruns_std,sigma_std], chi2,min_ind


