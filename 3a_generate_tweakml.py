
import pycs
from module import tweakml_PS_from_data as twk
from module.optimisation import grid_search_PS as grid
from module.optimisation import mcmc_function as mcmc
import numpy as np
import matplotlib.pyplot as plt
import os
import module.util_func as util
import sys

execfile("config.py")
tweakml_plot_dir = figure_directory + 'tweakml_plots/'

if not os.path.isdir(tweakml_plot_dir):
	os.mkdir(tweakml_plot_dir)

for i,kn in enumerate(knotstep):
    for j, knml in enumerate(mlknotsteps):
        f = open(lens_directory + combkw[i, j] + '/tweakml_' + tweakml_name + '.py', 'w+')
        f.write('import pycs \n')
        f.write('from module import tweakml_PS_from_data as twk \n')
        lcs, spline = pycs.gen.util.readpickle(lens_directory + '%s/initopt_%s_ks%i_ksml%i.pkl' % (combkw[i, j], dataname, kn, knml))
        optim_directory = lens_directory + '%s/twk_optim_%s_%s'%(combkw[i, j], optimiser, tweakml_name)
        if not os.path.isdir(optim_directory):
            os.mkdir(optim_directory)
        for k,l in enumerate(lcs):
            if l.ml == None:
                pycs.gen.splml.addtolc(l, n=2)

        #Starting to write tweakml function depending on tweak_ml_type :
        if tweakml_type == 'colored_noise':
            if find_tweak_ml_param == True :
                if optimiser == 'PSO':
                    for k,l in enumerate(lcs):
                        fit_vector = mcmc.get_fit_vector(l,spline)
                        pycs.sim.draw.saveresiduals(l, spline)
                        PSO_opt = mcmc.PSO_Optimiser(l, fit_vector, spline, savefile=None,
                                                     knotstep=kn, max_core=max_core, shotnoise=shotnoise_type,
                                                     recompute_spline=True, n_curve_stat=n_curve_stat,
                                                     theta_init=None, savedirectory=optim_directory,
                                                     n_particles=n_particles, n_iter=n_iter,
                                                        mpi=mpi, tweak_ml_type = tweak_ml_type, tweakml_name= tweakml_name)

                        chain_list = PSO_opt.optimise()
                        best_chi2, best_param = PSO_opt.get_best_param()
                        PSO_opt.analyse_plot_results()
                        PSO_opt.dump_results()
                        if k == 0 :
                            PSO_opt.reset_report()
                            PSO_opt.report()
                        else :
                            PSO_opt.report()

                        #write the python file containing the function :
                        def tweakml_colored_NUMBER(lcs):
                            return pycs.sim.twk.tweakml(lcs, beta=BETA, sigma=SIGMA, fmin=1.0 / 50.0, fmax=0.2,
                                                        psplot=False)


                        util.write_func_append(tweakml_colored_NUMBER, f,
                                               BETA=str(best_param[0]),
                                               SIGMA=str(best_param[1]), NUMBER=str(k + 1))

                elif optimiser == 'MCMC' :
                    for k, l in enumerate(lcs):
                        fit_vector = mcmc.get_fit_vector(l, spline)
                        pycs.sim.draw.saveresiduals(l, spline)
                        sigma_step = [0.1, 0.005]
                        n_burn = 0.1 * n_iter
                        rdm_walk = 'log' #the step is made in logscal for sigma, this is made ot better sample the small sigmas
                        stopping_condition = True
                        initial_position = [-2.0, 0.1]
                        MH_opt = mcmc.Metropolis_Hasting_Optimiser(l, fit_vector, spline, gaussian_step=sigma_step,
                                                                   niter=n_iter, burntime=n_burn, savedirectory=optim_directory,
                                                                   recompute_spline=True,
                                                                   knotstep=kn, rdm_walk=rdm_walk,
                                                                   n_curve_stat=n_curve_stat,
                                                                   max_core=max_core,
                                                                   stopping_condition=stopping_condition,
                                                                   shotnoise=shotnoise,
                                                                   tweak_ml_type=tweak_ml_type, tweakml_name= tweakml_name,
                                                                   theta_init=initial_position)
                        theta_walk, chi2_walk, sz_walk, errorsz_walk = MH_opt.optimise()
                        best_chi2, best_param = MCMC_opt.get_best_param()
                        MH_opt.analyse_plot_results()
                        MH_opt.dump_results()
                        if k == 0 :
                            MH_opt.reset_report()
                            MH_opt.report()
                        else :
                            MH_opt.report()
                        #write the python file containing the function :
                        def tweakml_colored_NUMBER(lcs):
                            return pycs.sim.twk.tweakml(lcs, beta=BETA, sigma=SIGMA, fmin=1.0 / 50.0, fmax=0.2,
                                                        psplot=False)


                        util.write_func_append(tweakml_colored_NUMBER, f,
                                               BETA=str(best_param[0]),
                                               SIGMA=str(best_param[1]), NUMBER=str(k + 1))



                else :
                    print "I don't this optimiser !"
                    sys.exit()


            else :
                print "Colored noise : I will add the beta and sigma that you gave in input."
                for k in range(len(lcs)):
                    def tweakml_colored_NUMBER(lcs):
                        return pycs.sim.twk.tweakml(lcs, beta=BETA,sigma=SIGMA, fmin=1.0 / 50.0, fmax=0.2, psplot=False)


                    util.write_func_append(tweakml_colored_NUMBER, f,
                                           BETA= str(colored_noise_param[k][0]), SIGMA = str(colored_noise_param[k][1]), NUMBER = str(k+1))

            list_string = 'tweakml_list = ['
            for k in range(len(lcs)):
                list_string += 'tweakml_colored_'+str(k+1) +','
            list_string +=']'
            f.write('\n')
            f.write(list_string)



        elif tweakml_type == 'PS_from_residuals':
            if shotnoise_type != None :
                print 'If you use PS_from_residuals, the shotnoise should be set to None. I will do it for you !'
                shotnoise_type = None

            if find_tweak_ml_param == True :
                tweak_ml_list = []
                pycs.sim.draw.saveresiduals(lcs, spline)

                for k, l in enumerate(lcs):
                    fit_vector = mcmc.get_fit_vector(l, spline)
                    print "I will try to find the parameter for lightcurve :", l.object

                    grid_opt = mcmc.Grid_Optimiser(l, fit_vector, spline, knotstep=kn,
                     savedirectory=optim_directory, recompute_spline=True, max_core = max_core,
                    n_curve_stat = n_curve_stat, shotnoise = shotnoise_type, tweakml_type = tweakml_type, tweakml_name= tweakml_name,
                 display = display, verbose = False, grid = grid)

                    chain = grid_opt.optimise()
                    grid_opt.analyse_plot_results()
                    chi2, B_best = grid_opt.get_best_param()
                    if k == 0:
                        grid_opt.reset_report()
                        grid_opt.report()
                    else:
                        grid_opt.report()

                    if grid_opt.success :
                        print "I succeeded finding a parameter falling in the 0.5 sigma from the original lightcurve."

                    else :
                        print "I didn't find a parameter that falls in the 0.5 sigma from the original lightcurve."
                        print "I then choose the best one... but be carefull ! "

                    def tweakml_PS_NUMBER(lcs):
                        return twk.tweakml_PS(lcs, spline, B_PARAM, f_min=1 / 300.0, psplot=False, verbose=False,
                                                  interpolation='linear')


                    util.write_func_append(tweakml_PS_NUMBER, f,
                                               B_PARAM=str(B_best), NUMBER=str(k + 1))


            else :
                print "Noise from Power Spectrum of the data : I use PS_param that you gave in input."
                for k in range(len(lcs)):
                    def tweakml_PS_NUMBER(lcs):
                        return twk.tweakml_PS(lcs, spline, B_PARAM, f_min=1 / 300.0, psplot=False, verbose=False,
                                       interpolation='linear')


                    util.write_func_append(tweakml_PS_NUMBER, f,
                                           B_PARAM= str(PS_param_B[k]), NUMBER = str(k+1))

            list_string = 'tweakml_list = ['
            for k in range(len(lcs)):
                list_string += 'tweakml_PS_'+str(k+1) +','
            list_string +=']'
            f.write('\n')
            f.write(list_string)

        else :
            print "I don't know your tweak_ml_type, please use colored_noise or PS_form_residuals."
