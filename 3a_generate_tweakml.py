
import pycs
import dill
from module import tweakml_PS_from_data as twk
from module.optimisation import grid_search_PS as grid
import numpy as np
import matplotlib.pyplot as plt
import os

execfile("config.py")
tweakml_plot_dir = figure_directory + 'tweakml_plots/'
if not os.path.isdir(tweakml_plot_dir):
	os.mkdir(tweakml_plot_dir)

for i,kn in enumerate(knotstep) :
    for j, knml in enumerate(mlknotsteps):
        lcs, spline = pycs.gen.util.readpickle(lens_directory + '%s/initopt_%s_ks%i_ksml%i.pkl' % (combkw[i, j], dataname, kn, knml))
        for k,l in enumerate(lcs):
            if l.ml == None:
                pycs.gen.splml.addtolc(l, n=2)

        if tweakml_type == 'colored_noise':
            if find_tweak_ml_param == True :
                #todo : the PSO optimisation here
                pass

            else :
                print "Colored noise : I will add the beta and sigma that you gave in input."
                tweak_ml_list = []
                for k in range(len(lcsl)):
                    def tweakml_colored(lcs):
                        return pycs.sim.twk.tweakml(lcs, beta=colored_noise_param[k][0],
                                                       sigma=colored_noise_param[k][1], fmin=1.0 / 50.0, fmax=0.2, psplot=False)


                    tweak_ml_list.append(tweakml_colored)

                f = open(lens_directory + combkw[i,j] + '/tweakml_' + tweakml_name + '.dill','w')
                dill.dump(tweak_ml_list,f)


        elif tweakml_type == 'PS_from_residuals':
            if find_tweak_ml_param == True :
                tweak_ml_list = []
                rls = pycs.gen.stat.subtract(lcs, spline)
                target = pycs.gen.stat.mapresistats(rls)
                pycs.sim.draw.saveresiduals(lcs, spline)

                for k in range(len(lcs)):
                    print "I will try to find the parameter for lightcurve :", lcs[k].object
                    print "Target is :", target[k]
                    B_vec = np.linspace(0.1, 2, 5)
                    success, B_best, [zruns, sigma], [zruns_std, sigma_std], chi2, min_ind = grid.grid_search_PS(lcs[k],spline,B_vec,
                                                                                                                         target[k],  max_core=max_core,
                                                                                                                         n_curve_stat=n_curve_stat, verbose = True,
                                                                                                                         shotnoise = None, knotstep = knml)

                    fig1 = plt.figure(1)
                    plt.errorbar(B_vec, zruns, yerr=zruns_std)
                    plt.hlines(target[k]['zruns'], B_vec[0], B_vec[-1], colors='r', linestyles='solid', label='target')
                    plt.xlabel('B in unit of Nymquist frequency)')
                    plt.legend()
                    plt.ylabel('zruns')
                    fig2 = plt.figure(2)
                    plt.errorbar(B_vec, sigma, yerr=sigma_std)
                    plt.hlines(target[k]['std'], B_vec[0], B_vec[-1], colors='r', linestyles='solid', label='target')
                    plt.xlabel('B in unit of Nymquist frequency)')
                    plt.legend()
                    plt.ylabel('sigma')
                    fig1.savefig(tweakml_plot_dir + tweakml_name + 'zruns_' + lcs[k].object + '.png')
                    fig2.savefig(tweakml_plot_dir + tweakml_name + 'std_' + lcs[k].object + '.png')

                    if display :
                        plt.show()
                    plt.close()

                    if success :
                        print "I succeeded finding a parameter falling in the 0.5 sigma from the original lightcurve."
                        def tweakml_PS(lcs):
                            return twk.tweakml_PS(lcs, spline, B_best, f_min=1 / 300.0, psplot=False, verbose=False,
                                                  interpolation='linear')


                        tweak_ml_list.append(tweakml_PS)

                    else :
                        print "I didn't find a parameter that falls in the 0.5 sigma from the original lightcurve."
                        print "I then choose the best one... but be carefull ! "
                        def tweakml_PS(lcs):
                            return twk.tweakml_PS(lcs, spline, B_best, f_min=1 / 300.0, psplot=False, verbose=False,
                                                  interpolation='linear')


                        tweak_ml_list.append(tweakml_PS)


                f = open(lens_directory + combkw[i, j] + '/tweakml_' + tweakml_name + '.dill', 'w')
                dill.dump(tweak_ml_list, f)

            else :
                print "Noise from Power Spectrum of the data : I use PS_param that you gave in input."
                tweak_ml_list = []
                for k in range(len(lcs)):
                    def tweakml_PS(lcs):
                        (_ , spline) = pycs.gen.util.readpickle(lens_directory + '%s/initopt_%s_ks%i_ksml%i.pkl' % (combkw[i,j], dataname, kn,knml))
                        return twk.tweakml_PS(lcs, spline, PS_param_B, f_min = 1/300.0,psplot=False, verbose = False, interpolation = 'linear')


                    tweak_ml_list.append(tweakml_PS)

                f = open(lens_directory + combkw[i, j]  + '/tweakml_' + tweakml_name + '.dill','w')
                dill.dump(tweak_ml_list,f )


