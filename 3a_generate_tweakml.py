import os
import matplotlib as mpl
# mpl.use('Agg') #these scripts re for cluster so need to be sure
import matplotlib.pyplot as plt
import pycs
from module import tweakml_PS_from_data as twk
from module.optimisation import Optimiser as mcmc
import module.util_func as util
import sys
import numpy as np
import argparse as ap
import importlib

def run_PSO(lcs,spline,fit_vector,kn,ml,optim_directory, config_file, stream):
    config = importlib.import_module(config_file)
    pycs.sim.draw.saveresiduals(lcs, spline)
    PSO_opt = mcmc.PSO_Optimiser(lcs, fit_vector, spline, config.attachml, ml,
                                 knotstep=kn, max_core=config.max_core, shotnoise=config.shotnoise_type,
                                 recompute_spline=True, n_curve_stat=config.n_curve_stat,
                                 theta_init=None, savedirectory=optim_directory,
                                 n_particles=config.n_particles, n_iter=config.n_iter, verbose=False,
                                 mpi=config.mpi, tweakml_type=config.tweakml_type, tweakml_name=config.tweakml_name)

    chain_list = PSO_opt.optimise()
    best_chi2, best_param = PSO_opt.get_best_param()
    PSO_opt.analyse_plot_results()
    PSO_opt.dump_results()
    PSO_opt.reset_report()
    PSO_opt.report()

    # write the python file containing the function :
    if config.tweakml_type == "colored_noise":
        for k in range(len(lcs)):
            def tweakml_colored_NUMBER(lcs, spline):
                return pycs.sim.twk.tweakml(lcs, spline, beta=BETA, sigma=SIGMA, fmin=1.0 / 500.0, fmax=0.2,
                                            psplot=False)

            util.write_func_append(tweakml_colored_NUMBER, stream,
                                   BETA=str(best_param[k, 0]),
                                   SIGMA=str(best_param[k, 1]), NUMBER=str(k + 1))

    elif config.tweakml_type =="PS_from_residuals":
        A = PSO_opt.A_correction
        for k in range(len(lcs)):
            def tweakml_PS_NUMBER(lcs, spline):
                return twk.tweakml_PS(lcs, spline, B_PARAM, f_min=1 / 300.0, psplot=False, verbose=False,
                                      interpolation='linear', A_correction=A_PARAM)

            util.write_func_append(tweakml_PS_NUMBER, stream,
                                   B_PARAM=str(best_param[k, 0]), NUMBER=str(k + 1), A_PARAM=str(A[k]))


def run_MCMC(lcs,spline,fit_vector,kn,ml,optim_directory, config_file,stream):
    config = importlib.import_module(config_file)
    pycs.sim.draw.saveresiduals(lcs, spline)
    sigma_step = [0.1, 0.005]
    n_burn = 0.1 * config.n_iter
    rdm_walk = 'log'  # the step is made in logscal for sigma, this is made ot better sample the small sigmas
    stopping_condition = True
    initial_position = [[-1.9, 0.1], [-1.9, 0.1], [-1.9, 0.1], [-1.9, 0.1]]
    MH_opt = mcmc.Metropolis_Hasting_Optimiser(lcs, fit_vector, spline,config.attachml,ml, gaussian_step=sigma_step,
                                               n_iter=config.n_iter, burntime=n_burn, savedirectory=optim_directory,
                                               recompute_spline=True,
                                               knotstep=kn, rdm_walk=rdm_walk,
                                               n_curve_stat=config.n_curve_stat,
                                               max_core=config.max_core,
                                               stopping_condition=stopping_condition,
                                               shotnoise=config.shotnoise_type,
                                               tweakml_type=config.tweakml_type, tweakml_name=config.tweakml_name,
                                               theta_init=initial_position)
    theta_save, chi2_save, z_save, s_save, errorz_save, errors_save = MH_opt.optimise()
    MH_opt.dump_results()
    MH_opt.analyse_plot_results()
    MH_opt.reset_report()
    MH_opt.report()
    best_chi2, best_param = MH_opt.chi2_mini, np.asarray(MH_opt.best_param)

    # write the python file containing the function :
    for k in range(len(lcs)):
        def tweakml_colored_NUMBER(lcs, spline):
            return pycs.sim.twk.tweakml(lcs, spline, beta=BETA, sigma=SIGMA, fmin=1.0 / 500.0, fmax=0.2,
                                        psplot=False)

        util.write_func_append(tweakml_colored_NUMBER, stream,
                               BETA=str(best_param[k, 0]),
                               SIGMA=str(best_param[k, 1]), NUMBER=str(k + 1))

def run_DIC(lcs,spline,fit_vector, kn,ml,optim_directory, config_file,stream, tolerance = 0.75):
    config = importlib.import_module(config_file)
    pycs.sim.draw.saveresiduals(lcs, spline)
    print "I'll try to recover these parameters :", fit_vector
    dic_opt = mcmc.Dic_Optimiser(lcs, fit_vector, spline, config.attachml, ml, knotstep=kn,
                                 savedirectory=optim_directory,
                                 recompute_spline=True, max_core=config.max_core,
                                 n_curve_stat=config.n_curve_stat,
                                 shotnoise=config.shotnoise_type, tweakml_type=config.tweakml_type,
                                 tweakml_name=config.tweakml_name, display=config.display, verbose=False,
                                 correction_PS_residuals=True, max_iter=config.max_iter, tolerance=tolerance,
                                 theta_init =None)

    chain = dic_opt.optimise()
    dic_opt.analyse_plot_results()
    chi2, B_best = dic_opt.get_best_param()
    A = dic_opt.A_correction
    dic_opt.reset_report()
    dic_opt.report()

    if dic_opt.success:
        print "I succeeded finding a parameter falling in the %2.2f sigma from the original lightcurve."%tolerance

    else:
        print "I didn't find a parameter that falls in the %2.2f sigma from the original lightcurve."%tolerance
        print "I then choose the best one... but be carefull ! "

    for k in range(len(lcs)):
        def tweakml_PS_NUMBER(lcs, spline):
            return twk.tweakml_PS(lcs, spline, B_PARAM, f_min=1 / 300.0, psplot=False, verbose=False,
                                  interpolation='linear', A_correction=A_PARAM)

        util.write_func_append(tweakml_PS_NUMBER, stream,
                               B_PARAM=str(B_best[k][0]), NUMBER=str(k + 1), A_PARAM=str(A[k]))


def main(lensname,dataname,work_dir='./'):
    sys.path.append(work_dir + "config/")
    config_file = "config_" + lensname + "_" + dataname
    config = importlib.import_module(config_file)
    tweakml_plot_dir = config.figure_directory + 'tweakml_plots/'
    optim_directory = tweakml_plot_dir + 'twk_optim_%s_%s/'%(config.optimiser, config.tweakml_name)


    if not os.path.isdir(tweakml_plot_dir):
        os.mkdir(tweakml_plot_dir)

    if config.mltype == "splml":
        if config.forcen :
            ml_param = config.nmlspl
            string_ML ="nmlspl"
        else :
            ml_param = config.mlknotsteps
            string_ML = "knml"
    elif config.mltype == "polyml" :
        ml_param = config.degree
        string_ML = "deg"
    else :
        raise RuntimeError('I dont know your microlensing type. Choose "polyml" or "spml".')

    for i,kn in enumerate(config.knotstep):
        for j, ml in enumerate(ml_param):
            f = open(config.lens_directory + config.combkw[i, j] + '/tweakml_' + config.tweakml_name + '.py', 'w+')
            f.write('import pycs \n')
            f.write('from module import tweakml_PS_from_data as twk \n')
            lcs, spline = pycs.gen.util.readpickle(config.lens_directory + '%s/initopt_%s_ks%i_%s%i.pkl' % (config.combkw[i,j], dataname, kn, string_ML, ml))
            fit_vector = mcmc.get_fit_vector(lcs, spline) #we get the target parameter now
            if not os.path.isdir(optim_directory):
                os.mkdir(optim_directory)

            #We need spline microlensing for tweaking the curve, if it is not the case we change it here to a flat spline that can be tweaked.
            #the resulting mock light curve will have no ML anyway, we will attach it the ML defined in your config file before optimisation.
            polyml = False
            for k,l in enumerate(lcs):
                if l.ml == None:
                    print('I dont have ml, I have to introduce minimal extrinsic variation to generate the mocks. Otherwise I have nothing to modulate.')
                    pycs.gen.splml.addtolc(l, n=2)
                elif l.ml.mltype == 'poly' :
                    polyml = True
                    print('I have polyml and it can not be tweaked. I will replace it with a flat spline just for the mock light curve generation.')
                    l.rmml()

            if polyml :
                spline = pycs.spl.topopt.opt_fine(lcs, nit=5, knotstep=kn,
                                                              verbose=False, bokeps=kn / 3.0,
                                                              stabext=100) # we replace the spline optimised with poly ml by one without ml
                for l in lcs :
                    pycs.gen.splml.addtolc(l, n=2)
                pycs.gen.util.writepickle((lcs,spline), config.lens_directory + '%s/initopt_%s_ks%i_%s%i_generative_polyml.pkl' % (config.combkw[i, j], dataname, kn, string_ML, ml))

            #Starting to write tweakml function depending on tweak_ml_type :
            if config.tweakml_type == 'colored_noise':
                if config.shotnoise_type == None :
                    print 'WARNING : you are using no shotnoise with the colored noise ! That will probably not work.'

                if config.find_tweak_ml_param == True :
                    if config.optimiser == 'PSO':
                        run_PSO(lcs,spline,fit_vector,kn, ml,optim_directory,config_file,f)
                    elif config.optimiser == 'MCMC' :
                        run_MCMC(lcs, spline,fit_vector, kn,ml, optim_directory,config_file,f)
                    else :
                        print "I don't this optimiser, please use PSO or MCMC for the colored noise !"
                        sys.exit()
                else :
                    print "Colored noise : I will add the beta and sigma that you gave in input."
                    for k in range(len(lcs)):
                        def tweakml_colored_NUMBER(lcs,spline):
                            return pycs.sim.twk.tweakml(lcs,spline, beta=BETA,sigma=SIGMA, fmin=1.0 / 500.0, fmax=0.2, psplot=False)

                        util.write_func_append(tweakml_colored_NUMBER, f,
                                               BETA= str(config.colored_noise_param[k][0]), SIGMA = str(config.colored_noise_param[k][1]), NUMBER = str(k+1))

                list_string = 'tweakml_list = ['
                for k in range(len(lcs)):
                    list_string += 'tweakml_colored_'+str(k+1) +','
                list_string +=']'
                f.write('\n')
                f.write(list_string)



            elif config.tweakml_type == 'PS_from_residuals':
                if config.shotnoise_type != None :
                    print 'If you use PS_from_residuals, the shotnoise should be set to None. I will do it for you !'
                    config.shotnoise_type = None

                if config.find_tweak_ml_param == True :
                    if config.optimiser == 'PSO':
                        run_PSO(lcs, spline,fit_vector, kn,ml, optim_directory,config_file,f)
                    elif config.optimiser == 'DIC':
                        run_DIC(lcs, spline,fit_vector, kn,ml, optim_directory,config_file,f)
                    else :
                        print 'I do not recognise your optimiser, please use PSO or DIC with PS_from_residuals'

                else :
                    print "Noise from Power Spectrum of the data : I use PS_param that you gave in input."
                    for k in range(len(lcs)):
                        def tweakml_PS_NUMBER(lcs,spline):
                            return twk.tweakml_PS(lcs, spline, B_PARAM, f_min=1 / 300.0, psplot=False, verbose=False,
                                           interpolation='linear')


                        util.write_func_append(tweakml_PS_NUMBER, f,
                                               B_PARAM= str(config.PS_param_B[k]), NUMBER = str(k+1))

                list_string = 'tweakml_list = ['
                for k in range(len(lcs)):
                    list_string += 'tweakml_PS_'+str(k+1) +','
                list_string +=']'
                f.write('\n')
                f.write(list_string)

            else :
                print "I don't know your tweak_ml_type, please use colored_noise or PS_form_residuals."

            #rename the file :
            files = [file for file in os.listdir(optim_directory)
                     if os.path.isfile(os.path.join(optim_directory, file)) and (string_ML not in file)]

            for file in files:
                prefix,extension = file.split('.')
                os.rename(os.path.join(optim_directory, file), os.path.join(optim_directory, prefix + "_kn%i_%s%i."%(kn,string_ML,ml) + extension))

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Find the noise parameter to reproduce the data.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_lensname = "name of the lens to process"
    help_dataname = "name of the data set to process (Euler, SMARTS, ... )"
    help_work_dir = "name of the working directory"
    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument(dest='dataname', type=str,
                        metavar='dataname', action='store',
                        help=help_dataname)
    parser.add_argument('--dir', dest='work_dir', type=str,
                            metavar='', action='store', default='./',
                            help=help_work_dir)
    args = parser.parse_args()
    main(args.lensname,args.dataname, work_dir=args.work_dir)
