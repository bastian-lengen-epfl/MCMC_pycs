import matplotlib.pyplot as plt
import numpy as np
import pycs
import pycs.regdiff
import copy, os
import time
import multiprocess
from module import tweakml_PS_from_data as twk
from module.plots import plot_functions as pltfct
from cosmoHammer import MpiParticleSwarmOptimizer
from cosmoHammer import ParticleSwarmOptimizer
import pickle
from functools import partial

class Optimiser(object):
    def __init__(self, lcs, fit_vector, spline, attachml_function,attachml_param, knotstep=None,
                     savedirectory="./", recompute_spline=True, max_core = 16, theta_init = None,
                    n_curve_stat = 32, shotnoise = "magerrs", tweakml_type = 'colored_noise', display = False, verbose = False,
                 tweakml_name = '', correction_PS_residuals = True, tolerance = 0.75):

        if len(fit_vector) != len(lcs):
            print "Error : Your target vector and list of light curves must have the same size !"
            exit()
        if recompute_spline == True and knotstep == None :
            print "Error : I can't recompute spline if you don't give me the knotstep ! "
            exit()
        if tweakml_type != 'colored_noise' and tweakml_type != 'PS_from_residuals':
            print "I don't recognize your tweakml type, choose either colored_noise or PS_from_residuals."
            exit()

        self.lcs = lcs
        self.ncurve = len(lcs)
        self.fit_vector = np.asarray(fit_vector)
        self.spline = spline
        self.attachml_function = attachml_function
        self.attachml_param = attachml_param
        if theta_init == None :
            if tweakml_type == 'colored_noise':
                theta_init = [[-2.0,0.1] for i in range(self.ncurve)]
            elif tweakml_type == 'PS_from_residuals':
                theta_init = [[0.5] for i in range(self.ncurve)]
        if len(theta_init) != len(lcs):
            print "Error : Your init vector and list of light curves must have the same size !"
            exit()
        self.theta_init = theta_init
        self.knotstep = knotstep
        self.savedirectory = savedirectory
        self.recompute_spline = recompute_spline
        self.success = False
        self.mean_zruns_mini = None #mean zruns computed with the best parameters
        self.mean_sigma_mini = None #mean sigma computed with the best parameters
        self.std_zruns_mini = None #error on  sigma computed with the best parameters
        self.std_sigma_mini = None #error of zruns and sigma computed with the best parameters
        self.chi2_mini = None
        self.rel_error_zruns_mini= None #relative error in term of zruns
        self.rel_error_sigmas_mini= None #relative error in term of sigmas
        self.best_param = None
        self.time_start = None
        self.time_stop = None

        if max_core != None :
            self.max_core = max_core
        else :
            self.max_core = multiprocess.cpu_count()
            print "You will run on %i cores."%self.max_core

        if self.max_core > 1 :
            self.para = True
        else :
            self.para = False
            print "I won't compute the mock curves in parallel."

        self.n_curve_stat = n_curve_stat
        self.shotnoise = shotnoise
        self.shotnoisefrac = 1.0
        self.tweakml_type =tweakml_type
        self.tweakml_name = tweakml_name
        self.correction_PS_residuals = correction_PS_residuals #boolean to set if you want to use the correction, True by default
        self.A_correction = np.ones(self.ncurve) # this is the correction for the amplitude of the power spectrum of the risduals, this is use only for PS_from_residuals
        self.display = display
        self.verbose = verbose
        self.grid = None
        self.message = '\n'
        self.error_message = []
        self.tolerance = tolerance # tolerance in unit of sigma for the fit
        self.timeshifts = [l.timeshift for l in self.lcs]
        self.magshifts = [l.magshift for l in self.lcs]

    def make_mocks_para(self, theta):
        stat = []
        zruns = []
        sigmas = []
        nruns = []

        pool = multiprocess.Pool(processes=self.max_core)

        job_args = [(theta) for j in range(self.n_curve_stat)]

        out = pool.map(self.fct_para, job_args)
        pool.close()
        pool.join()

        stat_out = np.asarray([x['stat'] for x in out if x['stat'] is not None]) #clean failed optimisation
        message_out = np.asarray([x['error'] for x in out if x['error'] is not None]) #clean failed optimisation
        self.error_message.append(message_out)
        zruns = np.asarray([[stat_out[i,j]['zruns'] for j in range(self.ncurve)] for i in range(len(stat_out))])
        sigmas = np.asarray([[stat_out[i,j]['std'] for j in range(self.ncurve)] for i in range(len(stat_out))])
        nruns = np.asarray([[stat_out[i,j]['nruns'] for j in range(self.ncurve)] for i in range(len(stat_out))])

        mean_zruns = []
        std_zruns = []
        mean_sigmas = []
        std_sigmas = []

        for i in range(self.ncurve):
            mean_zruns.append(np.mean(zruns[:, i]))
            std_zruns.append(np.std(zruns[:, i]))
            mean_sigmas.append(np.mean(sigmas[:, i]))
            std_sigmas.append(np.std(sigmas[:, i]))
            if self.verbose :
                print 'Curve %i :' % (i + 1)
                print 'Mean zruns (simu): ', np.mean(zruns[:, i]), '+/-', np.std(zruns[:, i])
                print 'Mean sigmas (simu): ', np.mean(sigmas[:, i]), '+/-', np.std(sigmas[:, i])
                print 'Mean nruns (simu): ', np.mean(nruns[:, i]), '+/-', np.std(nruns[:, i])

        return mean_zruns, mean_sigmas, std_zruns, std_sigmas, zruns, sigmas


    def compute_chi2(self, theta):
        #theta : proposed step
        #fit vector : target vector to fit in terms of [zruns, sigma], zruns are a list of the target zruns

        chi2 = 0.0
        count = 0.0
        if self.n_curve_stat == 1:
            print "Warning : I cannot compute statistics with one single curves !!"

        if self.para:
            mean_zruns, mean_sigmas, std_zruns, std_sigmas, _ , _  = self.make_mocks_para(theta)
        else:
            mean_zruns, mean_sigmas, std_zruns, std_sigmas, _ ,_  = self.make_mocks(theta)

        # for i in range(len(out)):
        #     chi2 += (fit_vector[i] - out[i]) ** 2 / error[i] ** 2

        for i in range(self.ncurve):
            chi2 += (self.fit_vector[i][0] - mean_zruns[i]) ** 2 / std_zruns[i] ** 2
            chi2 += (self.fit_vector[i][1] - mean_sigmas[i]) ** 2 / (std_sigmas[i] ** 2)
            count +=1.0

        chi2 = chi2 / count
        return chi2, np.asarray(mean_zruns), np.asarray(mean_sigmas), np.asarray(std_zruns), np.asarray(std_sigmas)

    def fct_para(self, theta):

        tweak_list = self.get_tweakml_list(theta)
        mocklc = pycs.sim.draw.draw(self.lcs, self.spline,
                                    tweakml= tweak_list,shotnoise=self.shotnoise,scaletweakresi = False,
                                    shotnoisefrac=self.shotnoisefrac, keeptweakedml=False, keepshifts=False,
                                    keeporiginalml=False, inprint_fake_shifts= None) #this return mock curve without ML

        # sampleshifts = [np.random.uniform(low=-10, high=10, size=1) + self.timeshifts[i] for i in
        #                 range(self.ncurve)]
        self.applyshifts(mocklc, self.timeshifts, self.magshifts)
        self.attachml_function(mocklc, self.attachml_param)  # adding the microlensing here ! Before the optimisation

        if self.recompute_spline:
            if self.knotstep == None:
                print "Error : you must give a knotstep to recompute the spline"
            try :
                spline_on_mock = pycs.spl.topopt.opt_fine(mocklc, nit=5, knotstep=self.knotstep,
                                                          verbose=self.verbose, bokeps=self.knotstep/3.0, stabext=100)
                mockrls = pycs.gen.stat.subtract(mocklc, spline_on_mock)
                stat = pycs.gen.stat.mapresistats(mockrls)
            except Exception as e:
                print 'Warning : light curves could not be optimised for parameter :', theta
                error_message = 'The following error occured : %s for parameters %s \n' %(e, str(theta))
                return {'stat' : None, 'error' : error_message }
            else :
                return {'stat' : stat, 'error' : None }
        else:
            mockrls = pycs.gen.stat.subtract(mocklc, self.spline)
            stat = pycs.gen.stat.mapresistats(mockrls)
            return {'stat' : stat, 'error' : None }


    def get_tweakml_list(self, theta):
        tweak_list = []
        if self.tweakml_type == 'colored_noise':
            def tweakml_colored(lcs, spline, beta, sigma):
                return pycs.sim.twk.tweakml(lcs,spline, beta=beta, sigma=sigma, fmin=1.0 / 500.0, fmax=0.2,
                                            psplot=False)
            for i in range(self.ncurve):
                tweak_list.append(partial(tweakml_colored, beta=theta[i][0], sigma=theta[i][1]))

        elif self.tweakml_type == 'PS_from_residuals':
            def tweakml_PS(lcs, spline, B, A_correction):
                return twk.tweakml_PS(lcs, spline, B, f_min=1 / 300.0,psplot=False, save_figure_folder=None,
                                     verbose=self.verbose,interpolation='linear',A_correction=A_correction)
            for i in range(self.ncurve):
                tweak_list.append(partial(tweakml_PS, B = theta[i][0], A_correction = self.A_correction[i]))
        return tweak_list


    def fct_para_aux(self,args):
        kwargs = args[-1]
        args = args[0:-1]
        return self.fct_para(*args, **kwargs)


    def make_mocks(self, theta):

        mocklc = []
        mockrls = []
        stat = []
        zruns = []
        sigmas = []
        nruns = []

        for i in range(self.n_curve_stat):
            tweak_list = self.get_tweakml_list(theta)
            mocklc.append(pycs.sim.draw.draw(self.lcs, self.spline,
                                        tweakml=tweak_list, shotnoise=self.shotnoise,shotnoisefrac=self.shotnoisefrac,
                                             keeptweakedml=False, keepshifts=False, keeporiginalml=False,
                                             scaletweakresi = False, inprint_fake_shifts= None)) # this will return mock curve WITHOUT microlensing !

            # print mocklc[i][0].ml
            self.applyshifts(mocklc[i], self.timeshifts, self.magshifts)
            self.attachml_function(mocklc[i], self.attachml_param) # adding the microlensing here
            # print mocklc[i][0].ml

            if self.recompute_spline:
                if self.knotstep == None:
                    print "Error : you must give a knotstep to recompute the spline"
                spline_on_mock = pycs.spl.topopt.opt_fine(mocklc[i], nit=5, knotstep=self.knotstep,
                                                          verbose=self.verbose, bokeps=self.knotstep/3.0, stabext=100) #TODO : maybe pass the optimisation function to the class in argument
                mockrls.append(pycs.gen.stat.subtract(mocklc[i], spline_on_mock))
                # pycs.gen.lc.display(mocklc[i], [spline_on_mock], showlegend=True, showdelays=True, filename="screen")
                # pycs.gen.stat.plotresiduals([mockrls[i]])
            else:
                mockrls.append(pycs.gen.stat.subtract(mocklc[i], self.spline))


            if self.recompute_spline and self.display:
                    pycs.gen.lc.display([self.lcs], [spline_on_mock], showdelays=True)
                    pycs.gen.stat.plotresiduals([mockrls[i]])

            stat.append(pycs.gen.stat.mapresistats(mockrls[i]))
            zruns.append([stat[i][j]['zruns'] for j in range(self.ncurve)])
            sigmas.append([stat[i][j]['std'] for j in range(self.ncurve)])
            nruns.append([stat[i][j]['nruns'] for j in range(self.ncurve)])

        zruns = np.asarray(zruns)
        sigmas = np.asarray(sigmas)
        nruns = np.asarray(nruns)
        mean_zruns = []
        std_zruns = []
        mean_sigmas = []
        std_sigmas = []
        for i in range(self.ncurve):
            mean_zruns.append(np.mean(zruns[:, i]))
            std_zruns.append(np.std(zruns[:, i]))
            mean_sigmas.append(np.mean(sigmas[:, i]))
            std_sigmas.append(np.std(sigmas[:, i]))
            if self.verbose :
                print 'Curve %s :' % self.lcs[i].object
                print 'Mean zruns (simu): ', np.mean(zruns[:, i]), '+/-', np.std(zruns[:, i])
                print 'Mean sigmas (simu): ', np.mean(sigmas[:, i]), '+/-', np.std(sigmas[:, i])
                print 'Mean nruns (simu): ', np.mean(nruns[:, i]), '+/-', np.std(nruns[:, i])

        return mean_zruns, mean_sigmas, std_zruns, std_sigmas, zruns, sigmas

    def check_success(self):
        if any(self.rel_error_zruns_mini[i] == None for i in range(self.ncurve)) :
            print "Error you should run analyse_plot_results() first !"
            exit()
        else :
            if all(self.rel_error_zruns_mini[i] < self.tolerance for i in range(self.ncurve)) \
                    and all(self.rel_error_sigmas_mini[i] < self.tolerance for i in range(self.ncurve)):
                return True
            else :
                return False

    def report(self):
        if self.chain_list == None:
            print "Error : you should run optimise() first !"
            print "I can't write the report"
            exit()

        f = open(self.savedirectory + 'report_tweakml_optimisation.txt', 'a')
        for i in range(self.ncurve):

            f.write('Lightcurve %s : \n'%self.lcs[i].object)
            f.write('\n')
            if self.rel_error_zruns_mini[i] < self.tolerance and self.rel_error_sigmas_mini[i] < self.tolerance:
                f.write('I succeeded in finding a set of parameters that match the '
                        'statistical properties of the real lightcurve within %2.2f sigma. \n'%self.tolerance)

            else :
                f.write('I did not succeed in finding a set of parameters that '
                        'match the statistical properties of the real lightcurve within %2.2f sigma. \n'%self.tolerance)

            f.write(self.message)
            f.write('Best parameters are : %s \n'%str(self.best_param[i]) )
            if self.tweakml_type == 'PS_from_residuals':
                f.write('A correction for PS_from_residuals : %2.2f \n'%self.A_correction[i])
            f.write("Corresponding Chi2 : %2.2f \n" %self.chi2_mini)
            f.write("Target zruns, sigma : %2.6f, %2.6f \n"%(self.fit_vector[i,0],self.fit_vector[i,1]))
            f.write("At minimum zruns, sigma : %2.6f +/- %2.6f, %2.6f +/- %2.6f \n"%(self.mean_zruns_mini[i],self.std_zruns_mini[i],
                                                                                     self.mean_sigma_mini[i], self.std_sigma_mini[i]))
            f.write("For minimum Chi2, we are standing at " + str(self.rel_error_zruns_mini[i]) + " sigma [zruns] \n")
            f.write("For minimum Chi2, we are standing at " + str(self.rel_error_sigmas_mini[i])+ " sigma [sigma] \n")
            f.write('------------------------------------------------\n')
            f.write('\n')
        f.write('Optimisation done in %4.4f seconds on %i cores'%((self.time_stop - self.time_start),self.max_core))
        f.close()

        #Write the error report :
        g = open(self.savedirectory + 'errors_tweakml_optimisation.txt', 'a')
        for mes in self.error_message:
            g.write(mes)
        g.close()

    def reset_report(self):
        if os.path.isfile(self.savedirectory + 'report_tweakml_optimisation.txt'):
            os.remove(self.savedirectory + 'report_tweakml_optimisation.txt')

    def compute_set_A_correction(self, eval_pts, reset_A = True):
        #this function compute the sigma obtained after optimisation in the middle of the grid and return the correction that will be used for the rest of the optimisation

        self.A_correction = [1.0 for i in range(self.ncurve)] #reset the A correction

        if self.para:
            mean_zruns, mean_sigmas, std_zruns, std_sigmas, _, _ = self.make_mocks_para(eval_pts)
        else:
            mean_zruns, mean_sigmas, std_zruns, std_sigmas, _, _ = self.make_mocks(eval_pts)

        self.A_correction = self.fit_vector[:,1] / mean_sigmas # set the A correction
        return self.A_correction, mean_zruns, mean_sigmas, std_zruns, std_sigmas

    def applyshifts(self, lcs, timeshifts, magshifts):

        if not len(lcs) == len(timeshifts) and len(lcs) == len(magshifts):
            print "Hey, give me arrays of the same lenght !"
            sys.exit()

        for lc, timeshift, magshift in zip(lcs, timeshifts, magshifts):
            lc.resetshifts()
            # lc.shiftmag(-np.median(lc.getmags()))
            lc.shiftmag(magshift)
            lc.shifttime(timeshift)


class Metropolis_Hasting_Optimiser(Optimiser):
    def __init__(self, lcs, fit_vector, spline, attachml_function,attachml_param, knotstep=None, n_iter=1000,
                    burntime=100, savedirectory="./", recompute_spline=True, rdm_walk='gaussian', max_core = 16,
                    n_curve_stat = 32, stopping_condition = True, shotnoise = "magerrs", theta_init = None
                 , gaussian_step = [0.1, 0.01], tolerance = 0.75,
                 tweakml_type = 'coloired_noise' ,tweakml_name = '',correction_PS_residuals = True):

        Optimiser.__init__(self,lcs, fit_vector,spline, attachml_function,attachml_param, knotstep = knotstep,
                           savedirectory= savedirectory, recompute_spline=recompute_spline,
                            max_core =max_core, n_curve_stat = n_curve_stat, shotnoise = shotnoise, theta_init= theta_init,
                           tweakml_type= tweakml_type, tweakml_name= tweakml_name, correction_PS_residuals = correction_PS_residuals, tolerance = tolerance)
        self.n_iter = n_iter
        self.burntime = burntime
        self.rdm_walk = rdm_walk
        self.stopping_condition = stopping_condition
        self.gaussian_step = gaussian_step
        open(self.savedirectory + self.tweakml_name + '_MCMC_outfile_i' + str(n_iter) + "_" + rdm_walk + ".txt",
             'w').close()  # to clear the file
        self.savefile =open(self.savedirectory + self.tweakml_name + '_MCMC_outfile_i' + str(n_iter) + "_" + rdm_walk + ".txt",'a')
        self.hundred_last = 100
        self.chain_list = None


    def optimise(self):

        theta_save = []
        chi2_save = []
        z_save = []
        s_save = []
        errorz_save = []
        errors_save = []
        self.hundred_last = 100
        theta = np.asarray(copy.deepcopy(self.theta_init))
        chi2_current, mean_zruns_current, mean_sigmas_current, std_zruns_current, std_sigmas_current = self.compute_chi2(self.theta_init)
        self.time_start = time.time()

        for i in range(self.n_iter):
            t_now = time.time() - self.time_start
            print "time : ", t_now

            if self.rdm_walk == 'gaussian':
                theta_new = np.asarray(self.make_random_step_gaussian(theta, self.gaussian_step))
            elif self.rdm_walk == 'log':
                theta_new = np.asarray(self.make_random_step_log(theta, self.gaussian_step))

            if not self.prior(theta_new):
                print "Rejected sample (outside the hard bound) :", theta_new
                theta_save.append(theta)
                chi2_save.append(chi2_current)
                z_save.append(mean_zruns_current)
                s_save.append(mean_sigmas_current)
                errorz_save.append(std_zruns_current)
                errors_save.append(std_sigmas_current)
                continue

            chi2_new, mean_zruns_new, mean_sigmas_new, std_zruns_new, std_sigmas_new = self.compute_chi2(theta_new)
            ratio = np.exp((-chi2_new + chi2_current) / 2.0);
            if self.verbose :
                print "Iteration, Theta, Chi2, mean zruns, mean sigmas, std zruns, std sigmas :", i, theta_new, chi2_new, mean_zruns_current, \
                mean_sigmas_current, std_zruns_current, std_sigmas_current

            if np.random.rand() < ratio:
                print "step %i accepted with ratio :"%i, ratio
                theta = copy.deepcopy(theta_new)
                chi2_current = copy.deepcopy(chi2_new)
                mean_zruns_current = copy.deepcopy(mean_zruns_new)
                mean_sigmas_current = copy.deepcopy(mean_sigmas_new)
                std_zruns_current = copy.deepcopy(std_zruns_new)
                std_sigmas_current = copy.deepcopy(std_sigmas_new)
            else:
                print "step %i rejected with ratio :" %i , ratio

            theta_save.append(theta)
            chi2_save.append(chi2_current)
            z_save.append(mean_zruns_current)
            s_save.append(mean_sigmas_current)
            errorz_save.append(std_zruns_current)
            errors_save.append(std_sigmas_current)

            #write in file :
            data = []
            for i in range(self.ncurve):
                for j in range(len(theta[0,:])):
                    data.append(np.asarray(theta)[i,j])
            data.append(chi2_current)
            [data.append(mean_zruns_current[i]) for i in range(self.ncurve)]
            [data.append(mean_sigmas_current[i]) for i in range(self.ncurve)]
            [data.append(std_zruns_current[i]) for i in range(self.ncurve)]
            [data.append(std_sigmas_current[i]) for i in range(self.ncurve)]
            data = np.asarray(data)
            np.savetxt(self.savefile, data, newline=' ', delimiter=',')

            if self.stopping_condition == True:
                if self.check_if_stop(self.fit_vector, mean_zruns_current,mean_sigmas_current, std_zruns_current, std_sigmas_current):
                    break

        self.chain_list = [theta_save, chi2_save, z_save, s_save, errorz_save, errors_save] #theta_save has dimension(n_iter,ncurve,2)
        self.save_best_param()  # to save the best params
        self.success = self.check_success()
        self.time_stop = time.time()
        return theta_save, chi2_save, z_save, s_save, errorz_save, errors_save


    def prior(self,theta):
        if all(-8.0< theta[i,0] < 0.0 for i in range(self.ncurve)) and all(0 < theta[i,1] < 0.5 for i in range(self.ncurve)):
            return True
        else:
            return False


    def make_random_step_gaussian(self,theta, sigma_step):
        theta_new = [theta[i][:] + sigma_step * np.random.randn(2) for i in range(self.ncurve)]
        print theta_new
        return theta_new

    def make_random_step_log(self,theta, sigma_step):
        s = np.asarray(theta)[:,1]
        s = np.log10(s) + sigma_step[1] * np.random.randn(self.ncurve)
        s = 10.0**s
        return [[theta[i][0] + sigma_step[0] * np.random.randn(),s[i]] for i in range(self.ncurve)]


    def check_if_stop(self,fitvector, mean_zruns, mean_sigmas, std_zruns, std_sigmas):
        if self.hundred_last != 100 :
            self.hundred_last -= 1# check if we already reached the condition once
            print "I have already matched the stopping condition, I will do %i more steps." %self.hundred_last

        elif all(np.abs(fitvector[i,0] - mean_zruns[i]) < self.tolerance*std_zruns[i] for i in range(self.ncurve)) \
                and all(np.abs(fitvector[i,1] - mean_sigmas[i]) < self.tolerance*std_sigmas[i] for i in range(self.ncurve)):
            self.hundred_last -= 1
            print "I'm matching the stopping condition at this iteration, I will do %i more steps."%self.hundred_last
        else :
            print "Stopping condition not reached."

        if self.hundred_last == 0 :
            return True
        else :
            return False

    def save_best_param(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else:
            print self.chain_list[1][:]
            ind_min = np.argmin(self.chain_list[1][:])
            self.mean_zruns_mini = self.chain_list[2][:][ind_min]
            self.mean_sigma_mini = self.chain_list[3][:][ind_min]
            self.std_zruns_mini = self.chain_list[4][:][ind_min]
            self.std_sigma_mini = self.chain_list[5][:][ind_min]
            self.chi2_mini = self.chain_list[1][ind_min]
            self.best_param = self.chain_list[0][:][ind_min]
            self.rel_error_zruns_mini = np.abs(self.mean_zruns_mini - self.fit_vector[:,0]) / self.std_zruns_mini
            self.rel_error_sigmas_mini = np.abs(self.mean_sigma_mini - self.fit_vector[:,1]) / self.std_sigma_mini

    def analyse_plot_results(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            self.save_best_param()
            print "Best position : ", self.best_param
            print "Corresponding Chi2 : ", self.chi2_mini

            print "Target sigma, zruns : ", self.fit_vector
            print "Minimum sigma, zruns : ", [self.mean_zruns_mini, self.mean_sigma_mini]
            print "Minimum chi2 : ", self.chi2_mini
            print "For minimum Chi2, we are standing at " + str(self.rel_error_zruns_mini) + " sigma [zruns]"
            print "For minimum Chi2, we are standing at " + str(self.rel_error_sigmas_mini)+ " sigma [sigma]"

            self.success = self.check_success()

            for i in range(self.ncurve):
                theta_chain = np.asarray(self.chain_list[0])
                theta_chain[:,i,1] = np.log10(theta_chain[:,i,1])
                fig2,fig3 = pltfct.plot_chain_MCMC(theta_chain[:,i,:], self.chain_list[1], ["$beta$", "log $\sigma$"])
                fig2.savefig(self.savedirectory +self.tweakml_name + "_MCMC_chi2_" + self.lcs[i].object + ".png")
                fig3.savefig(self.savedirectory +self.tweakml_name +"_MCMC_chain_" + self.lcs[i].object + ".png")

                if self.display:
                    plt.show()
            param_name = [["$beta_%i$"%(i+1),"$sigma_%i$"%(i+1)] for i in range(self.ncurve)]
            param_name = np.reshape(param_name,2 * self.ncurve )
            if self.n_iter > 4 * self.ncurve : #to be sure to have enough points
                fig1 = pltfct.corner_plot_MCMC(np.reshape(theta_chain, (self.n_iter, 2 * self.ncurve)), param_name)
                fig1.savefig(self.savedirectory + self.tweakml_name + "_MCMC_corner_plot.png")
#TODO : implement the burnin in the plot !!
    def dump_results(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            pickle.dump(self.chain_list, open(self.savedirectory +  self.tweakml_name + "_chain_list_MCMC" + "_i"
                                         + str(self.n_iter) + ".pkl", "wb"))
            pickle.dump(self, open(self.savedirectory + self.tweakml_name + "_MCMC_opt" + "_i"
                                      + str(self.n_iter) + ".pkl", "wb"))


class PSO_Optimiser(Optimiser) :
    #Attention here : You cannot use the parrallel computing of the mock curves because PSO, already launch the particles on several thread !

    def __init__(self, lcs, fit_vector, spline,attachml_function,attachml_param,savedirectory ="./", knotstep = None, max_core = 8, shotnoise = 'magerrs',
                 recompute_spline = True, n_curve_stat= 32, theta_init = None, n_particles = 30, n_iter = 50, verbose = False,
                 lower_limit = [-8., 0.], upper_limit = [-1.0, 0.5], mpi = False, tweakml_type = 'colored_noise', tweakml_name = '',
                 correction_PS_residuals = True, tolerance = 0.75):

        Optimiser.__init__(self,lcs, fit_vector,spline,attachml_function,attachml_param, knotstep = knotstep,
                           savedirectory= savedirectory, recompute_spline=recompute_spline,
                            max_core =1, n_curve_stat = n_curve_stat, shotnoise = shotnoise, theta_init= theta_init,
                           tweakml_type = tweakml_type, tweakml_name = tweakml_name, correction_PS_residuals= correction_PS_residuals,
                           verbose= verbose, display= False, tolerance = tolerance)


        self.n_particles = n_particles
        self.n_iter = n_iter

        if self.tweakml_type == "colored_noise" :
            self.lower_limit = [lower_limit[0] for i in range(self.ncurve)] + [lower_limit[1] for i in range(self.ncurve)] #PSO needs another format of the parameter to fit
            self.upper_limit = [upper_limit[0] for i in range(self.ncurve)] + [upper_limit[1] for i in range(self.ncurve)]
        elif self.tweakml_type == "PS_from_residuals" :
            self.lower_limit = [0.0 for i in range(self.ncurve)]
            self.upper_limit = [2.0 for i in range(self.ncurve)]
        else :
            print "ERROR : unknown tweak_ml type, choose colored_noise or PS_from_residuals"
            exit()

        self.mpi = mpi
        self.max_thread = max_core
        self.chain_list = None
        self.savefile = self.savedirectory + self.tweakml_name + '_PSO_file' + "_i" + str(self.n_iter)+"_p"+str(self.n_particles)+ ".txt"

    def __call__(self, theta):
        return self.likelihood(theta)

    def likelihood(self, theta):
        #reformat theta to macth the other optimiser :
        theta = self.reformat(theta)
        chi2, zruns , sigmas , zruns_std , sigmas_std = self.compute_chi2(theta)
        return [-0.5*chi2]

    def reformat(self,theta):
        if self.tweakml_type == "colored_noise":
            theta_ref = [[theta[i],theta[i+self.ncurve]] for i in range(self.ncurve)]

        elif self.tweakml_type == "PS_from_residuals":
            theta_ref = [[theta[i]] for i in range(self.ncurve)]
        return theta_ref


    def optimise(self):
        self.time_start = time.time()
        if self.tweakml_type == "PS_from_residuals":
            self.A_correction, _ , _, _, _  = self.compute_set_A_correction(self.theta_init)

        if self.mpi is True:
            pso = MpiParticleSwarmOptimizer(self, self.lower_limit, self.upper_limit, self.n_particles)
        else:
            pso = ParticleSwarmOptimizer(self, self.lower_limit, self.upper_limit, self.n_particles, threads=self.max_thread)

        X2_list = []
        vel_list = []
        pos_list = []
        num_iter = 0

        f = open(self.savefile, "wb")

        for swarm in pso.sample(self.n_iter):
            print "iteration : ", num_iter
            X2_list_c = np.asarray(pso.gbest.fitness)* -2.0
            X2_list.append(X2_list_c)
            vel_list_c = np.asarray(pso.gbest.velocity)
            vel_list.append(vel_list_c.tolist())
            pos_list_c = np.asarray(pso.gbest.position)
            pos_list.append(pos_list_c.tolist())
            print X2_list_c, vel_list_c, pos_list_c
            data = np.concatenate(([X2_list_c],vel_list_c,pos_list_c))
            data = [str(d) for d in data.tolist()]
            f.write(" ".join(data))
            f.write("\n")
            # np.savetxt(f, data.transpose(), delimiter=',', newline='\n')
            num_iter += 1

        f.close()
        self.chain_list = [X2_list, pos_list, vel_list]
        self.chi2_mini, self.best_param = self.get_best_param()
        self.time_stop = time.time()
        return self.chain_list

    def get_best_param(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            best_chi2 = self.chain_list[0][-1]
            best_param = np.asarray(self.reformat(self.chain_list[1][-1][:]))
            return best_chi2,best_param

    def dump_results(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            pickle.dump(self.chain_list, open(self.savedirectory + self.tweakml_name +"_chain_list_PSO_" + "_i"
                                         + str(self.n_iter) + "_p" + str(self.n_particles) + ".pkl", "wb"))
            pickle.dump(self, open(self.savedirectory + self.tweakml_name +"_PSO_opt_" + "_i"
                                      + str(self.n_iter) + "_p" + str(self.n_particles) + ".pkl", "wb"))
    def analyse_plot_results(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            print "Converged position :", self.best_param
            print "Converged Chi2 : ", self.chi2_mini
            mean_zruns_mini, mean_sigmas_mini, std_zruns_mini, std_sigmas_mini, _, _ = self.make_mocks_para(self.best_param)
            self.mean_zruns_mini= np.asarray(mean_zruns_mini)
            self.mean_sigma_mini= np.asarray(mean_sigmas_mini)
            self.std_zruns_mini = np.asarray(std_zruns_mini)
            self.std_sigma_mini = np.asarray(std_zruns_mini)
            self.rel_error_zruns_mini = np.abs(self.mean_zruns_mini - self.fit_vector[:, 0]) / self.std_zruns_mini
            self.rel_error_sigmas_mini = np.abs(self.mean_sigma_mini - self.fit_vector[:, 1]) / self.std_sigma_mini

            print "Target sigma, zruns : " + str(self.fit_vector[:,1]) + ', ' + str(self.fit_vector[:,0])
            print "Minimum sigma, zruns : " + str(self.mean_sigma_mini) + ', ' + str(mean_zruns_mini)
            print "For minimum Chi2, we are standing at " + str(self.rel_error_zruns_mini[0]) + " sigma [zruns]"
            print "For minimum Chi2, we are standing at " + str(self.rel_error_sigmas_mini[1])+ " sigma [sigma]"

            self.success = self.check_success()

            if self.tweakml_type == 'PS_from_residuals':
                param_list = ['$B_%i$' %i for i in range(self.ncurve)]
            elif self.tweakml_type == 'colored_noise':
                param_list = ['$beta_%i$'%i for i in range(self.ncurve)] + ['$\sigma_%i$'%i for i in range(self.ncurve)]
            f, axes = pltfct.plot_chain_PSO(self.chain_list, param_list)

            f.savefig(self.savedirectory + self.tweakml_name + "_PSO_chain.png")

            if self.display:
                plt.show()

class Dic_Optimiser(Optimiser):
    def __init__(self, lcs, fit_vector, spline,attachml_function,attachml_param, knotstep=None,
                     savedirectory="./", recompute_spline=True, max_core = 16, theta_init = None,
                    n_curve_stat = 32, shotnoise = None, tweakml_type = 'PS_from_residuals', tweakml_name = '',
                 display = False, verbose = False, step = 0.1, correction_PS_residuals = True, max_iter = 10, tolerance = 0.75):

        Optimiser.__init__(self,lcs, fit_vector,spline,attachml_function,attachml_param,
                           knotstep = knotstep, savedirectory= savedirectory, recompute_spline=recompute_spline,
                                   max_core =max_core, n_curve_stat = n_curve_stat, shotnoise = shotnoise, theta_init= theta_init,
                           tweakml_type= tweakml_type, tweakml_name = tweakml_name, correction_PS_residuals=correction_PS_residuals,
                           verbose=verbose, display= display, tolerance = tolerance)

        self.chain_list = None
        self.step = [step for i in range(self.ncurve)]
        self.max_iter = max_iter
        self.iteration = 0
        self.turn_back = [0 for i in range(self.ncurve)]
        self.explored_param = []

    def optimise(self):
        self.time_start = time.time()
        sigma = []
        zruns = []
        sigma_std = []
        zruns_std = []
        chi2 = []
        zruns_target = self.fit_vector[:,0]
        sigma_target = self.fit_vector[:,1]
        B = copy.deepcopy(self.theta_init)

        if self.correction_PS_residuals:
            self.A_correction, _ ,_ ,_ , _ = self.compute_set_A_correction(B)
            print "I will slightly correct the amplitude of the Power Spectrum by a factor :", self.A_correction

        while True:
            self.iteration +=1
            print "Iteration %i, B vector : "%self.iteration, B
            chi2_c, zruns_c, sigma_c, zruns_std_c, sigma_std_c = self.compute_chi2(B)

            chi2.append(chi2_c)
            sigma.append(sigma_c)
            zruns.append(zruns_c)
            sigma_std.append(sigma_std_c)
            zruns_std.append(zruns_std_c)
            self.explored_param.append(copy.deepcopy(B))

            self.rel_error_zruns_mini = np.abs(zruns_c - self.fit_vector[:, 0]) / zruns_std_c #used to store the current relative error
            self.rel_error_sigmas_mini = np.abs(sigma_c - self.fit_vector[:, 1]) / sigma_std_c

            for i in range(self.ncurve):
                if self.step[i] > 0 and zruns_c[i] > zruns_target[i]:
                    self.turn_back[i] +=1
                    if self.iteration != 1:
                        self.step[i] =  - self.step[i] /2.0 # we go backward dividing the step by two
                    else :
                        self.step[i] =  - self.step[i]# we do two step backward if the first iteration was already too high.

                elif self.step[i] < 0 and zruns_c[i] < zruns_target[i]:
                    self.turn_back[i] += 1
                    self.step[i] = - self.step[i] / 2.0  # we go backward dividing the step by two

                elif self.step[i] > 0.6 : #max step size
                    self.step[i] = 0.6

                elif B[i][0] <= 0.4 and self.step[i] <= -0.2 : #condition to reach 0.1 aymptotically

                    self.step = self.step/ 2.0

                elif self.iteration%3 == 0 and self.turn_back[i] == 0:
                    self.step[i] = self.step[i]*2.0 #we double the step every 3 iterations if we didn't pass the optimum

            if self.check_if_stop():
                break

            for i in range(self.ncurve):
                B[i][0] += self.step[i]
                if B[i][0] <= 0.05 : B[i][0] = 0.05 #minimum for B

            if self.iteration%5 == 0:
                self.A_correction, _, _, _, _ = self.compute_set_A_correction(B) #recompute A correction every 5 iterations.
                print "I will slightly correct the amplitude of the Power Spectrum by a factor :", self.A_correction

        self.chain_list = [self.explored_param, chi2, zruns, sigma, zruns_std, sigma_std]#explored param has dimension(n_iter,ncurve,1)
        self.chi2_mini, self.best_param = chi2[-1], self.explored_param[-1] # take the last iteration as the best estimate
        self.mean_zruns_mini = zruns[-1]
        self.std_zruns_mini = zruns_std[-1]
        self.mean_sigma_mini = sigma[-1]
        self.std_sigma_mini = sigma_std[-1]
        self.success = self.check_success()
        self.time_stop = time.time()
        return self.chain_list

    def check_if_stop(self):
        if self.iteration >= self.max_iter:
            self.message = "I stopped because I reached the max number of iteration.\n"
            print self.message[:-2]
            return True
        if all(self.turn_back[i] > 4 for i in range(self.ncurve)):
            self.message = "I stopped because I passed four times the optimal value for all the curves.\n"
            print self.message[:-2]
            return True
        if self.check_success():
            self.message = "I stopped because I found a good set of parameters. \n"
            print self.message[:-2]
            return True
        else :
            return False

    def get_best_param(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else:
            ind_min = np.argmin(self.chain_list[1][:])
            self.chi2_mini = np.min(self.chain_list[1][:])
            return self.chi2_mini, self.chain_list[0][ind_min]

    def analyse_plot_results(self):
        pltfct.plot_chain_grid_dic(self)

def get_fit_vector(lcs,spline):
    rls = pycs.gen.stat.subtract(lcs, spline)
    fit_sigma = [pycs.gen.stat.mapresistats(rls)[i]["std"] for i in range(len(rls))]
    fit_zruns = [pycs.gen.stat.mapresistats(rls)[i]["zruns"] for i in range(len(rls))]
    fit_vector = [[fit_zruns[i], fit_sigma[i]] for i in range(len(rls))]
    return fit_vector
