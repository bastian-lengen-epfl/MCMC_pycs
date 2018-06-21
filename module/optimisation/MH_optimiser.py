from module.plots import plot_functions as pltfct
from module.optimisation import Optimiser
import time

import matplotlib.pyplot as plt
import pickle
import dill #this is important for multiprocessing
import numpy as np
import pycs
import copy, os



class Metropolis_Hasting_Optimiser(Optimiser):
    def __init__(self, lc, fit_vector, spline, knotstep=None, n_iter=1000,
                    burntime=100, savedirectory="./", recompute_spline=True, rdm_walk='gaussian', max_core = 16,
                    n_curve_stat = 32, stopping_condition = True, shotnoise = "magerrs", theta_init = [-2.0, 0.2], gaussian_step = [0.1, 0.01],
                 tweakml_type = 'coloired_noise' ,tweakml_name = '',correction_PS_residuals = True):

        Optimiser.__init__(self,lc, fit_vector,spline, knotstep = knotstep, savedirectory= savedirectory, recompute_spline=recompute_spline,
                                   max_core =max_core, n_curve_stat = n_curve_stat, shotnoise = shotnoise, theta_init= theta_init,
                           tweakml_type= tweakml_type, tweakml_name= tweakml_name, correction_PS_residuals = correction_PS_residuals)
        self.n_iter = n_iter
        self.burntime = burntime
        self.rdm_walk = rdm_walk
        self.stopping_condition = stopping_condition
        self.gaussian_step = gaussian_step
        self.savefile = self.savedirectory + self.tweakml_name + '_MCMC_outfile_i' + str(n_iter)+"_"+rdm_walk +"_"+self.lc.object+'.txt'
        self.hundred_last = 100
        self.chain_list = None


    def optimise(self):

        theta_save = []
        chi2_save = []
        sz_save = []
        errorsz_save = []
        self.hundred_last = 100
        theta = copy.deepcopy(self.theta_init)
        chi2_current, sz_current, errorsz_current = self.compute_chi2(self.theta_init)
        t = time.time()

        for i in range(self.n_iter):
            t_now = time.time() - t
            print "time : ", t_now

            if self.rdm_walk == 'gaussian':
                theta_new = self.make_random_step_gaussian(theta, self.gaussian_step)
            elif self.rdm_walk == 'exp':
                theta_new = self.make_random_step_exp(theta, self.gaussian_step)
            elif self.rdm_walk == 'log':
                theta_new = self.make_random_step_log(theta, self.gaussian_step)

            if not self.prior(theta_new):
                continue

            chi2_new, sz_new, errorsz_new = self.compute_chi2(theta_new)
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

                if self.savefile != None:
                    data = np.asarray([theta[0], theta[1], chi2_current, sz_current[0],sz_current[1], errorsz_current[0], errorsz_current[1]])
                    data = np.reshape(data, (1, 7))
                    np.savetxt(self.savefile, data, delimiter=',')

            if self.stopping_condition == True:
                if self.check_if_stop(self.fit_vector, sz_current, errorsz_current):
                    break

        self.chain_list = [theta_save, chi2_save, sz_save, errorsz_save]
        self.chi2_mini, self.best_param = self.get_best_param()  # to save the best params
        return theta_save, chi2_save, sz_save, errorsz_save


    def prior(self,theta):
        if -8.0 < theta[0] < -1.0 and 0 < theta[1] < 0.5:
            return True
        else:
            return False


    def make_random_step_gaussian(self,theta, sigma_step):
        return theta + sigma_step * np.random.randn(2)

    def make_random_step_log(self,theta, sigma_step):
        s = theta[1]
        s = np.log10(s) + sigma_step[1] * np.random.randn()
        s = 10.0**s
        return [theta[0] + sigma_step[0] * np.random.randn(), s]


    def make_random_step_exp(self,theta, sigma_step):
        sign = np.random.random()
        print sign
        if sign > 0.5:
            print "step proposed : ", np.asarray(theta) - [theta[0] + sigma_step[0] * np.random.randn(),theta[1] + np.random.exponential(scale=sigma_step[1])]
            return [theta[0] + sigma_step[0] * np.random.randn(), theta[1] + np.random.exponential(scale=sigma_step[1])]
        else:
            print "step proposed : ",np.asarray(theta) - [theta[0] + sigma_step[0] * np.random.randn(), theta[1] - np.random.exponential(scale=sigma_step[1])]
            return [theta[0] + sigma_step[0] * np.random.randn(), theta[1] - np.random.exponential(scale=sigma_step[1])]


    def check_if_stop(self,fitvector, sz, sz_error):
        if self.hundred_last != 100 :
            self.hundred_last -= 1# check if we already reached the condition once
            print "I have already matched the stopping condition, I will do %i more steps." %hundred_last

        elif np.abs(fitvector[0] - sz[0]) < 0.75*sz_error[0] and np.abs(fitvector[1] - sz[1]) < 0.75*sz_error[1]:
            self.hundred_last -= 1
            print "I'm matching the stopping condition at this iteration, I will do %i more steps."%hundred_last
        else :
            print "Stopping condition not reached."

        if self.hundred_last == 0 :
            return True
        else :
            return False

    def get_best_param(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else:
            print np.shape(self.chain_list),self.chain_list[1][:]
            ind_min = np.argmin(self.chain_list[1][:])
            self.mean_mini = (self.chain_list[2][ind_min])
            self.sigma_mini = (self.chain_list[3][ind_min])
            self.chi2_mini = (self.chain_list[1][ind_min])
            return self.chi2_mini, self.chain_list[0][ind_min]

    def analyse_plot_results(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            print "Best position : ", self.get_best_param()[1]
            print "Corresponding Chi2 : ", self.get_best_param()[0]
            self.rel_error_mini =  np.abs(np.asarray(self.mean_mini) - np.asarray(self.fit_vector)) / np.asarray(self.sigma_mini)

            print "Target sigma, zruns : " + str(self.fit_vector[1]) + ', ' + str(self.fit_vector[0])
            print "Minimum sigma, zruns : " + str(self.mean_mini[1]) + ', ' + str(self.mean_mini[0])
            print "Minimum chi2 : ", self.chi2_mini
            print "For minimum Chi2, we are standing at " + str(self.rel_error_mini[0]) + " sigma [zruns]"
            print "For minimum Chi2, we are standing at " + str(self.rel_error_mini[1])+ " sigma [sigma]"

            self.success = self.check_success()

            toplot = np.asarray(self.chain_list[0])
            toplot[:,1] = np.log10(toplot[:,1])
            fig1,fig2,fig3 = pltfct.plot_chain_MCMC(toplot, self.chain_list[1], ["$beta$", "log $\sigma$"])
            fig1.savefig(self.savedirectory +self.tweakml_name +  "_MCMC_corner_plot_" + self.lc.object + ".png")
            fig2.savefig(self.savedirectory +self.tweakml_name + "_MCMC_chi2_" + self.lc.object + ".png")
            fig3.savefig(self.savedirectory +self.tweakml_name +"_MCMC_chain_" + self.lc.object + ".png")

            if self.display:
                plt.show()

    def dump_results(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            pickle.dump(self.chain_list, open(self.savedirectory +  self.tweakml_name + "_chain_list_MCMC_" + "_i"
                                         + str(self.n_iter) + "_" + self.lc.object + ".pkl", "wb"))
            pickle.dump(self, open(self.savedirectory + self.tweakml_name + "_MCMC_opt_" + "_i"
                                      + str(self.n_iter) + "_" + self.lc.object + ".pkl", "wb"))

