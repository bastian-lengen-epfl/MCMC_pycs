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
import matplotlib.pyplot as plt
import pickle
# import dill #this is important for multiprocessing

class Optimiser(object):
    def __init__(self, lc, fit_vector, spline, knotstep=None,
                     savedirectory="./", recompute_spline=True, max_core = 16, theta_init = [-2.0 ,0.1],
                    n_curve_stat = 32, shotnoise = "magerrs", tweakml_type = 'colored_noise', display = False, verbose = False,
                 tweakml_name = ''):

        self.lc = lc
        self.fit_vector = fit_vector
        self.spline = spline
        self.theta_init = theta_init
        self.knotstep = knotstep
        self.savedirectory = savedirectory
        self.recompute_spline = recompute_spline
        self.sucess = False
        self.mean_mini = None #mean of zruns and sigma computed with the best parameters
        self.sigma_mini = None #std of zruns and sigma computed with the best parameters
        self.chi2_mini = None
        self.rel_error_mini= None #relative error in term of zruns and sigmas
        self.best_param = None

        if recompute_spline == True and knotstep == None :
            print "Error : I can't recompute spline if you don't give me the knotstep ! "
            exit()

        if max_core != None :
            self.max_core = max_core
        else :
            self.max_core = multiprocess.cpu_count()
            print "You will run on %i cores."%self.max_core

        if self.max_core > 1 :
            self.para = True
        else :
            self.para = False
            print "I won't compute the mockcurve in parallel."

        self.n_curve_stat = n_curve_stat
        self.shotnoise = shotnoise
        self.tweakml_type =tweakml_type
        self.tweakml_name = tweakml_name
        self.display = display
        self.verbose = verbose

    def make_mocks_para(self, theta):
        stat = []
        zruns = []
        sigmas = []
        nruns = []

        pool = multiprocess.Pool(processes=self.max_core)

        job_args = [(theta) for j in range(self.n_curve_stat)]

        stat_out = pool.map(self.fct_para, job_args)
        pool.close()
        pool.join()

        for i in range(len(stat_out)):
            zruns.append(stat_out[i][0]['zruns'])
            sigmas.append(stat_out[i][0]['std'])
            nruns.append(stat_out[i][0]['nruns'])

        if self.verbose:
            print 'Mean zruns (simu): ', np.mean(zruns), '+/-', np.std(zruns)
            print 'Mean sigmas (simu): ', np.mean(sigmas), '+/-', np.std(sigmas)
            print 'Mean nruns (simu): ', np.mean(nruns), '+/-', np.std(nruns)

        return [np.mean(zruns), np.mean(sigmas)], [np.std(zruns), np.std(sigmas)]

    def compute_chi2(self, theta):
        #theta : proposed step
        #fit vector : target vector to fit in terms of [zruns, sigma]

        chi2 = 0.0
        if self.n_curve_stat == 1:
            print "Warning : I cannot compute statistics with one single curves !!"

        if self.para:
            out, error = self.make_mocks_para(theta)
        else:
            out, error = self.make_mocks(theta)

        # for i in range(len(out)):
        #     chi2 += (fit_vector[i] - out[i]) ** 2 / error[i] ** 2

        chi2 = (self.fit_vector[0] - out[0]) ** 2 / error[0] ** 2
        chi2 += (self.fit_vector[1] - out[1]) ** 2 / (2 * error[1] ** 2)

        return chi2, out, error

    def fct_para(self, theta):
        if self.tweakml_type == 'colored_noise':
            mocklc = pycs.sim.draw.draw([self.lc], self.spline, tweakml=lambda x: pycs.sim.twk.tweakml(x, beta=theta[0],
                                                                                             sigma=theta[1],
                                                                                             fmin=1 / 300.0,
                                                                                             fmax=None,
                                                                                             psplot=False),
                                        shotnoise=self.shotnoise, keeptweakedml=False)

        elif self.tweakml_type == 'PS_from_residuals':
            mocklc = pycs.sim.draw.draw([self.lc], self.spline,
                                        tweakml=lambda x: twk.tweakml_PS(x, self.spline, theta[0], f_min=1 / 300.0,
                                                                         psplot=False, save_figure_folder=None,
                                                                         verbose=self.verbose,
                                                                         interpolation='linear')
                                        , shotnoise=self.shotnoise, keeptweakedml=False)

        if self.recompute_spline:
            if self.knotstep == None:
                print "Error : you must give a knotstep to recompute the spline"
            spline_on_mock = pycs.spl.topopt.opt_fine(mocklc, nit=5, knotstep=self.knotstep, verbose=False)
            mockrls = pycs.gen.stat.subtract(mocklc, spline_on_mock)
        else:
            mockrls = pycs.gen.stat.subtract(mocklc, self.spline)

        stat = pycs.gen.stat.mapresistats(mockrls)
        return stat

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
            if self.tweakml_type == 'colored_noise':
                mocklc.append(
                    pycs.sim.draw.draw([self.lc], self.spline, tweakml=lambda x: pycs.sim.twk.tweakml(x, beta=theta[0],
                                                                                            sigma=theta[1],
                                                                                            fmin=1 / 300.0,
                                                                                            fmax=None,
                                                                                            psplot=False),
                                       shotnoise=self.shotnoise, keeptweakedml=False))

            elif self.tweakml_type == 'PS_from_residuals':
                mocklc.append(pycs.sim.draw.draw([self.lc], self.spline, tweakml=lambda x: twk.tweakml_PS(x, self.spline, theta[0],
                                                                                                f_min=1 / 300.0,
                                                                                                psplot=False,
                                                                                                save_figure_folder=None,
                                                                                                verbose=self.verbose,
                                                                                                interpolation='linear')
                                                 , shotnoise=self.shotnoise, keeptweakedml=False))

            if self.recompute_spline:
                if self.knotstep == None:
                    print "Error : you must give a knotstep to recompute the spline"
                spline_on_mock = pycs.spl.topopt.opt_fine(mocklc[i], nit=5, knotstep=self.knotstep, verbose=self.verbose)
                mockrls.append(pycs.gen.stat.subtract(mocklc[i], spline_on_mock))
            else:
                mockrls.append(pycs.gen.stat.subtract(mocklc[i], self.spline))


            if self.recompute_spline and self.display:
                    pycs.gen.lc.display([self.lc], [spline_on_mock], showdelays=True)
                    pycs.gen.stat.plotresiduals([mockrls[i]])

            stat.append(pycs.gen.stat.mapresistats(mockrls[i]))
            zruns.append(stat[i][0]['zruns'])
            sigmas.append(stat[i][0]['std'])
            nruns.append(stat[i][0]['nruns'])

        if self.verbose:
            print 'Mean zruns (simu): ', np.mean(zruns), '+/-', np.std(zruns)
            print 'Mean sigmas (simu): ', np.mean(sigmas), '+/-', np.std(sigmas)
            print 'Mean nruns (simu): ', np.mean(nruns), '+/-', np.std(nruns)

        return [np.mean(zruns), np.mean(sigmas)], [np.std(zruns), np.std(sigmas)]

    def check_success(self):
        if self.rel_error_mini == None :
            print "Error you should run analyse_plot_results() first !"
            exit()
        else :
            if self.rel_error_mini[0] <0.5 and self.rel_error_mini[1] < 0.5 :
                return True
            else :
                return False

    def report(self):
        #TODO : finish this
        if self.best_param == None :
            print "Error : you should run optimise() first !"
            exit()

        if os.path.isfile(self.savedirectory + 'report_tweakml_optimisation.txt') :
            f = open(self.savedirectory + 'report_tweakml_optimisation.txt', 'a')
            f.write('Best parameters for %s : \n'%self.tweakml_name)
            f.write('------------------------------------------------\n')
        else :
            f = open(self.savedirectory + 'report_tweakml_optimisation.txt', 'a')

        f.write('Lightcurves %s : \n'%self.lc.object)
        f.write('\n')
        if self.success == True:
            f.write('I succeeded in finding a set of parameters that match the statistical properties of the real lightcurve within 0.5sigma. \n')

        else :
            f.write('I did not succeed in finding a set of parameters that match the statistical properties of the real lightcurve within 0.5sigma. \n')

        f.write('Best parameters are : %s \n'%str(self.best_param) )
        f.write("Corresponding Chi2 : %2.2f \n"%self.chi2_mini)
        f.write("Target zruns, sigma : %2.2f, %2.2f \n"%(self.fit_vector[0],self.fit_vector[1]))
        f.write("At minimum zruns, sigma : %2.2f, %2.2f \n"%(self.mean_mini[0], self.mean_mini[1]))
        f.write("For minimum Chi2, we are standing at " + str(self.rel_error_mini[0]) + " sigma [zruns] \n")
        f.write("For minimum Chi2, we are standing at " + str(self.rel_error_mini[1])+ " sigma [sigma] \n")
        f.write('------------------------------------------------\n')
        f.close()

    def reset_report(self):
        open(self.savedirectory + 'report_tweakml_optimisation.txt', 'w').close()





class Metropolis_Hasting_Optimiser(Optimiser):
    def __init__(self, lc, fit_vector, spline, knotstep=None, niter=1000,
                    burntime=100, savedirectory="./", recompute_spline=True, rdm_walk='gaussian', max_core = 16,
                    n_curve_stat = 32, stopping_condition = True, shotnoise = "magerrs", theta_init = [-2.0, 0.2], gaussian_step = [0.1, 0.01],
                 tweakml_type = 'coloired_noise' ,tweakml_name = ''):

        Optimiser.__init__(self,lc, fit_vector,spline, knotstep = knotstep, savedirectory= savedirectory, recompute_spline=recompute_spline,
                                   max_core =max_core, n_curve_stat = n_curve_stat, shotnoise = shotnoise, theta_init= theta_init,
                           tweakml_type= tweakml_type, tweakml_name= tweakml_name)
        self.niter = niter
        self.burntime = burntime
        self.rdm_walk = rdm_walk
        self.stopping_condition = stopping_condition
        self.gaussian_step = gaussian_step
        self.savefile = self.savedirectory + self.tweakml_name + '_MCMC_outfile_i' + str(niter)+"_"+rdm_walk +"_"+self.lc.object+'.txt'
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

        for i in range(self.niter):
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
            self.rel_error_mini =  np.abs(self.mean_mini - self.fit_vector) / self.sigma_mini

            print "Target sigma, zruns : " + str(self.fit_vector[1]) + ', ' + str(self.fit_vector[0])
            print "Minimum sigma, zruns : " + str(self.mean_mini[1]) + ', ' + str(self.mean_mini[0])
            print "Minimum chi2 : ", self.chi2_mini
            print "For minimum Chi2, we are standing at " + str(self.rel_error_mini[0]) + " sigma [zruns]"
            print "For minimum Chi2, we are standing at " + str(self.rel_error_mini[1])+ " sigma [sigma]"

            self.sucess = self.check_success()

            toplot = self.chain_list[0,:]
            toplot[1,:] = np.log10(toplot[1,:])
            fig1,fig2,fig3 = pltfct.plot_chain_MCMC(toplot, self.chain_list[1][:], ["$beta$", "log $\sigma$"])
            fig1.savefig(self.savedirectory +self.tweakml_name +  "_MCMC_corner_plot_" + self.lc.object + ".png")
            fig2.savefig(self.savedirectory +self.tweakml_name + "_MCMC_chi2_" + self.lc.object + ".png")
            fig3.savefig(self.savedirectory +self.tweakml_name +"_MCMC_chain_" + self.lc.object + ".png")

            if self.display:
                plt.show()

    def dump_results(self):
        if self.chain == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            pickle.dump(self.chain_list, open(self.savedirectory +  self.tweakml_name + "_chain_list_MCMC_" + "_i"
                                         + str(self.n_iter) + "_" + self.lc.object + ".pkl", "wb"))
            pickle.dump(self, open(self.savedirectory + self.tweakml_name + "_MCMC_opt_" + "_i"
                                      + str(self.n_iter) + "_" + self.lc.object + ".pkl", "wb"))


class PSO_Optimiser(Optimiser) :
    #Attention here : You cannot use the parrallel computing of the mock curves because PSO, already launch the particles on several thread !

    def __init__(self, lc, fit_vector, spline, savedirectory ="./", knotstep = None, max_core = 8, shotnoise = 'magerrs',
                 recompute_spline = True, n_curve_stat= 32, theta_init = None, n_particles = 30, n_iter = 50,
                 lower_limit = [-8., 0.], upper_limit = [-1.0, 0.5], mpi = False, tweakml_type = 'colored_noise', tweakml_name = ''):

        Optimiser.__init__(self,lc, fit_vector,spline, knotstep = knotstep, savedirectory= savedirectory, recompute_spline=recompute_spline,
                                   max_core =1, n_curve_stat = n_curve_stat, shotnoise = shotnoise, theta_init= theta_init,
                           tweakml_type = tweakml_type, tweakml_name = tweakml_name, verbose= False, display= False)


        self.n_particles = n_particles
        self.n_iter = n_iter
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.mpi = mpi
        self.max_thread = max_core * 2
        self.chain_list = None
        self.savefile = self.savedirectory + self.tweakml_name + '_PSO_file _' + "_i" + str(self.n_iter)+"_p"+str(self.n_particles)+ "_" +self.lc.object+".txt"

    def __call__(self, theta):
        return self.likelihood(theta)

    def likelihood(self, theta):
        chi2, out ,error  = self.compute_chi2(theta)
        return [-0.5*chi2]

    def optimise(self):

        if self.mpi is True:
            pso = MpiParticleSwarmOptimizer(self, self.lower_limit, self.upper_limit, self.n_particles, threads=self.max_thread)
        else:
            pso = ParticleSwarmOptimizer(self, self.lower_limit, self.upper_limit, self.n_particles, threads=self.max_thread)

        X2_list = []
        vel_list = []
        pos_list = []
        num_iter = 0

        f = open(self.savefile, "wb")

        for swarm in pso.sample(self.n_iter):
            print "iteration : ", num_iter
            X2_list.append(pso.gbest.fitness * 2)
            vel_list.append(pso.gbest.velocity)
            pos_list.append(pso.gbest.position)
            data = np.asarray([X2_list[-1], vel_list[-1][0], vel_list[-1][1], pos_list[-1][0], pos_list[-1][1]])
            data = np.reshape(data, (1, 5))
            np.savetxt(f, data, delimiter=',')
            num_iter += 1

        self.chain_list = [X2_list, pos_list, vel_list]
        self.chi2_mini, self.best_param = self.get_best_param()
        return self.chain_list

    def get_best_param(self):
        if self.chain == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            best_chi2 = self.chain_list[-1:0]
            best_param = self.chain_list[-1:1]
            return best_chi2,best_param

    def dump_results(self):
        if self.chain == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            pickle.dump(self.chain_list, open(self.savedirectory + self.tweakml_name +"_chain_list_PSO_" + "_i"
                                         + str(self.n_iter) + "_p" + str(self.n_particles) + "_" + self.lc.object + ".pkl", "wb"))
            pickle.dump(self, open(self.savedirectory + self.tweakml_name +"_PSO_opt_" + "_i"
                                      + str(self.n_iter) + "_p" + str(self.n_particles) + "_" + self.lc.object + ".pkl", "wb"))
    def analyse_plot_results(self):
        if self.chain == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            print "Converged position :" + self.get_best_param()[1]
            print "Converged Chi2 : " + self.get_best_param()[0]
            n_curve_stat_save = copy.deepcopy(self.n_curve_stat)
            self.n_curve_stat = 32
            mean_mini, sigma_mini = self.make_mocks_para(self.get_best_param()[1])
            self.mean_mini = np.asarray(mean_mini)
            self.sigma_mini = np.asarray(sigma_mini)
            self.chi2_mini = np.sum((mean_mini - self.fit_vector) ** 2 / (sigma_mini ** 2))
            self.rel_error_mini =  np.abs(mean_mini - self.fit_vector) / sigma_mini
            self.n_curve_stat = n_curve_stat_save

            print "Target sigma, zruns : " + str(self.fit_vector[1]) + ', ' + str(self.fit_vector[0])
            print "Minimum sigma, zruns : " + str(mean_mini[1]) + ', ' + str(mean_mini[0])
            print "For minimum Chi2, we are standing at " + str(self.rel_error_mini[0]) + " sigma [zruns]"
            print "For minimum Chi2, we are standing at " + str(self.rel_error_mini[1])+ " sigma [sigma]"

            self.sucess = self.check_success()

            param_list = ['beta', 'sigma']
            f, axes = pltfct.plot_chain_PSO(self.chain, param_list)

            f.savefig(self.savedirectory + self.tweakml_name + "_PSO_chain_" + self.lc.object + ".png")

            if self.display:
                plt.show()

class Grid_Optimiser(Optimiser):
    def __init__(self, lc, fit_vector, spline, knotstep=None,
                     savedirectory="./", recompute_spline=True, max_core = 16, theta_init = [-2.0 ,0.1],
                    n_curve_stat = 32, shotnoise = "magerrs", tweakml_type = 'PS_from_residuals', tweakml_name = '',
                 display = False, verbose = False, grid = np.linspace(0.5,2,10)):

        Optimiser.__init__(self,lc, fit_vector,spline, knotstep = knotstep, savedirectory= savedirectory, recompute_spline=recompute_spline,
                                   max_core =max_core, n_curve_stat = n_curve_stat, shotnoise = shotnoise, theta_init= theta_init,
                           tweakml_type= tweakml_type, tweakml_name = tweakml_name, verbose=verbose, display= display)

        self.grid = grid #should be only a 1D array for the moment
        self.chain_list = None

    def optimise(self):

        sigma = []
        zruns = []
        sigma_std = []
        zruns_std = []
        chi2 = []
        zruns_target = self.fit_vector[0]
        sigma_target = self.fit_vector[1]

        for i, B in enumerate(self.grid):
            if self.para :
                [[zruns_c, sigma_c], [zruns_std_c, sigma_std_c]] = self.make_mocks_para(theta=B) #for some reason do not work with multiprocessing !
            else :
                [[zruns_c,sigma_c],[zruns_std_c,sigma_std_c]] = self.make_mocks(theta=[B]) #to debug, remove multiprocessing
            chi2.append(
                (zruns_c - zruns_target) ** 2 / zruns_std_c ** 2 + (sigma_c - sigma_target) ** 2 / sigma_std_c ** 2)

            sigma.append(sigma_c)
            zruns.append(zruns_c)
            sigma_std.append(sigma_std_c)
            zruns_std.append(zruns_std_c)

        min_ind = np.argmin(chi2)
        self.chi2_mini = np.min(chi2)
        self.mean_mini = [zruns[min_ind], sigma[min_ind]]
        self.sigma_mini = [zruns_std[min_ind], sigma_std[min_ind]]
        self.rel_error_mini = [np.abs(zruns[min_ind] - zruns_target) / zruns_std[min_ind],
                               np.abs(sigma[min_ind] - sigma_target) / sigma_std[min_ind]]

        self.chain_list = [self.grid, chi2, [zruns,sigma], [zruns_std,sigma_std]]
        self.chi2_mini, self.best_param = self.get_best_param()


        if self.verbose:
            print "target :", self.fit_vector
            print "Best parameter from grid search :", self.grid[min_ind]
            print "Associated chi2 : ", chi2[min_ind]
            print "Zruns : %2.6f +/- %2.6f (%2.4f sigma from target)" % (
            zruns[min_ind], zruns_std[min_ind], self.rel_error_mini[0])
            print "Sigma : %2.6f +/- %2.6f (%2.4f sigma from target)" % (sigma[min_ind], sigma_std[min_ind], self.rel_error_mini[1])

        if self.rel_error_mini[0] < 0.5 and self.rel_error_mini[1] < 0.5:
            self.success = True
        else:
            self.success = False

        return self.chain_list

    def get_best_param(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else:
            ind_min = np.argmin(self.chain_list[1][:])
            return self.chi2_mini, self.chain_list[0][ind_min]

    def analyse_plot_results(self):
        fig1 = plt.figure(1)
        plt.errorbar(self.grid, self.chain_list[2][0], yerr=self.chain_list[3][0])
        plt.hlines(self.fit_vector[0], self.grid[0], self.grid[-1], colors='r', linestyles='solid', label='target')
        plt.xlabel('B in unit of Nymquist frequency)')
        plt.ylabel('zruns')
        plt.legend()

        fig2 = plt.figure(2)
        plt.errorbar(self.grid, self.chain_list[2][1], yerr=self.chain_list[3][1])
        plt.hlines(self.fit_vector[1], self.grid[0], self.grid[-1], colors='r', linestyles='solid', label='target')
        plt.xlabel('B in unit of Nymquist frequency)')
        plt.ylabel('sigma')
        plt.legend()

        fig3 = plt.figure(3)
        plt.plot(self.grid, self.chain_list[0])
        plt.xlabel('B in unit of Nymquist frequency)')
        plt.ylabel('$\chi^2$')

        fig1.savefig(self.savedirectory + self.tweakml_name + '_zruns_' + self.lc.object + '.png')
        fig2.savefig(self.savedirectory + self.tweakml_name + '_std_' + self.lc.object + '.png')
        fig3.savefig(self.savedirectory + self.tweakml_name + '_chi2_' + self.lc.object + '.png')

        if self.display:
            plt.show()
        plt.gcf().clear()

def get_fit_vector(l,spline):
    rls = pycs.gen.stat.subtract([l], spline)
    print 'Residuals from the fit : '
    print pycs.gen.stat.resistats(rls[0])
    fit_sigma = pycs.gen.stat.resistats(rls[0])["std"]
    fit_zruns = pycs.gen.stat.resistats(rls[0])["zruns"]
    fit_vector = [fit_zruns, fit_sigma]
    return fit_vector
