from module.optimisation import Optimiser
from cosmoHammer import MpiParticleSwarmOptimizer
from cosmoHammer import ParticleSwarmOptimizer
import numpy as np
import pycs
import copy, os

class PSO_Optimiser(Optimiser) :
    #Attention here : You cannot use the parrallel computing of the mock curves because PSO, already launch the particles on several thread !

    def __init__(self, lc, fit_vector, spline, savedirectory ="./", knotstep = None, max_core = 8, shotnoise = 'magerrs',
                 recompute_spline = True, n_curve_stat= 32, theta_init = None, n_particles = 30, n_iter = 50,
                 lower_limit = [-8., 0.], upper_limit = [-1.0, 0.5], mpi = False, tweakml_type = 'colored_noise', tweakml_name = '',
                 correction_PS_residuals = True):

        Optimiser.__init__(self,lc, fit_vector,spline, knotstep = knotstep, savedirectory= savedirectory, recompute_spline=recompute_spline,
                                   max_core =1, n_curve_stat = n_curve_stat, shotnoise = shotnoise, theta_init= theta_init,
                           tweakml_type = tweakml_type, tweakml_name = tweakml_name, correction_PS_residuals= correction_PS_residuals,
                           verbose= False, display= False)


        self.n_particles = n_particles
        self.n_iter = n_iter
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.mpi = mpi
        self.max_thread = max_core * 2
        self.chain_list = None
        self.savefile = self.savedirectory + self.tweakml_name + '_PSO_file' + "_i" + str(self.n_iter)+"_p"+str(self.n_particles)+ "_" +self.lc.object+".txt"

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
            X2_list.append(pso.gbest.fitness * -2.0)
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
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            best_chi2 = self.chain_list[0][-1]
            best_param = self.chain_list[1][-1][:]
            return best_chi2,best_param

    def dump_results(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            pickle.dump(self.chain_list, open(self.savedirectory + self.tweakml_name +"_chain_list_PSO_" + "_i"
                                         + str(self.n_iter) + "_p" + str(self.n_particles) + "_" + self.lc.object + ".pkl", "wb"))
            pickle.dump(self, open(self.savedirectory + self.tweakml_name +"_PSO_opt_" + "_i"
                                      + str(self.n_iter) + "_p" + str(self.n_particles) + "_" + self.lc.object + ".pkl", "wb"))
    def analyse_plot_results(self):
        if self.chain_list == None :
            print "Error you should run optimise() first !"
            exit()
        else :
            print "Converged position :", self.best_param
            print "Converged Chi2 : ", self.chi2_mini
            mean_mini, sigma_mini = self.make_mocks(self.best_param) #TODO : set back to para here when it is repaired
            self.mean_mini = np.asarray(mean_mini)
            self.sigma_mini = np.asarray(sigma_mini)
            self.rel_error_mini = np.abs(np.asarray(self.mean_mini) - np.asarray(self.fit_vector)) / np.asarray(
                self.sigma_mini)

            print "Target sigma, zruns : " + str(self.fit_vector[1]) + ', ' + str(self.fit_vector[0])
            print "Minimum sigma, zruns : " + str(mean_mini[1]) + ', ' + str(mean_mini[0])
            print "For minimum Chi2, we are standing at " + str(self.rel_error_mini[0]) + " sigma [zruns]"
            print "For minimum Chi2, we are standing at " + str(self.rel_error_mini[1])+ " sigma [sigma]"

            self.success = self.check_success()

            param_list = ['beta', 'sigma']
            f, axes = pltfct.plot_chain_PSO(self.chain_list, param_list)

            f.savefig(self.savedirectory + self.tweakml_name + "_PSO_chain_" + self.lc.object + ".png")

            if self.display:
                plt.show()