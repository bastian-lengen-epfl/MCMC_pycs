import pickle
import matplotlib.pyplot as plt
import numpy as np
import pycs
import plot_functions as pltfct
import mcmc_function as fmcmc
import os


makeplot = True
display = True
recompute_sz = True
source ="pickle"
object = "HE0435"

picklepath = "./"+object+"/save/"
sim_path = "./"+object+"/simulation_PSO/"
plot_path = sim_path + "figure/"

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

kntstp = 40
ml_kntstep =360
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
n_iterations = 200
n_particles = 16

param_list = ['beta','sigma']

nlcs = [3] #curve to process, can be a list of indices

for i in nlcs :
    if source == "pickle":

        chain = pickle.load(open(sim_path+"chain_PSO_" + object +"_"+ picklename[:-4] + "_i"
                             + str(n_iterations)+"_p"+str(n_particles)+ "_" +str(i)+".pkl"))


    elif source == "rt_file":
        print"hello"
        pass

    position = chain[2]
    (lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)
    if recompute_sz :

        rls = pycs.gen.stat.subtract(lcs, spline)
        pycs.sim.draw.saveresiduals(lcs, spline)
        print 'Curve ', i
        print 'Residuals from the fit : '
        print pycs.gen.stat.mapresistats(rls)[i]
        fit_sigma = pycs.gen.stat.mapresistats(rls)[i]["std"]
        fit_zruns = pycs.gen.stat.mapresistats(rls)[i]["zruns"]
        fit_nruns = pycs.gen.stat.mapresistats(rls)[i]["nruns"]
        fit_vector = [fit_zruns,fit_sigma]
        print "Converged position :", position[-1]
        mean_mini,sigma_mini = fmcmc.make_mocks_para(position[-1],lcs,spline,n_curve_stat=64, recompute_spline= True, knotstep=kntstp, nlcs=i, verbose=True)
        print "Target sigma, nruns, zruns : "+ str(fit_sigma) + ', ' + str(fit_nruns) + ', ' + str(fit_zruns)
        print "Minimum sigma, zruns : "+ str(mean_mini[1]) + ', ' + str(mean_mini[0])
        print "For minimum Chi2, we are standing at " + str(np.abs(mean_mini[0]-fit_zruns)/sigma_mini[0]) + " sigma [zruns]"
        print "For minimum Chi2, we are standing at " + str(np.abs(mean_mini[1]-fit_sigma)/sigma_mini[1]) + " sigma [sigma]"


    f, axes = pltfct.plot_chain(chain, param_list)


    f.savefig(plot_path + "PSO_chain_" + str(nlcs) + ".png")

    if display:
        plt.show()



