import pickle
import matplotlib.pyplot as plt
import numpy as np
import pycs
import plot_functions as pltfct
import os


makeplot = True
display = True
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
n_iterations = 5
n_particles = 5

param_list = ['beta','sigma']

nlcs = [3] #curve to process, can be a list of indices

for i in nlcs :
    if source == "pickle":

        chain = pickle.load(open(sim_path+"chain_PSO_" + object +"_"+ picklename[:-4] + "_i"
                             + str(n_iterations)+"_p"+str(n_particles)+ "_" +str(i)+".pkl"))

    elif source == "rt_file":
        print"hello"
        pass


    print chain[0]
    f, axes = pltfct.plot_chain(chain, param_list)

    if display:
        plt.show()

    else :
        f.savefig(plot_path + "PSO_chain_" + str(nlcs) + ".png")




