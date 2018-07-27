import pickle as pkl
from module.optimisation import Optimiser as mcmc

object = "HE0435"
sim_path = "./"+object+"/simulation_log2_multi/"
picklename ="colored_noise_MCMC_i2000.pkl"

opt = pkl.load(sim_path + picklename)

opt.analyse_plot_results()
opt.reset_report()
opt.report()