import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import pickle as pkl
from module.optimisation import Optimiser as mcmc

object = "HE0435"
sim_path = "./"+object+"/simulation_log2_multi2000/"
picklename ="colored_noise_MCMC_opt_i2000.pkl"

opt = pkl.load(open(sim_path + picklename, 'r'))

opt.analyse_plot_results()
opt.reset_report()
opt.report()
