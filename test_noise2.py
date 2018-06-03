from module import tweakml_PS_from_data as twk
from module.optimisation import grid_search_PS as grid
import pycs
import numpy as np

source ="pickle"
object = "HE0435"

picklepath = "./"+object+"/save/"

kntstp = 40
ml_kntstep =360
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
curve = 3

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

A = 1.0 #this is the maximum frequency you want to add (in unit of the Nymquist frequency of your signal)
B = 1.0 #this is the scaling of the power spectrum

#EXEMPLE :
# lcs[curve].ml.spline.display()
# print "Nb coefficient before tweak :", len(lcs[curve].ml.spline.c)
# pycs.sim.twk.tweakml([lcs[curve]], beta=-1.0, sigma=0.05, fmin=1.0/50.0, fmax=0.2, psplot=False)
# lcs[curve].ml.spline.display()
# print "Nb coefficient after tweak :", len(lcs[curve].ml.spline.c)
# exit()

#To check what does the microlensing curve looks like
rls = pycs.gen.stat.subtract([lcs[curve]], spline)
# lcs[curve].ml.spline.display()
# print "Nb coefficient before tweak :", len(lcs[curve].ml.spline.c)
# print "before tweak :", pycs.gen.stat.resistats(rls[0])
# twk.tweakml_PS([lcs[curve]],spline, B,f_min = 1/300.0,save_figure_folder=None,  psplot=True, verbose =True, interpolation = 'linear')
# lcs[curve].ml.spline.display()
# print "Nb coefficient after tweak :", len(lcs[curve].ml.spline.c)


target = pycs.gen.stat.mapresistats(rls)[0]
B_vec = np.linspace(0.5,3.0,30)
pycs.sim.draw.saveresiduals(lcs, spline)
_ , B_min, _, _ , _ = grid.grid_search_PS(lcs[curve],spline,B_vec, target,  max_core=8, n_curve_stat=32, verbose = True, shotnoise = None, knotstep= kntstp)
