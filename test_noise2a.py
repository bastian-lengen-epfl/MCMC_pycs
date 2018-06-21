from module import tweakml_PS_from_data as twk
import pycs

source ="pickle"
object = "UM673_Euler"

picklepath = "./"+object+"/save/"

kntstp = 60
# kntstp = 40
ml_kntstep =500
# ml_kntstep =360
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
curve =0

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

A = 1.0 #this is the maximum frequency you want to add (in unit of the Nymquist frequency of your signal)
B = 1.0 #this is the scaling of the power spectrum

#EXEMPLE :
# lcs[curve].ml.spline.display()
# print "Nb coefficient before tweak :", len(lcs[curve].ml.spline.c)
# pycs.sim.twk.tweakml([lcs[curve]], beta=-1.0, sigma=1.0, fmin=1.0/50.0, fmax=0.2, psplot=False)
# lcs[curve].ml.spline.display()
# print "Nb coefficient after tweak :", len(lcs[curve].ml.spline.c)
#
#
# #To check what does the microlensing curve looks like
rls = pycs.gen.stat.subtract([lcs[curve]], spline)
lcs[curve].ml.spline.display()
print "Nb coefficient before tweak :", len(lcs[curve].ml.spline.c)
print "before tweak :", pycs.gen.stat.resistats(rls[0])
twk.tweakml_PS([lcs[curve]],spline, B,f_min = 1/300.0,save_figure_folder='./',  psplot=True, verbose =True, interpolation = 'linear')
lcs[curve].ml.spline.display()
print "Nb coefficient after tweak :", len(lcs[curve].ml.spline.c)