import sys
sys.path.append("..")
from module import tweakml_PS_from_data as twk
import pycs

source ="pickle"
object = "WFI2033"
dataname="WFI"

kntstp = 35
ml_kntstep =50

picklepath = "../Simulation/" + object +'_'+dataname + '/' + 'spl1_ks%i_splml_ksml_%i/'%(kntstp, ml_kntstep)


picklename ="initopt_%s_ks%i_ksml%i.pkl"%(dataname, kntstp, ml_kntstep)
curve =0

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

A = 1.0 #this is the maximum frequency you want to add (in unit of the Nymquist frequency of your signal)
B = [0.1] #this is the scaling of the power spectrum


# #To check what does the microlensing curve looks like
rls = pycs.gen.stat.subtract(lcs, spline)
# lcs[0].ml.spline.display()
# lcs[1].ml.spline.display()
# lcs[2].ml.spline.display()
print "Nb coefficient before tweak :", len(lcs[curve].ml.spline.c)
print "before tweak :", pycs.gen.stat.resistats(rls[0])
twk.tweakml_PS(lcs[curve], spline, B,f_min = 1/300.0,save_figure_folder='./',  psplot=True, verbose =True, interpolation = 'linear')
# lcs[0].ml.spline.display()
# lcs[1].ml.spline.display()
# lcs[2].ml.spline.display()
print "Nb coefficient after tweak :", len(lcs[curve].ml.spline.c)