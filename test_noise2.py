import tweakml_PS_from_data as twk
import pycs

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


twk.tweakml_PS(lcs,A,B, psplot=False, sampling=0.1, verbose = True)
