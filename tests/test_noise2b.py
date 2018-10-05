#This script show how you can use the POwer Spectrum of the residuals to tweak your ml spline.
import sys
sys.path.append("..")
from module import tweakml_PS_from_data as twk
import pycs
import numpy as np
import matplotlib.pyplot as plt

source ="pickle"
object = "WFI2033"
dataname="WFI"

kntstp = 35
ml_kntstep =50

picklepath = "../Simulation/" + object +'_'+dataname + '/' + 'spl1_ks%i_splml_ksml_%i/'%(kntstp, ml_kntstep)
picklename ="initopt_%s_ks%i_ksml%i.pkl"%(dataname, kntstp, ml_kntstep)
curve =1

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

A = 1.0 #this is the maximum frequency you want to add (in unit of the Nymquist frequency of your signal)
B = 1.0 #this is the scaling of the power spectrum

rls = pycs.gen.stat.subtract(lcs, spline)
target = pycs.gen.stat.mapresistats(rls)
zrunA,stdA = target[0]['zruns'], target[0]['std']
zrunB,stdB = target[1]['zruns'], target[1]['std']
zrunC,stdC = target[2]['zruns'], target[2]['std']

paramA = []
paramB = []
paramC = []

ml = []

for l in lcs :
    ml.append(l.ml.spline.copy())

B_vec = np.linspace(0.1,1.0,10)
for B in B_vec :
    #set back the original microelnsing :

    for i,l in enumerate(lcs) :
        l.ml.spline = ml[i]
        # lcs[i].ml.spline.display()

    twk.tweakml_PS([lcs[0]], spline, B,f_min = 1/300.0,save_figure_folder='./',  psplot=False, verbose =True, interpolation = 'linear')
    twk.tweakml_PS([lcs[1]], spline, B,f_min = 1/300.0,save_figure_folder='./',  psplot=False, verbose =True, interpolation = 'linear')
    twk.tweakml_PS([lcs[2]], spline, B,f_min = 1/300.0,save_figure_folder='./',  psplot=False, verbose =True, interpolation = 'linear')

    rls = pycs.gen.stat.subtract(lcs, spline)
    rlsA = pycs.gen.stat.resistats(rls[0])
    rlsB = pycs.gen.stat.resistats(rls[1])
    rlsC = pycs.gen.stat.resistats(rls[2])

    paramA.append([rlsA['zruns'], rlsA['std']])
    paramB.append([rlsB['zruns'], rlsB['std']])
    paramC.append([rlsC['zruns'], rlsC['std']])


paramA = np.asarray(paramA)
paramB = np.asarray(paramB)
paramC = np.asarray(paramC)

plt.figure(1)
plt.title("zruns")
plt.plot(B_vec, paramA[:,0],'r', label = 'Curve A')
plt.plot(B_vec, paramB[:,0],'b', label = 'Curve B')
plt.plot(B_vec, paramC[:,0],'g', label = 'Curve C')
plt.hlines(zrunA, np.min(B_vec), np.max(B_vec), colors='r', linestyles='dashed', label='target')
plt.hlines(zrunB, np.min(B_vec), np.max(B_vec), colors='b', linestyles='dashed', label='target')
plt.hlines(zrunC, np.min(B_vec), np.max(B_vec), colors='g', linestyles='dashed', label='target')
plt.legend()

plt.figure(2)
plt.title("std")
plt.plot(B_vec, paramA[:,1],'r', label = 'Curve A')
plt.plot(B_vec, paramB[:,1],'b', label = 'Curve B')
plt.plot(B_vec, paramC[:,1],'g', label = 'Curve C')
plt.hlines(stdA, np.min(B_vec), np.max(B_vec), colors='r', linestyles='dashed', label='target')
plt.hlines(stdB, np.min(B_vec), np.max(B_vec), colors='b', linestyles='dashed', label='target')
plt.hlines(stdC, np.min(B_vec), np.max(B_vec), colors='g', linestyles='dashed', label='target')
plt.legend()
plt.show()


