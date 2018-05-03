import pickle
import corner
import matplotlib.pyplot as plt
import numpy as np
import pycs
import mcmc_function as fmcmc


makeplot = True
source ="pickle"
object = "HE0435"

picklepath = "./"+object+"/save/"
sim_path = "./"+object+"/simulation_log/"
plot_path = sim_path + "figure/"
display = True
kntstp = 40
ml_kntstep =360
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
burntime = 0
niter = 10
rdm_walk = 'log'
nlcs = [0] #curve to process, can be a list of indices

for i in nlcs :
    if source == "pickle":
        theta = pickle.load(open(sim_path + "theta_walk_"+ object + "_" + picklename[:-4] + "_" + str(niter) + "_"+rdm_walk+"_"+str(i)+".pkl", "rb"))
        chi2 = pickle.load(open(sim_path + "chi2_walk_"+ object + "_" + picklename[:-4] + "_" + str(niter) + "_"+rdm_walk+"_"+str(i)+".pkl", "rb"))

    if source == "rt_file":
        rt_filename = sim_path + 'rt_file_' + object +"_"+ picklename[:-4]  + "_" + str(niter)+"_"+rdm_walk +"_"+str(i)+'.txt'
        vec = np.loadtxt(rt_filename, delimiter=',')
        vec = np.asarray(vec)
        theta = vec[burntime:,0:2]
        chi2 = vec[burntime:,2]

    theta = np.asarray(theta)
    chi2 = np.asarray(chi2)

    (lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

    #Control the residuals :
    rls = pycs.gen.stat.subtract(lcs, spline)
    pycs.sim.draw.saveresiduals(lcs, spline)
    print 'Curve ', i
    print 'Residuals from the fit : '
    print pycs.gen.stat.mapresistats(rls)[i]
    fit_sigma = pycs.gen.stat.mapresistats(rls)[i]["std"]
    fit_zruns = pycs.gen.stat.mapresistats(rls)[i]["zruns"]
    fit_nruns = pycs.gen.stat.mapresistats(rls)[i]["nruns"]
    fit_vector = [fit_zruns,fit_sigma]

    min_chi2 = np.min(chi2)
    N_min = np.argmin(chi2)
    min_theta = theta[N_min,:]

    print "min Chi2 : ", min_chi2
    print "min theta :", min_theta
    mean_mini,sigma_mini = fmcmc.make_mocks_para(min_theta,lcs,spline,n_curve_stat=64, recompute_spline= True, knotstep=kntstp, nlcs=i, verbose=True)
    print "compared to sigma, nruns, zruns : "+ str(fit_sigma) + ', ' + str(fit_nruns) + ', ' + str(fit_zruns)
    print "For minimum Chi2, we are standing at " + str(np.abs(mean_mini[0]-fit_zruns)/sigma_mini[0]) + " sigma [zruns]"
    print "For minimum Chi2, we are standing at " + str(np.abs(mean_mini[1]-fit_sigma)/sigma_mini[1]) + " sigma [sigma]"

    if makeplot :
        theta[:,1] = np.log10(theta[:,1])
        fig1 = corner.corner(theta, labels=["$beta$", "log $\sigma$"])

        fig2 = plt.figure(2)
        x = np.arange(len(chi2))
        plt.xlabel('N', fontdict={"fontsize" : 16})
        plt.ylabel('$\chi^2$', fontdict={"fontsize" : 16})
        plt.plot(x,chi2)


        fig3, axe = plt.subplots(2,1,sharex=True)
        axe[0].plot(x,theta[:,0],'r')
        axe[1].plot(x,theta[:,1],'g')
        plt.xlabel('N', fontdict={"fontsize" : 16})
        axe[0].set_ylabel('beta', fontdict={"fontsize" : 16})
        axe[1].set_ylabel('$log \sigma$', fontdict={"fontsize" : 16})

        fig1.savefig(plot_path +"cornerplot_" +object+ "_" + picklename[:-4] + "_" + str(niter) + "_"+rdm_walk+"_"+str(i)+'.png')
        fig2.savefig(plot_path + 'chi2-random_'+ object+ "_" + picklename[:-4] + "_" + str(niter) +"_"+rdm_walk+"_"+str(i)+ '.png')
        fig3.savefig(plot_path +'beta-sigma-random_'+ object+ "_" + picklename[:-4] + "_" + str(niter) + "_"+rdm_walk+"_"+str(i)+'.png')

        if display :
            plt.show()

