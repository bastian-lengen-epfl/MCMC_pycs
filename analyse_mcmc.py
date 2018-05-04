import pickle
import corner
import matplotlib.pyplot as plt
import numpy as np
import pycs
import mcmc_function as fmcmc


makeplot = True
display = True
measure_posterior = True
source ="pickle"
object = "HE0435"

picklepath = "./"+object+"/save/"
sim_path = "./"+object+"/simulation_exp/"
plot_path = sim_path + "figure/"

kntstp = 40
ml_kntstep =360
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
niter = 10000
burntime = 1000

rdm_walk = 'exp'
nlcs = [0] #curve to process, can be a list of indices

for i in nlcs :
    recompute_sz = False
    if source == "pickle":
        theta = pickle.load(open(sim_path + "theta_walk_"+ object + "_" + picklename[:-4] + "_" + str(niter) + "_"+rdm_walk+"_"+str(i)+".pkl", "rb"))
        chi2 = pickle.load(open(sim_path + "chi2_walk_"+ object + "_" + picklename[:-4] + "_" + str(niter) + "_"+rdm_walk+"_"+str(i)+".pkl", "rb"))
        try :
            sz = pickle.load(open(sim_path + "sz_walk_" + object + "_" + picklename[:-4] + "_" + str(
                niter) + "_" + rdm_walk + "_" + str(i) + ".pkl", "rb"))
            errorsz = pickle.load(open(sim_path + "errorsz_walk_" + object + "_" + picklename[:-4] + "_" + str(
                niter) + "_" + rdm_walk + "_" + str(i) + ".pkl", "rb"))\

        except :
            print "You didn't save the sigma, zruns for this chain."
            recompute_sz = True

    if source == "rt_file":
        rt_filename = sim_path + 'rt_file_' + object +"_"+ picklename[:-4]  + "_" + str(niter)+"_"+rdm_walk +"_"+str(i)+'.txt'
        vec = np.loadtxt(rt_filename, delimiter=',')
        vec = np.asarray(vec)
        theta = vec[burntime:,0:2]
        chi2 = vec[burntime:,2]
        try :
            sz = vec[burntime:,3:5]
            errorsz = vec[burntime:,5:7]
        except :
            print "You didn't save the sigma, zruns for this chain."
            recompute_sz = True

    theta = np.asarray(theta)
    chi2 = np.asarray(chi2)
    sz = np.asarray(sz)
    errorsz=np.asarray(errorsz)

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
    if recompute_sz :
        mean_mini,sigma_mini = fmcmc.make_mocks_para(min_theta,lcs,spline,n_curve_stat=64, recompute_spline= True, knotstep=kntstp, nlcs=i, verbose=True)
    else :
        mean_mini = sz[N_min,:]
        sigma_mini = errorsz[N_min,:]

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

    if measure_posterior:
        disperions_sig = np.std()





