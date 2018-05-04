#!/urs/bin/env python

import pycs
import os,sys, glob
from optparse import OptionParser
from multiprocessing import Pool, Lock, cpu_count
import numpy as np
import time


execfile("config.py")


def parse_options():

    usage = "usage: %prog [options] file"
    parser = OptionParser(usage=usage)

    parser.add_option("--draw",
        action="store_true",
        dest="draw",
        default = False,
        help="draw simulated curves")

    parser.add_option("--sim",
        action="store_true",
        dest="sim",
        default = False,
        help="compute td and tderr on the simulations")

    parser.add_option("--display",
        action="store_true",
        dest="display",
        default = False,
        help="plot results")

    (options, args) = parser.parse_args()
    files = args
    return files, options

files, opt = parse_options()


#####

if opt.draw is False and opt.sim is False and opt.display is False:

    print "Hey, give me an option to execute ! (--draw, --sim or --display)"
    sys.exit()

##### Here, we start by drawing a set of mock curves. This is done by using the results of the spline optimizer

if opt.draw:
    os.chdir(lens_directory+combkw)
    lcs, spline = pycs.gen.util.readpickle('initopt_%s.pkl' % dataname)

    #pycs.gen.splml.addtolc(lcs[1], knotstep=200, bokeps=66)

    pycs.sim.draw.saveresiduals(lcs, spline)

    if run_on_copies:
        files_copy = glob.glob("sims_"+ simset_copy + '/*.pkl')
        if len(files_copy)!=0 and askquestions == True:
            answer = raw_input("You already have copies in the folder %s. Do you want to add more ? (yes/no)" %simset_copy)
            if answer[:3] == "yes":
                pycs.sim.draw.multidraw(lcs, onlycopy=True, n=ncopy, npkl=ncopypkls, simset=simset_copy)
            elif answer[:2]  == "no":
                answer2 = raw_input("Should I erase everything and create new ones ? (yes/no)")
                if answer2[:3] == "yes":
                    print "OK, deleting everything ! "
                    for f in files_copy:
                        os.remove(f)
                    pycs.sim.draw.multidraw(lcs, onlycopy=True, n=ncopy, npkl=ncopypkls, simset=simset_copy)
                elif answer2[:2] == "no":
                    print "OK, I am doing nothing then !"
        else :
            for f in files_copy:
                os.remove(f)
                print "deleting %s" % f
            pycs.sim.draw.multidraw(lcs, onlycopy=True, n=ncopy, npkl=ncopypkls, simset=simset_copy)


    if run_on_sims:
            # add splml so that mytweakml will be applied by multidraw
            # Will not work if you have polyml ! But why would you do that ?
            for l in lcs:
                if l.ml == None:
                    pycs.gen.splml.addtolc(l, n=2)

            files_mock = glob.glob("sims_" + simset_mock + '/*.pkl')
            if len(files_mock) != 0 and askquestions == True:
                answer = raw_input("You already have mocks in the folder %s. Do you want to add more ? (yes/no)" % simset_mock)
                if answer[:3] == "yes":
                    pycs.sim.draw.multidraw(lcs, spline, onlycopy=False, n=nsim, npkl=nsimpkls, simset=simset_mock, tweakml=mytweakml, shotnoise=shotnoise_type, truetsr=truetsr, shotnoisefrac=1.0)
                elif answer[:2] == "no":
                    answer2 = raw_input("Should I erase everything and create new ones ? (yes/no)")
                    if answer2[:3] == "yes":
                        print "OK, deleting everything ! "
                        for f in files_mock:
                            os.remove(f)
                        pycs.sim.draw.multidraw(lcs, spline, onlycopy=False, n=nsim, npkl=nsimpkls,
                                                    simset=simset_mock, tweakml=mytweakml, shotnoise=shotnoise_type,
                                                    truetsr=truetsr, shotnoisefrac=1.0)
                    elif answer2[:2] == "no":
                        print "OK, I am doing nothing then !"
            else:
                for f in files_mock:
                    os.remove(f)
                    print "deleting %s"%f
                pycs.sim.draw.multidraw(lcs, spline, onlycopy=False, n=nsim, npkl=nsimpkls,
                                        simset=simset_mock, tweakml=mytweakml, shotnoise=shotnoise_type,
                                        truetsr=truetsr, shotnoisefrac=1.0)


##### Now, we can analyse these results with the method of our choice. WARNING : this may take loooooooong

if opt.sim:

    def applyshifts(lcs, timeshifts, magshifts):

        if not len(lcs) == len(timeshifts) and len(lcs) == len(magshifts):
            print "Hey, give me arrays of the same length !"
            sys.exit()

        for lc, timeshift, magshift in zip(lcs, timeshifts, magshifts):
            lc.resetshifts()
            lc.shiftmag(-np.median(lc.getmags()))
            lc.shifttime(timeshift)

    lcs = pycs.gen.util.readpickle(data)

    ##### We start by shifting our curves "by eye", to get close to the result and help the optimisers to do a good job
    applyshifts(lcs, timeshifts, magshifts)

    # We also give them a microlensing model (here, similar to Courbin 2011)
    attachml(lcs)

    os.chdir(lens_directory + combkw)  # Because carrot

    nworkers = int(cpu_count()*2)
    if run_on_copies:
        def exec_worker(i):
            print "worker %i starting..." %i
            time.sleep(i)
            pycs.sim.run.multirun(simset_copy, lcs, simoptfct, optset=optset, tsrand=tsrand)
        p = Pool(nworkers)
        if simoptfctkw == "spl1":
            p.map(exec_worker, np.arange(nworkers))
        elif simoptfctkw == "regdiff":
            exec_worker(0)  # because for some reason, regdiff does not like multiproc.

    if run_on_sims:
        def exec_worker(i):
            print "worker %i starting..." %i
            time.sleep(i)
            pycs.sim.run.multirun(simset_mock, lcs, simoptfct, optset=optset, tsrand=tsrand, keepopt=True)
        p = Pool(nworkers)
        if simoptfctkw == "spl1":
            p.map(exec_worker, np.arange(nworkers))
        elif simoptfctkw == "regdiff":
            exec_worker(0)  # because for some reason, regdiff does not like multiproc.


##### Finally, we can display the results
if opt.display:

    os.chdir(lens_directory + combkw)
    if run_on_copies:
        copiesres = [pycs.sim.run.collect('sims_%s_opt_%s' %(simset_copy, optset), 'blue', dataname + "_" + combkw)]
        pycs.sim.plot.hists(copiesres, r=50.0, nbins=100, dataout=True)

    if run_on_sims:
        simres = [pycs.sim.run.collect('sims_%s_opt_%s' %(simset_mock, optset), 'blue', dataname + "_" + combkw)]
        pycs.sim.plot.measvstrue(simres, r=30.0, nbins = 10, plotpoints=True, ploterrorbars=True, sidebyside=True, errorrange=20., binclip=True, binclipr=20.0, dataout=True)


    toplot = []
    spl = (pycs.gen.util.readpickle('sims_%s_opt_%s_delays.pkl' %(simset_copy, optset)), pycs.gen.util.readpickle('sims_%s_opt_%s_errorbars.pkl' %(simset_mock, optset)))

    toplot.append(spl)

    text = [(0.43, 0.85, r"$\mathrm{"+full_lensname+"}$", {"fontsize": 18}),
            (0.7, 0.5, 'Warning, spline delays and errors \n are preliminary', {"fontsize": 10})]

    pycs.sim.plot.newdelayplot(toplot, rplot=5.0, displaytext=True, text=text, filename="fig_delays_%s_%s.png" % (simset_mock, optset))
    os.system("display fig_delays_%s_%s.png" % (simset_mock, optset))


