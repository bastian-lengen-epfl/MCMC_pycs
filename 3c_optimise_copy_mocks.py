##### Now, we can analyse these results with the method of our choice. WARNING : this may take loooooooong
import sys
import pycs
import time
from multiprocess import Pool, Lock, cpu_count
import os

execfile("config.py")

def applyshifts(lcs, timeshifts, magshifts):

    if not len(lcs) == len(timeshifts) and len(lcs) == len(magshifts):
        print "Hey, give me arrays of the same length !"
        sys.exit()

    for lc, timeshift, magshift in zip(lcs, timeshifts, magshifts):
        lc.resetshifts()
        lc.shiftmag(-np.median(lc.getmags()))
        lc.shifttime(timeshift)


for a,kn in enumerate(knotstep) :
    for  b, knml in enumerate(mlknotsteps):
        print combkw[a,b]
        os.chdir(lens_directory + combkw[a, b]) # Because carrot
        lcs = pycs.gen.util.readpickle(data)
        ##### We start by shifting our curves "by eye", to get close to the result and help the optimisers to do a good job
        applyshifts(lcs, timeshifts, magshifts)

        # We also give them a microlensing model (here, similar to Courbin 2011)
        attachml(lcs,knml)
        nworkers = int(cpu_count()*2)

        for c, opts in enumerate(optset):
            if simoptfctkw == "spl1":
                kwargs = {'kn' : kn}
            elif simoptfctkw == "regdiff":
                kwargs = kwargs_optimiser_simoptfct[c]
            else :
                print "Error : simoptfctkw mus be spl1 or regdiff"

            if run_on_copies:
                print "I will run the optimiser on the copies with the parameters :", kwargs
                def exec_worker(i):
                    print "worker %i starting..." %i
                    time.sleep(i)
                    pycs.sim.run.multirun(simset_copy, lcs, simoptfct, kwargs_optim=kwargs,
                                          optset=opts, tsrand=tsrand)

                p = Pool(nworkers)
                if simoptfctkw == "spl1":
                    p.map(exec_worker, np.arange(nworkers))
                elif simoptfctkw == "regdiff":
                    exec_worker(0)
                    # p.map(exec_worker, np.arange(nworkers))# because for some reason, regdiff does not like multiproc.

            if run_on_sims:
                print "I will run the optimiser on the simulated lcs with the parameters :", kwargs
                def exec_worker(i):
                    print "worker %i starting..." %i
                    time.sleep(i)
                    pycs.sim.run.multirun(simset_mock, lcs, simoptfct, kwargs_optim=kwargs,
                                          optset=opts, tsrand=tsrand, keepopt=True)

                p = Pool(nworkers)
                if simoptfctkw == "spl1":
                    p.map(exec_worker, np.arange(nworkers))
                elif simoptfctkw == "regdiff":
                    exec_worker(0)  # because for some reason, regdiff does not like multiproc.
                    # p.map(exec_worker, np.arange(nworkers))