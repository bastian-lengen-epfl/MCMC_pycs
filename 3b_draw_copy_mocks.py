import pycs
import os,sys, glob
import numpy as np
import dill



execfile("config.py")

for i,kn in enumerate(knotstep) :
    for j, knml in enumerate(mlknotsteps):

        os.chdir(lens_directory+combkw[i,j])
        lcs, spline = pycs.gen.util.readpickle('initopt_%s_ks%i_ksml%i.pkl' % (dataname, kn,knml))

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

                f = open(lens_directory + combkw[i, j] + '/tweakml_' + tweakml_name + '.dill', 'r')
                mytweakml = dill.load(f)

                for l in lcs:
                    if l.ml == None:
                        pycs.gen.splml.addtolc(l, n=2)

                files_mock = glob.glob("sims_" + simset_mock + '/*.pkl')
                if len(files_mock) != 0 and askquestions == True:
                    answer = raw_input("You already have mocks in the folder %s. Do you want to add more ? (yes/no)" % simset_mock)
                    if answer[:3] == "yes":
                        pycs.sim.draw.multidraw(lcs, spline, onlycopy=False, n=nsim, npkl=nsimpkls,
                                                simset=simset_mock, tweakml=mytweakml, shotnoise=shotnoise_type, truetsr=truetsr, shotnoisefrac=1.0)
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
