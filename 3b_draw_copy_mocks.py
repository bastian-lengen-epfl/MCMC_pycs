import pycs
import os,sys, glob
import argparse as ap

def main(lensname,dataname,work_dir='./'):
    import importlib
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)
    main_path = os.getcwd()

    for i,kn in enumerate(config.knotstep) :
        for j, knml in enumerate(config.mlknotsteps):
            os.chdir(main_path)
            print "I am drawing curves for ks%i, ksml%i" %(kn,knml)
            os.chdir(config.lens_directory+config.combkw[i,j])
            lcs, spline = pycs.gen.util.readpickle('initopt_%s_ks%i_ksml%i.pkl' % (dataname, kn,knml))

            pycs.sim.draw.saveresiduals(lcs, spline)

            if config.run_on_copies:
                files_copy = glob.glob("sims_"+ config.simset_copy + '/*.pkl')
                if len(files_copy)!=0 and config.askquestions == True:
                    answer = raw_input("You already have copies in the folder %s. Do you want to add more ? (yes/no)" %config.simset_copy)
                    if answer[:3] == "yes":
                        pycs.sim.draw.multidraw(lcs, onlycopy=True, n=config.ncopy, npkl=config.ncopypkls, simset=config.simset_copy)
                    elif answer[:2]  == "no":
                        answer2 = raw_input("Should I erase everything and create new ones ? (yes/no)")
                        if answer2[:3] == "yes":
                            print "OK, deleting everything ! "
                            for f in files_copy:
                                os.remove(f)
                            pycs.sim.draw.multidraw(lcs, onlycopy=True, n=config.ncopy, npkl=config.ncopypkls, simset=config.simset_copy)
                        elif answer2[:2] == "no":
                            print "OK, I am doing nothing then !"
                else :
                    for f in files_copy:
                        os.remove(f)
                        print "deleting %s" % f
                    pycs.sim.draw.multidraw(lcs, onlycopy=True, n=config.ncopy, npkl=config.ncopypkls, simset=config.simset_copy)


            if config.run_on_sims:
                # add splml so that mytweakml will be applied by multidraw
                # Will not work if you have polyml ! But why would you do that ?

                for l in lcs:
                    if l.ml == None:
                        pycs.gen.splml.addtolc(l, n=2)

                #import the module with the parameter of the noise :
                print 'I will use the parameter from : %s'%('tweakml_' + config.tweakml_name + '.py')
                # sys.path.append(os.getcwd())
                # noise_module = importlib.import_module('tweakml_' + config.tweakml_name)
                execfile('tweakml_' + config.tweakml_name +'.py', globals())

                files_mock = glob.glob("sims_" + config.simset_mock + '/*.pkl')
                if len(files_mock) != 0 and config.askquestions == True:
                    answer = raw_input("You already have mocks in the folder %s. Do you want to add more ? (yes/no)" % simset_mock)
                    if answer[:3] == "yes":
                        pycs.sim.draw.multidraw(lcs, spline, onlycopy=False, n=config.nsim, npkl=config.nsimpkls,
                                                simset=config.simset_mock, tweakml=tweakml_list, shotnoise=config.shotnoise_type, truetsr=config.truetsr, shotnoisefrac=1.0)
                    elif answer[:2] == "no":
                        answer2 = raw_input("Should I erase everything and create new ones ? (yes/no)")
                        if answer2[:3] == "yes":
                            print "OK, deleting everything ! "
                            for f in files_mock:
                                os.remove(f)
                            pycs.sim.draw.multidraw(lcs, spline, onlycopy=False, n=config.nsim, npkl=config.nsimpkls,
                                                        simset=config.simset_mock, tweakml=tweakml_list, shotnoise=config.shotnoise_type,
                                                        truetsr=config.truetsr, shotnoisefrac=1.0)
                        elif answer2[:2] == "no":
                            print "OK, I am doing nothing then !"
                else:
                    for f in files_mock:
                        os.remove(f)
                        print "deleting %s"%f
                    pycs.sim.draw.multidraw(lcs, spline, onlycopy=False, n=config.nsim, npkl=config.nsimpkls,
                                            simset=config.simset_mock, tweakml=tweakml_list, shotnoise=config.shotnoise_type,
                                            truetsr=config.truetsr, shotnoisefrac=1.0)


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Prepare the copies of the light curves and draw some mock curves.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_lensname = "name of the lens to process"
    help_dataname = "name of the data set to process (Euler, SMARTS, ... )"
    help_work_dir = "name of the working directory"
    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument(dest='dataname', type=str,
                        metavar='dataname', action='store',
                        help=help_dataname)
    parser.add_argument('--dir', dest='work_dir', type=str,
                            metavar='', action='store', default='./',
                            help=help_work_dir)
    args = parser.parse_args()
    main(args.lensname,args.dataname, work_dir=args.work_dir)