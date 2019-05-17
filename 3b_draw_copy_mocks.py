import os
import matplotlib as mpl
mpl.use('Agg') #these scripts re for cluster so need to be sure
import matplotlib.pyplot as plt
import pycs
import sys, glob
import argparse as ap
import multiprocess

def draw_mock_para(i, j, kn, ml,string_ML, lensname, dataname, work_dir):
    current_dir = os.getcwd()
    import importlib
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    print "I am drawing curves for ks%i, ksml%i" % (kn, ml)
    os.chdir(config.lens_directory + config.combkw[i, j])
    lcs, spline = pycs.gen.util.readpickle('initopt_%s_ks%i_%s%i.pkl' % (dataname, kn,string_ML,ml))

    pycs.sim.draw.saveresiduals(lcs, spline)

    if config.run_on_copies:
        files_copy = glob.glob("sims_" + config.simset_copy + '/*.pkl')
        pycs.sim.draw.multidraw(lcs, onlycopy=True, n=config.ncopy, npkl=config.ncopypkls,
                                    simset=config.simset_copy)

    if config.run_on_sims:
        # add splml so that mytweakml will be applied by multidraw

        for l in lcs:
            if l.ml == None:
                print ('Adding flat ML')
                pycs.gen.splml.addtolc(l, n=2)
            elif l.ml.mltype == 'poly' :
                print('Adding flat ML, that can be tweaked')
                l.resetml()
                pycs.gen.splml.addtolc(l, n=2)

        # import the module with the parameter of the noise :
        print 'I will use the parameter from : %s' % ('tweakml_' + config.tweakml_name + '.py')
        # sys.path.append(os.getcwd())
        # noise_module = importlib.import_module('tweakml_' + config.tweakml_name)
        execfile('tweakml_' + config.tweakml_name + '.py', globals())

        files_mock = glob.glob("sims_" + config.simset_mock + '/*.pkl')
        pycs.sim.draw.multidraw(lcs, spline, onlycopy=False, n=config.nsim, npkl=config.nsimpkls,
                                    simset=config.simset_mock, tweakml=tweakml_list,
                                    shotnoise=config.shotnoise_type,
                                    truetsr=config.truetsr, shotnoisefrac=1.0)
    os.chdir(current_dir)

def draw_mock_para_aux(args):
    return draw_mock_para(*args)

def main(lensname,dataname,work_dir='./'):
    import importlib
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    if config.max_core == None :
        processes = multiprocess.cpu_count()
    else :
        processes = config.max_core

    p = multiprocess.Pool(processes=processes)
    print "Runing on %i cores. "%processes
    job_args = []

    if config.mltype == "splml":
        if config.forcen :
            ml_param = config.nmlspl
            string_ML ="nmlspl"
        else :
            ml_param = config.mlknotsteps
            string_ML = "knml"
    elif config.mltype == "polyml" :
        ml_param = config.degree
        string_ML = "deg"
    else :
        raise RuntimeError('I dont know your microlensing type. Choose "polyml" or "spml".')



    for i,kn in enumerate(config.knotstep) :
        for j, ml in enumerate(ml_param):
            for simset in [config.simset_mock, config.simset_copy] :
                file= glob.glob(os.path.join(config.lens_directory + config.combkw[i, j],"sims_" + simset + '/*.pkl'))
                if len(file) != 0 and config.askquestions == True:
                    while True :
                        answer = int(raw_input(
                            "You already have files in the folder %s. Do you want to add more (1) or replace the existing file (2) ? (1/2)" % simset))
                        if answer != 1 or answer != 2 :
                            break
                        else :
                            print "I did not understand your answer."

                    if answer == 1:
                        print "OK, deleting everything ! "
                        for f in file:
                            os.remove(f)
                    elif answer == 2 :
                        print "OK, I'll add more mocks !"
                elif len(file) != 0 :
                    print "You already have files in the folder %s. You did not turn your ask question flag. By default, I will replace your simulation !"%simset
                    print "OK, deleting ! "
                    for f in file:
                        os.remove(f)

            job_args.append((i,j,kn,ml,string_ML,lensname, dataname, work_dir))
    p.map(draw_mock_para_aux, job_args)
    print "Done."

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