##### Now, we can analyse these results with the method of our choice. WARNING : this may take loooooooong
import sys
import pycs
import time
import os, copy
import argparse as ap
import importlib
import numpy
import dill as pkl

def applyshifts(lcs, timeshifts, magshifts):
    if not len(lcs) == len(timeshifts) and len(lcs) == len(magshifts):
        print "Hey, give me arrays of the same length !"
        sys.exit()

    for lc, timeshift, magshift in zip(lcs, timeshifts, magshifts):
        lc.resetshifts()
        lc.shiftmag(-numpy.median(lc.getmags()))
        lc.shifttime(timeshift)


def main(lensname,dataname,work_dir='./'):
    main_path = os.getcwd()
    sys.path.append(work_dir + "config/")
    config_path = os.path.abspath(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)
    base_lcs = pycs.gen.util.readpickle(config.data)
    pkl_n = 0

    for a,kn in enumerate(config.knotstep) :
        for  b, knml in enumerate(config.mlknotsteps):

            print config.combkw[a,b]
            os.chdir(config.lens_directory + config.combkw[a, b]) # Because carrot
            c_path = os.getcwd()
            os.chdir(main_path)

            lcs = copy.deepcopy(base_lcs)
            ##### We start by shifting our curves "by eye", to get close to the result and help the optimisers to do a good job
            applyshifts(lcs, config.timeshifts, config.magshifts)

            # We also give them a microlensing model (here, similar to Courbin 2011)
            config.attachml(lcs,knml)

            for c, opts in enumerate(config.optset):
                pkl_n += 1
                pkl_name_copie = os.path.join(c_path, 'transfert_pickle_copie_%i.pkl'%pkl_n)
                pkl_name_mocks = os.path.join(c_path, 'transfert_pickle_mocks_%i.pkl'%pkl_n)

                if config.simoptfctkw == "spl1":
                    kwargs = {'kn' : kn}
                elif config.simoptfctkw == "regdiff":
                    kwargs = config.kwargs_optimiser_simoptfct[c]
                else :
                    print "Error : simoptfctkw must be spl1 or regdiff"

                if config.run_on_copies:
                    print "I will run the optimiser on the copies with the parameters :", kwargs
                    with open(pkl_name_copie, 'wb') as f :
                        pkl.dump([config.simset_copy, lcs, config.simoptfct, kwargs, opts, config.tsrand], f)

                    if config.simoptfctkw == "spl1":
                        print "Not implemented yet, please use regdiff"

                    elif config.simoptfctkw == "regdiff":
                        os.system("srun -n 1 -c 1 -J %s -u -e %s -o %s python exec_regdiff.py %s %s %s %s &"%
                                  (lensname+'_copies_'+str(kn)+'-'+str(knml),os.path.join(main_path, 'cluster/slurm_regdiff_%i_copie.err'%pkl_n),
                                   os.path.join(main_path, 'cluster/slurm_regdiff_%i_copie.out'%pkl_n) ,pkl_name_copie,'1', c_path, config_path ))
                        print "Job launched on copies ! "
                        time.sleep(0.1)

                if config.run_on_sims:
                    print "I will run the optimiser on the simulated lcs with the parameters :", kwargs
                    with open(pkl_name_mocks, 'wb') as f:
                        pkl.dump([config.simset_mock, lcs, config.simoptfct, kwargs, opts, config.tsrand], f)
                    if config.simoptfctkw == "spl1":
                        print "Not implemented yet, please use regdiff"
                    elif config.simoptfctkw == "regdiff":
                        os.system("srun -n 1 -c 1 -J %s -u -e %s -o %s python exec_regdiff.py %s %s %s %s &" %
                                  (lensname+'_mocks_'+str(kn)+'-'+str(knml),os.path.join(main_path, 'cluster/slurm_regdiff_%i_mocks.err' % pkl_n),
                                   os.path.join(main_path, 'cluster/slurm_regdiff_%i_mocks.out' % pkl_n), pkl_name_mocks, '0',
                                   c_path, config_path))
                        print "Job launched on mocks ! "
                        time.sleep(0.1)

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Shift the mock curves and the copies.",
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