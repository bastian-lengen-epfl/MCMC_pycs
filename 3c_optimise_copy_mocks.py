##### Now, we can analyse these results with the method of our choice. WARNING : this may take loooooooong
##### I'm not re-running on already optimized lcs !
import sys
import pycs
import time
from multiprocess import Pool, Lock, cpu_count
import os, copy
import argparse as ap
import importlib
import numpy
import pickle as pkl

def applyshifts(lcs, timeshifts, magshifts):
    if not len(lcs) == len(timeshifts) and len(lcs) == len(magshifts):
        print "Hey, give me arrays of the same length !"
        sys.exit()

    for lc, timeshift, magshift in zip(lcs, timeshifts, magshifts):
        lc.resetshifts()
        lc.shiftmag(-numpy.median(lc.getmags()))
        lc.shifttime(timeshift)

def exec_worker_copie_aux(args):
    return exec_worker_copie(*args)


def exec_worker_copie(i, simset_copy, lcs, simoptfct, kwargs_optim, optset, tsrand, destpath):
    print "worker %i starting..." % i
    time.sleep(i)
    sucess_dic = pycs.sim.run.multirun(simset_copy, lcs, simoptfct, kwargs_optim=kwargs_optim,
                          optset=optset, tsrand=tsrand, destpath= destpath)
    return sucess_dic

def exec_worker_mocks_aux(args):
    return exec_worker_mocks(*args)


def exec_worker_mocks(i, simset_mock, lcs, simoptfct, kwargs_optim, optset, tsrand, destpath):
    print "worker %i starting..." % i
    time.sleep(i)
    sucess_dic = pycs.sim.run.multirun(simset_mock, lcs,simoptfct, kwargs_optim=kwargs_optim,
                          optset=optset, tsrand=tsrand, keepopt=True, destpath=destpath)
    return sucess_dic

def write_report_optimisation(f, success_dic):
    if success_dic == None :
        f.write('This set was already optimised.\n')
    else :
        if success_dic['success']:
            f.write('None of the optimisations have failed. \n')
        else :
            f.write('The optimisation of the following curves have failed : \n')
            for id in success_dic['failed_id']:
                f.write("Curve %i :"%id + success_dic['error_list'][id])
                f.write('\n')


def main(lensname,dataname,work_dir='./'):
    main_path = os.getcwd()
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)
    base_lcs = pycs.gen.util.readpickle(config.data)
    f = open(os.path.join(config.report_directory, 'report_optimisation_%s.txt'%config.simoptfctkw), 'w')
    for a,kn in enumerate(config.knotstep) :
        for  b, knml in enumerate(config.mlknotsteps):
            lcs = copy.deepcopy(base_lcs)
            destpath = os.path.join(main_path, config.lens_directory + config.combkw[a, b] + '/')
            print destpath
            ##### We start by shifting our curves "by eye", to get close to the result and help the optimisers to do a good job
            applyshifts(lcs, config.timeshifts, config.magshifts)

            # We also give them a microlensing model (here, similar to Courbin 2011)
            config.attachml(lcs,knml)
            if config.max_core == None :
                nworkers = cpu_count()
            else :
                nworkers = config.max_core

            for c, opts in enumerate(config.optset):
                if config.simoptfctkw == "spl1":
                    kwargs = {'kn' : kn, 'name':'spl1'}
                elif config.simoptfctkw == "regdiff":
                    kwargs = config.kwargs_optimiser_simoptfct[c]
                else :
                    print "Error : simoptfctkw must be spl1 or regdiff"

                if config.run_on_copies:
                    print "I will run the optimiser on the copies with the parameters :", kwargs
                    p = Pool(nworkers)
                    if config.simoptfctkw == "spl1":
                        job_args = [(j, config.simset_copy, lcs, config.simoptfct, kwargs, opts, config.tsrand, destpath) for j in
                                    range(nworkers)]
                        success_list_copies = p.map(exec_worker_copie_aux, job_args)
                    elif config.simoptfctkw == "regdiff":
                        if a == 0 and b == 0 : # for copies, run on only 1 (knstp,mlknstp) as it the same for others
                            job_args = (0, config.simset_copy, lcs, config.simoptfct, kwargs, opts, config.tsrand, destpath)
                            success_list_copies = exec_worker_copie_aux(job_args)
                            dir_link = os.path.join(destpath,"sims_%s_opt_%s" % (config.simset_copy, opts))
                            print "Dir link :", dir_link
                            pkl.dump(dir_link,open(os.path.join(config.lens_directory,'regdiff_copies_link_%s.pkl'%kwargs['name']),'w'))
                        # p.map(exec_worker_copie_aux, job_args)# because for some reason, regdiff does not like multiproc.
                    f.write('COPIES, kn%i, knml%i, optimiseur %s : \n' % (kn, knml, kwargs['name']))
                    write_report_optimisation(f, success_list_copies)
                    f.write('################### \n')

                if config.run_on_sims:
                    print "I will run the optimiser on the simulated lcs with the parameters :", kwargs
                    p = Pool(nworkers)
                    if config.simoptfctkw == "spl1":
                        job_args = [(j, config.simset_mock, lcs, config.simoptfct, kwargs, opts, config.tsrand, destpath) for j in
                                    range(nworkers)]
                        success_list_simu = p.map(exec_worker_mocks_aux, job_args)
                    elif config.simoptfctkw == "regdiff":
                        job_args = (0, config.simset_mock, lcs, config.simoptfct, kwargs, opts, config.tsrand, destpath)
                        success_list_simu = exec_worker_mocks_aux(job_args)  # because for some reason, regdiff does not like multiproc.
                        # p.map(exec_worker_copie_aux, job_args)
                    f.write('SIMULATIONS, kn%i, knml%i, optimiseur %s : \n' % (kn, knml, kwargs['name']))
                    write_report_optimisation(f, success_list_simu)
                    f.write('################### \n')

    print "OPTIMISATION DONE : report written in %s"%(os.path.join(config.report_directory, 'report_optimisation_%s.txt'%config.simoptfctkw))
    f.close()

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