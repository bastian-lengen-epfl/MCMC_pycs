
#script to erase all the pickle transfert file created during 3e
#Use it when the regdiff optimisation (3e) is finished


import pycs,sys
import os, glob, importlib
import argparse as ap
import shutil


def main(lensname,dataname, work_dir = './'):
    os.chdir('..')
    main_path = os.getcwd()
    sys.path.append(work_dir + "config/")
    sys.path.append(work_dir)
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    if config.simoptfctkw == "regdiff" :
        print "ERROR : your simoptfctkw is regdiff, change it to spl1 to remove the spline optimisations."
        exit()

    print "Warning : I will delete all the optimisation folders."
    from module.util_func import proquest
    proquest(True)

    for a,kn in enumerate(config.knotstep) :
        for  b, knml in enumerate(config.mlknotsteps):
            for o, opt in enumerate(config.optset):

                os.chdir(os.path.join(main_path, config.lens_directory + config.combkw[a, b]))
                files = glob.glob('sims_*_opt_%s' %(opt))
                print "files to remove : ", files
                for file in files :
                    shutil.rmtree(file)


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Remove all the folder related to the spline optimisation",
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
    main(args.lensname, args.dataname, work_dir=args.work_dir)