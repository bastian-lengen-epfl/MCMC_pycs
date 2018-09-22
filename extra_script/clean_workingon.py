#script to erase all the workingon file in the simulations of an object.
#Use it when the optimisation (3c) has crashed


import pycs,sys
import os, glob, importlib
import numpy as np
import argparse as ap


def main(lensname,dataname, work_dir = './'):
    os.chdir(work_dir)
    sys.path.append(work_dir + "config/")
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    for a,kn in enumerate(config.knotstep) :
        for  b, knml in enumerate(config.mlknotsteps):
            os.chdir(config.lens_directory + config.combkw[a, b])
            files = glob.glob('sims*/*.workingon')
            print "files to remove : ", files
            for fil in files :
                os.remove(fil)


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Remove the temporary files created during the optimisation.",
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
