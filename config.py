#####################
#  Configuration file
#####################
import os
from shutil import copyfile
from module import util_func as ut
import argparse as ap
import numpy as np


def main(lens_name, data_name):
    module_directory = "/Users/martin/Desktop/MCMC_pycs/"
    data_directory = module_directory + "data/"
    pickle_directory = module_directory + "pkl/"
    simu_directory = module_directory + "Simulation/"
    lens_directory = module_directory + "Simulation/" + lensname + "_" + dataname + "/"
    figure_directory = module_directory + "Simulation/" + lensname + "_" + dataname + "/figure/"
    report_directory = module_directory + "Simulation/" + lensname + "_" + dataname + "/report/"
    if not os.path.exists(data_directory):
        print "I will create the data directory for you ! "
        os.mkdir(data_directory)
    if not os.path.exists(pickle_directory):
        print "I will create the pickle directory for you ! "
        os.mkdir(pickle_directory)
    if not os.path.exists(simu_directory):
        print "I will create the simulation directory for you ! "
        os.mkdir(simu_directory)
    if not os.path.exists(lens_directory):
        print "I will create the lens directory for you ! "
        os.mkdir(lens_directory)

    rdbfile = 'data/' + lensname + "_" + dataname + '.rdb'
    d = np.loadtxt(rdbfile, skiprows=2)
    n_curve = (len(d[0,:]) -1) / 2

    if not os.path.isfile(lens_directory + "config_" + lensname + "_" + dataname + ".py"):
        print "I will create you lens config file ! "
        if n_curve == 2 :
            copyfile("config_default_double.py",lens_directory+"config_"+ lensname + "_" + dataname + ".py")
        elif n_curve == 4 :
            copyfile("config_default_quads.py", lens_directory + "config_" + lensname + "_" + dataname + ".py")
        else :
            print " Warning : do you have a quad or a double ? Make sure you update lcs_label in the config file ! I'll copy the double template for this time !"
            copyfile("config_default_double.py", lens_directory + "config_" + lensname + "_" + dataname + ".py")

        print "Default config file created ! You might want to change the default parameters. "
        ut.proquest(True)

    if not os.path.exists(figure_directory):
        os.mkdir(figure_directory)
    if not os.path.exists(report_directory):
        os.mkdir(report_directory)

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Config script to organise the folder when using pycs pipeline.",
                               formatter_class=ap.RawTextHelpFormatter)

    help_lensname = "name of the lens to process"
    help_dataname = "name of the data set to process (Euler, SMARTS, ... )"
    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument(dest='dataname', type=str,
                        metavar='dataname', action='store',
                        help=help_dataname)
    args = parser.parse_args()
    main(args.lensname,args.dataname)



