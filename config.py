#####################
#  Configuration file
#####################
import os
from shutil import copyfile
import util_func as ut

lensname = 'UM673'
full_lensname ='UM673'
dataname = "Euler"
lens_directory = "/Users/martin/Desktop/MCMC_pycs/" + lensname + "_" + dataname + "/"
figure_directory = "/Users/martin/Desktop/MCMC_pycs/" + lensname + "_" + dataname + "/figure/"
report_directory = "/Users/martin/Desktop/MCMC_pycs/" + lensname + "_" + dataname + "/report/"

if not os.path.exists(lens_directory):
    print "I will create the lens directory for you ! "
    os.mkdir(lens_directory)

if not os.path.isfile(lens_directory + "config_" + lensname + "_" + dataname + ".py"):
    print "I will create you lens config file ! "
    copyfile("config_default.py",lens_directory+"config_"+ lensname + "_" + dataname + ".py")
    print "Default config file created ! You might want to change the default parameters. "
    ut.proquest(True)

if not os.path.exists(figure_directory):
    os.mkdir(figure_directory)
if not os.path.exists(report_directory):
    os.mkdir(report_directory)

execfile(lens_directory + "config_" + lensname + "_" + dataname + ".py")





