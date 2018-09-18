#####################
#  Configuration file
#####################
import os
from shutil import copyfile
from module import util_func as ut

lensname = 'HE0435b'
full_lensname ='HE0435-1223'
dataname = "Euler"
# lcs_label = ['A','B']
lcs_label = ['A','B','C','D']


module_directory = "/home/epfl/millon/Desktop/MCMC_pycs/"
lens_directory = module_directory + "Simulation/" + lensname + "_" + dataname + "/"
figure_directory = module_directory + "Simulation/" + lensname + "_" + dataname + "/figure/"
report_directory = module_directory + "Simulation/" + lensname + "_" + dataname + "/report/"


if not os.path.exists(lens_directory):
    print "I will create the lens directory for you ! "
    print lens_directory
    os.mkdir(lens_directory)

if not os.path.isfile(lens_directory + "config_" + lensname + "_" + dataname + ".py"):
    print "I will create you lens config file ! "
    if len(lcs_label) == 2 :
        copyfile("config_default_double.py",lens_directory+"config_"+ lensname + "_" + dataname + ".py")
    elif len(lcs_label) == 4 :
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

execfile(lens_directory + "config_" + lensname + "_" + dataname + ".py")





