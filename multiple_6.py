import sys 
import os
import argparse as ap
import subprocess
import pycs
import pickle
import matplotlib.pyplot as plt
import numpy as np

def main(name, name_type, number_pair = 1, work_dir = './'): 
	config_directory = work_dir + "config/"
	multiple_config_directory = config_directory + "multiple/"
	data_directory = work_dir + "data/"
	guess_directory = data_directory + name + "/guess/"
	Simulation_directory = work_dir + "Simulation/"
	dataname = 'ECAM'
	
	truth = []
	with open(guess_directory + 'guess_' + name + '.txt','r') as f:
		Lines=f.readlines()
		for line in Lines :
			truth.append(float(line.partition(' ')[2]))	
	
	### Open the multiple_config 
	with open(multiple_config_directory + 'config_multiple_' + name + '.py', 'r') as f:
		Lines=f.readlines()
		# Update the guess with the sign of the config_multiple file
		if (int(Lines[21][6:9])==-1):
			truth = [-x for x in truth]
		elif (int(Lines[21][6:9])!=1):
			print('ERROR : Make sure the sign of the config_multiple file is +1 or -1')
			sys.exit()			

	lens_name = []
	median = []
	error_up = []
	error_down = []
	for i in range(1,number_pair+1):
		if(i==53) : 
			median.append(0)
			error_up.append(0)
			error_down.append(0)
		else :
			lens_name.append(name + '_' + name_type + '_' + 'pair%i'%i)
			path =  Simulation_directory + lens_name[i-1] + "_ECAM/marginalisation_spline/marginalisation_spline_sigma_0.50_combined.pkl"
			tmp = pickle.load(open(path,'rb'))
			median.append(tmp.medians[0])
			error_up.append(tmp.errors_up[0])
			error_down.append(tmp.errors_down[0]) 
	
	yerr = [error_down, error_up]
	x = range(1,number_pair+1)
	y = median
	plt.errorbar(x,y,yerr, marker = '.', linestyle='none', label = 'sim')
	plt.plot(x, truth[0:number_pair+1],'.', label = 'truth')
	plt.legend()
	plt.savefig("_____TEST")
	
	

	
	
	

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Reformat the txt file from the Time Delay Challenge to an usable rdb file.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_name = "name of the sample. Make sure the directory has the same name."
    help_name_type = "Type of the data ie double or quad"
    help_number_pair = "number of pair in the rung folder. Make sure the folder have the format name_pair0"
    help_work_dir = "name of the work directory"
    parser.add_argument(dest='name', type=str,
                        metavar='name', action='store',
                        help=help_name)
    parser.add_argument(dest='name_type', type=str,
                        metavar='name_type', action='store',
                        help=help_name_type)
    parser.add_argument(dest='number_pair', type=int,
                        metavar='number_pair', action='store',
                        help=help_number_pair)                    
    parser.add_argument('--dir', dest='work_dir', type=str,
                        metavar='', action='store', default='./',
                        help=help_work_dir)
    args = parser.parse_args()
    main(args.name, args.name_type, args.number_pair, work_dir=args.work_dir)

