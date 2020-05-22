import sys 
import os
import argparse as ap
import subprocess
import pycs
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def main(name, name_type, number_pair = 1, work_dir = './'): 
	config_directory = work_dir + "config/"
	multiple_config_directory = config_directory + "multiple/"
	data_directory = work_dir + "data/"
	guess_directory = data_directory + name + "/guess/"
	Simulation_directory = work_dir + "Simulation/"
	dataname = 'ECAM'
	Simulation_multiple_directory = Simulation_directory + "multiple/" + name + "_double/"
	if not os.path.exists(Simulation_directory + "multiple/"):
		print "I will create the multiple simulation directory for you ! "
		os.mkdir(Simulation_directory + "multiple/")
	if not os.path.exists(Simulation_multiple_directory):
		os.mkdir(Simulation_multiple_directory)

	
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
	
	### Change here ###
	failed_sim = [42, 53, 58, 95] #all the sim that have failed rung
	golden_sample = [4,5,6,7,8,9,10,14,15,16,17,19,20,21,22,24,25,26,27,29,31,32,36,38,39, 
					 41,43,44,45,48,52,55,56,59,60,61,62,63,64,65,66,67,60,72,74,75,76,81,83,84,91,96,98,99,100]
	###################

	
	for i in range(1,number_pair+1):
		lens_name.append(name + '_' + name_type + '_' + 'pair%i'%i)
		if(i in failed_sim) : 
			median.append('-')
			error_up.append('-')
			error_down.append('-')
		else :
			path =  Simulation_directory + lens_name[i-1] + "_ECAM/marginalisation_spline/marginalisation_spline_sigma_0.50_combined.pkl"
			tmp = pickle.load(open(path,'rb'))
			median.append(tmp.medians[0])
			error_up.append(tmp.errors_up[0])
			error_down.append(tmp.errors_down[0]) 
		print('pair, truth, median, error+, error-')
		print(i, truth[i-1], median[i-1], error_up[i-1], error_down[i-1])
	
	
	### all but failed_sim ###	
	#Summary plot
	x = range(1,number_pair+1)
	for i in failed_sim :
		x.remove(i)
	y = []
	yerr = [[],[]]
	truth_temp = []
	for i in x :
		y.append(median[i-1])
		yerr[0].append(error_down[i-1])
		yerr[1].append(error_up[i-1])
		truth_temp.append(truth[i-1])

	fig1, ax1 = plt.subplots()
	ax1.errorbar(x,y,yerr, marker = '.', linestyle='none', label = 'sim')
	ax1.plot(x, truth_temp,'.', label = 'truth')
	ax1.set_xlabel('Simulation [ ]')
	ax1.set_ylabel('Time delay [d]')
	ax1.set_title('Final results of the simulations')
	ax1.legend()
	fig1.savefig(Simulation_multiple_directory + "Summary.png")
	
	#Accuracy
	y, yerr, truth_temp=np.array(y),np.array(yerr),np.array(truth_temp)
	error = truth_temp-y
	error_rel = []
	for i in range(y.size):
		if y[i-1]<=truth_temp[i-1] :
			error_rel.append((truth_temp[i-1]-y[i-1])/yerr[1][i-1])
		else :
			error_rel.append((truth_temp[i-1]-y[i-1])/yerr[0][i-1])
			
	fig1a, ax1a = plt.subplots()
	ax1a.hist(error_rel, bins=50, density = True, label = 'Simulations')
	ax1a.plot([-1, -1], [0, .5], 'k--', label = '1$\sigma$')
	ax1a.plot([1, 1], [0, .5], 'k--')
	mu_rel, std_rel = norm.fit(error_rel)
	ox = np.linspace(-4.5, 4.5, 100)
	p = norm.pdf(ox, mu_rel, std_rel)
	ax1a.plot(ox, p, linewidth=2, label = 'Gaussian fit : mu = %.2f,  std = %.2f' % (mu_rel, std_rel))
	ax1a.set_xlabel('Relative Error [$\sigma$]')
	ax1a.set_ylabel('Density [ ]')
	ax1a.set_title('Relative error histogram')
	ax1a.legend()
	fig1a.savefig(Simulation_multiple_directory + "Error_rel_hist.png")
	
	fig1b, ax1b = plt.subplots()
	ax1b.hist(error, bins=50, density = True, label = 'Simulations')
	mu, std = norm.fit(error)
	ox = np.linspace(-60, 60, 100)
	p = norm.pdf(ox, mu, std)
	ax1b.plot(ox, p, linewidth=2, label = 'Gaussian fit : mu = %.2f,  std = %.2f' % (mu, std))
	ax1b.set_xlabel('Error [d]')
	ax1b.set_ylabel('Density [ ]')
	ax1b.set_title('Error histogram')
	ax1b.legend()
	fig1b.savefig(Simulation_multiple_directory + "Error_hist.png")
	
	### golden_sample ###
	#Summary plot
	x = golden_sample
	y = []
	yerr = [[],[]]
	truth_temp = []
	for i in x :
		y.append(median[i-1])
		yerr[0].append(error_down[i-1])
		yerr[1].append(error_up[i-1])
		truth_temp.append(truth[i-1])
	fig2, ax2 = plt.subplots()
	ax2.errorbar(x,y,yerr, marker = '.', linestyle='none', label = 'sim')
	ax2.plot(x, truth_temp,'.', label = 'truth')
	ax2.set_xlabel('Simulation [ ]')
	ax2.set_ylabel('Time delay [d]')
	ax2.set_title('Final results of the simulations (golden sample)')
	ax2.legend()
	fig2.savefig(Simulation_multiple_directory + "Summary_golden_sample.png")
	
	#Accuracy
	y, yerr, truth_temp=np.array(y),np.array(yerr),np.array(truth_temp)
	error = truth_temp-y
	error_rel = []
	for i in range(y.size):
		if y[i-1]<=truth_temp[i-1] :
			error_rel.append((truth_temp[i-1]-y[i-1])/yerr[1][i-1])
		else :
			error_rel.append((truth_temp[i-1]-y[i-1])/yerr[0][i-1])
			
	fig2a, ax2a = plt.subplots()
	ax2a.hist(error_rel, bins=50, density = True, label = 'Simulations')
	ax2a.plot([-1, -1], [0, .5], 'k--', label = '1$\sigma$')
	ax2a.plot([1, 1], [0, .5], 'k--')
	mu_rel, std_rel = norm.fit(error_rel)
	ox = np.linspace(-4.5, 4.5, 100)
	p = norm.pdf(ox, mu_rel, std_rel)
	ax2a.plot(ox, p, linewidth=2, label = 'Gaussian fit : mu = %.2f,  std = %.2f' % (mu_rel, std_rel))
	ax2a.set_xlabel('Relative Error [$\sigma$]')
	ax2a.set_ylabel('Counts [ ]')
	ax2a.set_title('Relative error histogram (golden sample)')
	ax2a.legend()
	fig2a.savefig(Simulation_multiple_directory + "Error_rel_hist_golden_sample.png")
	
	fig2b, ax2b = plt.subplots()
	ax2b.hist(error, bins=50, density = True, label = 'Simulations')
	mu, std = norm.fit(error)
	ox = np.linspace(-60, 60, 100)
	p = norm.pdf(ox, mu, std)
	ax2b.plot(ox, p, linewidth=2, label = 'Gaussian fit : mu = %.2f,  std = %.2f' % (mu, std))
	ax2b.set_xlabel('Error [d]')
	ax2b.set_ylabel('Density [ ]')
	ax2b.set_title('Error histogram (golden sample)')
	ax2b.legend()
	fig2b.savefig(Simulation_multiple_directory + "Error_hist_golden_sample.png")
	
	
	

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

