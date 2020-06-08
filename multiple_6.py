# coding=utf-8

import sys 
import os
import argparse as ap
import subprocess
import pycs
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
	failed_sim = [53, 58, 76, 131] #all the sim that have failed rung
	silver_sample = [] # precision >40% + cut delay 100d, done later automatically
	golden_sample = [11, 13, 18, 28, 30, 33, 34, 35, 46, 47, 49, 51, 54, 57, 73, 80, 86, 87, 88, 89, 90, 92, 93, 102,
					 105, 106, 107, 108, 116, 118, 120, 124, 135, 137, 139, 143] # red and yellow with D3CS
	###################
	
	### For the final plot ###
	total_f    = [ 0.22,   0.18,   0.02,   0.34,   0.34,   0.30,   0.30,   0.30,   0.28]
	total_chi2 = [ 0.59,   0.78,   0.51,   1.165,  0.458,  0.099,  0.813,  0.494,  1.28]
	total_P    = [ 0.097,  0.06,   0.155,  0.036,  0.059,  0.247,  0.068,  0.042,  0.051]
	total_A    = [ 0.000, -0.003,  0.037,  0.002, -0.020, -0.030, -0.004, -0.001,  0.007]
	total_X    = [ 0.66,   0.96,   0.95,   0.98,   1.0,    1.0,    1.0,    1.0,    0.95]
	total_f, total_chi2, total_P, total_A, total_X = np.array(total_f), np.array(total_chi2), np.array(total_P), np.array(total_A), np.array(total_X)
	#total_chi2 = np.log10(total_chi2)
	#total_P = np.log10(total_P)
	#Author
	author = ['Rumbaugh', 'Hojjati', 'Kumar', 'Jackson', 'Shafieloo', 'pyCS-D3CS', 'pyCS-SDI',
	        'pyCS-SPL', 'JPL', 'Full Sample', 'Silver Sample', 'Golden Sample']
	color = ['silver','red', 'orange', 'chartreuse', 'green', 'cornflowerblue', 'blue',
	         'darkblue', 'darkmagenta', 'black', 'dimgray', 'darkgoldenrod']
	mark = ['o','d','s','D','*','p','<',
	        '>','^','X','X','X']


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
	
	#remove the overlaping
	for i in range(number_pair):
		if (np.abs(truth[i])>100) :
			silver_sample.append(i+1)
	print("n# Truth > 100 days  = " , len(silver_sample))
	
	
	
	
	################################################
	########  Plot for all but failed_sim ##########
	################################################
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
	ax1.errorbar(x,y,yerr, marker = '.', linestyle='none', label = 'Simulated time-delay')
	ax1.plot(x, truth_temp,'.r', label = 'True time-delay')
	ax1.set_xlabel('Simulation [ ]')
	ax1.set_ylabel('Time delay [d]')
	ax1.legend(loc='upper left')
	ax1.set_ylim([-170, 200])
	fig1.savefig(Simulation_multiple_directory + "Summary.png", dpi = 200)
	
	#error_rel plot
	y, yerr, truth_temp=np.array(y),np.array(yerr),np.array(truth_temp)
	error = truth_temp-y
	error_rel = []
	for i in range(y.size):
		if y[i-1]<=truth_temp[i-1] :
			error_rel.append(error[i-1]/yerr[1][i-1])
		else :
			error_rel.append(error[i-1]/yerr[0][i-1])
			
	fig1a, ax1a = plt.subplots()
	n, bins, jspckoi = ax1a.hist(error_rel, bins=50, label = 'Simulations')
	ax1a.plot([-1, -1], [0, 20], 'k--', linewidth=2, label = '1$\sigma$ threshold')
	ax1a.plot([1, 1], [0, 20], 'k--', linewidth=2)
	mu_rel, std_rel = norm.fit(error_rel)
	ox = np.linspace(-4, 8, 100)
	p = sum(n * np.diff(bins))*norm.pdf(ox, mu_rel, std_rel)
	ax1a.plot(ox, p, 'r', linewidth=2, label = 'Gaussian fit : mean = %.2f,  std = %.2f' % (mu_rel, std_rel))
	ax1a.set_xlabel('Relative Error [$\sigma$]')
	ax1a.set_ylabel('Count [ ]')
	ax1a.legend()
	ax1a.set_xlim([-4,8])
	ax1a.set_ylim([0, 20])
	ax1a.set_yticks([0, 5, 10, 15, 20])
	fig1a.savefig(Simulation_multiple_directory + "Error_rel_hist.png", dpi = 200)
	
	#Relative precision
	rel_prec = 1/2.*(yerr[1]+yerr[0])/np.abs(y)*100
	#Add silver_sample >40% (to remove)
	for i in range(len(x)) :
		if rel_prec[i]>=40 :
			if x[i] not in silver_sample :
				silver_sample.append(x[i])
	silver_sample = sorted(silver_sample)
	

	fig1c, ax1c = plt.subplots()
	mu_rel_prec, std_rel_prec =np.mean(rel_prec), np.std(rel_prec)
	ax1c.hist(rel_prec, bins=50, label = 'Simulations')
	p16=np.percentile(rel_prec, 16)
	p84=np.percentile(rel_prec, 84)
	ax1c.plot([mu_rel_prec, mu_rel_prec], [0, 35], 'r', linewidth=2, label = 'mean = %.2f' % mu_rel_prec)
	ax1c.plot([p16, p16], [0, 35], '--k', linewidth=2, label = '16th percentile = %.2f' % p16)
	ax1c.plot([p84, p84], [0, 35], '-.k', linewidth=2, label = '84th percentile = %.2f' % p84)
	ax1c.set_xlabel('Relative Precision [%]')
	ax1c.set_ylabel('Count [ ]')
	ax1c.legend()
	ax1c.set_xlim([0, 175])
	ax1c.set_ylim([0, 35])
	fig1c.savefig(Simulation_multiple_directory + "Relative_precision_hist.png", dpi = 200)
	
	#Relative accuracy 
	rel_acc = error/truth_temp*100
	
	fig1d, ax1d = plt.subplots()
	mu_rel_acc, std_rel_acc =np.mean(rel_acc), np.std(rel_acc)
	ax1d.hist(rel_acc, bins=50, label = 'Simulations')
	p16=np.percentile(rel_acc, 16)
	p84=np.percentile(rel_acc, 84)
	ax1d.plot([mu_rel_acc, mu_rel_acc], [0, 50], 'r', linewidth=2, label = 'mean = %.2f' % mu_rel_acc)
	ax1d.plot([p16, p16], [0, 50], '--k', linewidth=2, label = '16th percentile = %.2f' % p16)
	ax1d.plot([p84, p84], [0, 50], '-.k', linewidth=2, label = '84th percentile = %.2f' % p84)
	ax1d.set_xlabel('Relative Accuracy [%]')
	ax1d.set_ylabel('Count [ ]')
	ax1d.legend()
	ax1d.set_ylim([0, 50])
	fig1d.savefig(Simulation_multiple_directory + "Relative_accuracy_hist.png", dpi = 200)
	
	#Chi2
	chi2 = error**2./(1/2.*yerr[1]+1/2.*yerr[0])**2.
	
	fig1e, ax1e = plt.subplots()	
	mu_chi2, std_chi2 =np.mean(chi2), np.std(chi2)
	ax1e.hist(chi2, bins=50, label = 'Simulations')
	p16=np.percentile(chi2, 16)
	p84=np.percentile(chi2, 84)
	ax1e.plot([mu_chi2, mu_chi2], [0, 120], 'r', linewidth=2, label = 'mean = %.2f' % mu_chi2)
	ax1e.plot([p16, p16], [0, 120], '--k', linewidth=2, label = '16th percentile = %.2f' % p16)
	ax1e.plot([p84, p84], [0, 120], '-.k', linewidth=2, label = '84th percentile = %.2f' % p84)
	ax1e.set_xlabel('$\chi^2$ [ ]')
	ax1e.set_ylabel('Count [ ]')
	ax1e.legend()
	ax1e.set_xlim([0, 65])
	ax1e.set_ylim([0, 120])
	fig1e.savefig(Simulation_multiple_directory + "Chi2_hist.png", dpi = 200)
		
		
	# Summary stats and plot
	f = (number_pair-len(failed_sim))/float(number_pair) #n submitted / n total
	sum_chi2 = 1/(f*number_pair)*np.sum(chi2)
	sum_P = 1/(f*number_pair)*np.sum(rel_prec/100.)
	sum_A = 1/(f*number_pair)*np.sum(rel_acc/100.)
	X = 0
	for i in chi2 :
		if i<10 :
			X += 1;
	X = X/float(number_pair-len(failed_sim))

	print('all but failed :', f, sum_chi2, sum_P, sum_A, X)
	print('size of the sample :', number_pair-len(failed_sim))
	
	total_f   = np.append(total_f, f)
	total_chi2= np.append(total_chi2, sum_chi2)
	total_P   = np.append(total_P, sum_P)
	#total_chi2= np.append(total_chi2, np.log10(sum_chi2))
	#total_P   = np.append(total_P, np.log10(sum_P))
	total_A   = np.append(total_A, sum_A)
	total_X   = np.append(total_X, X)
	
	all_P = rel_prec
	all_A = rel_acc
	all_chi2 = chi2
		
	################################################
	#########  Plot for the silver sample ###########
	################################################
	#Summary plot
	for i in silver_sample :
		x.remove(i)
	y = []
	yerr = [[],[]]
	truth_temp = []
	for i in x :
		y.append(median[i-1])
		yerr[0].append(error_down[i-1])
		yerr[1].append(error_up[i-1])
		truth_temp.append(truth[i-1])

	fig2, ax2 = plt.subplots()
	ax2.errorbar(x,y,yerr, marker = '.', linestyle='none', label = 'Simulated time-delay')
	ax2.plot(x, truth_temp,'.r', label = 'True time-delay')
	ax2.set_xlabel('Simulation [ ]')
	ax2.set_ylabel('Time delay [d]')
	ax2.legend(loc='upper left')
	ax2.set_ylim([-170, 200])
	fig2.savefig(Simulation_multiple_directory + "Summary_SS.png", dpi = 200)
	
	#error_rel plot
	y, yerr, truth_temp=np.array(y),np.array(yerr),np.array(truth_temp)
	error = truth_temp-y
	error_rel = []
	for i in range(y.size):
		if y[i-1]<=truth_temp[i-1] :
			error_rel.append(error[i-1]/yerr[1][i-1])
		else :
			error_rel.append(error[i-1]/yerr[0][i-1])
			
	fig2a, ax2a = plt.subplots()
	n, bins, jspckoi = ax2a.hist(error_rel, bins=50, label = 'Simulations')
	ax2a.plot([-1, -1], [0, 10], 'k--', linewidth=2, label = '1$\sigma$ threshold')
	ax2a.plot([1, 1], [0, 10], 'k--', linewidth=2)
	mu_rel, std_rel = norm.fit(error_rel)
	ox = np.linspace(-2.5, 2.5, 100)
	p = sum(n*np.diff(bins))*norm.pdf(ox, mu_rel, std_rel)
	ax2a.plot(ox, p, 'r', linewidth=2, label = 'Gaussian fit : mean = %.2f,  std = %.2f' % (mu_rel, std_rel))
	ax2a.set_xlabel('Relative Error [$\sigma$]')
	ax2a.set_ylabel('Count [ ]')
	ax2a.set_xlim([-2.5, 2.5])
	ax2a.set_ylim([0, 10])
	ax2a.legend()
	fig2a.savefig(Simulation_multiple_directory + "Error_rel_hist_SS.png", dpi = 200)
	
	#Relative precision
	rel_prec = 1/2.*(yerr[1]+yerr[0])/np.abs(y)*100

	fig2c, ax2c = plt.subplots()
	#bins_tmp = np.linspace(0, 10, 50)
	mu_rel_prec, std_rel_prec =np.mean(rel_prec), np.std(rel_prec)
	ax2c.hist(rel_prec, bins=50, label = 'Simulations')
	p16=np.percentile(rel_prec, 16)
	p84=np.percentile(rel_prec, 84)
	ax2c.plot([mu_rel_prec, mu_rel_prec], [0, 10], 'r', label = 'mean = %.2f' % mu_rel_prec)
	ax2c.plot([p16, p16], [0, 10], '--k', linewidth=2, label = '16th percentile = %.2f' % p16)
	ax2c.plot([p84, p84], [0, 10], '-.k', linewidth=2, label = '84th percentile = %.2f' % p84)
	ax2c.set_xlabel('Relative Precision [%]')
	ax2c.set_ylabel('Count [ ]')
	ax2c.legend()
	ax2c.set_xlim([0, 40])
	ax2c.set_ylim([0,10])
	fig2c.savefig(Simulation_multiple_directory + "Relative_precision_hist_SS.png", dpi = 200)
	
	#Relative accuracy 
	rel_acc = error/truth_temp*100
	
	fig2d, ax2d = plt.subplots()
	ax2d.hist(rel_acc, bins=50, label = 'Simulations')
	mu_rel_acc, std_rel_acc =np.mean(rel_acc), np.std(rel_acc)
	p16=np.percentile(rel_acc, 16)
	p84=np.percentile(rel_acc, 84)
	ax2d.plot([mu_rel_acc, mu_rel_acc], [0, 25], 'r', linewidth=2, label = 'mean = %.2f' % mu_rel_acc)
	ax2d.plot([p16, p16], [0, 25], '--k', linewidth=2, label = '16th percentile = %.2f' % p16)
	ax2d.plot([p84, p84], [0, 25], '-.k', linewidth=2, label = '84th percentile = %.2f' % p84)
	ax2d.set_xlabel('Relative Accuracy [%]')
	ax2d.set_ylabel('Count [ ]')
	ax2d.legend()
	ax2d.set_ylim([0, 25])
	ax2d.set_xlim([-60, 40])
	fig2d.savefig(Simulation_multiple_directory + "Relative_accuracy_hist_SS.png", dpi = 200)
	
	#Chi2
	chi2 = error**2./(1/2.*yerr[1]+1/2.*yerr[0])**2.
	
	fig2e, ax2e = plt.subplots()
	ax2e.hist(chi2, bins=50, label = 'Simulations')
	mu_chi2, std_chi2 =np.mean(chi2), np.std(chi2)
	p16=np.percentile(chi2, 16)
	p84=np.percentile(chi2, 84)
	ax2e.plot([mu_chi2, mu_chi2], [0, 50], 'r', linewidth=2, label = 'mean = %.2f' % mu_chi2)
	ax2e.plot([p16, p16], [0, 50], '--k', linewidth=2, label = '16th percentile = %.2f' % p16)
	ax2e.plot([p84, p84], [0, 50], '-.k', linewidth=2, label = '84th percentile = %.2f' % p84)
	ax2e.set_xlabel('$\chi^2$ [ ]')
	ax2e.set_ylabel('Count [ ]')
	ax2e.legend()
	ax2e.set_xlim([0,8 ])
	ax2e.set_ylim([0, 50])
	fig2e.savefig(Simulation_multiple_directory + "Chi2_hist_SS.png", dpi = 200)
	
	
	# Summary stats and plot
	f = (number_pair-len(failed_sim)-len(silver_sample))/float(number_pair) #n submitted / n total
	sum_chi2 = 1/(f*number_pair)*np.sum(chi2)
	sum_P = 1/(f*number_pair)*np.sum(rel_prec/100.)
	sum_A = 1/(f*number_pair)*np.sum(rel_acc/100.)
	X = 0
	for i in chi2 :
		if i<10 :
			X += 1;
	X = X/(number_pair-len(failed_sim)-len(silver_sample))

	print('Silver Sample :', f, sum_chi2, sum_P, sum_A, X)
	print('size of the sample :', number_pair-len(failed_sim)-len(silver_sample))
	
	total_f   = np.append(total_f, f)
	total_chi2= np.append(total_chi2, sum_chi2)
	total_P   = np.append(total_P, sum_P)
	#total_chi2= np.append(total_chi2, np.log10(sum_chi2))
	#total_P   = np.append(total_P, np.log10(sum_P))
	total_A   = np.append(total_A, sum_A)
	total_X   = np.append(total_X, X)
	
	################################################
	#########  Plot for the golden sample ##########
	################################################
	#Summary plot
	for i in golden_sample :
		if i not in silver_sample :
			x.remove(i)
	y = []
	yerr = [[],[]]
	truth_temp = []
	for i in x :
		y.append(median[i-1])
		yerr[0].append(error_down[i-1])
		yerr[1].append(error_up[i-1])
		truth_temp.append(truth[i-1])
		
	'''
	count = 0
	for i in range(len(y)) :
		if ((truth_temp[i]>=y[i]-yerr[0][i]) and (truth_temp[i]<=y[i]+yerr[1][i])):
			count+=1
			print('True :', truth_temp[i], ' in ',y[i]-yerr[0][i], y[i]+yerr[1][i])
		else : 
			print('False :', truth_temp[i], ' not in ',y[i]-yerr[0][i], y[i]+yerr[1][i])
	print(' Count in 1 sigma range : ', count)
	'''

	fig3, ax3 = plt.subplots()
	ax3.errorbar(x,y,yerr, marker = '.', linestyle='none', label = 'Simulated time-delay')
	ax3.plot(x, truth_temp,'.r', label = 'True time-delay')
	ax3.set_xlabel('Simulation [ ]')
	ax3.set_ylabel('Time delay [d]')
	ax3.legend(loc='upper left')
	ax3.set_ylim([-170, 200])
	fig3.savefig(Simulation_multiple_directory + "Summary_GS.png", dpi = 200)
	
	#error_rel plot
	y, yerr, truth_temp=np.array(y),np.array(yerr),np.array(truth_temp)
	error = truth_temp-y
	error_rel = []
	for i in range(y.size):
		if y[i-1]<=truth_temp[i-1] :
			error_rel.append(error[i-1]/yerr[1][i-1])
		else :
			error_rel.append(error[i-1]/yerr[0][i-1])
			
	fig3a, ax3a = plt.subplots()
	n, bins, jspckoi = ax3a.hist(error_rel, bins=50, label = 'Simulations')
	ax3a.plot([-1, -1], [0, 10], 'k--', linewidth=2, label = '1$\sigma$ threshold')
	ax3a.plot([1, 1], [0, 10], 'k--', linewidth=2)
	mu_rel, std_rel = norm.fit(error_rel)
	ox = np.linspace(-3, 3, 100)
	p = sum(n*np.diff(bins))*norm.pdf(ox, mu_rel, std_rel)
	ax3a.plot(ox, p, 'r', linewidth=2, label = 'Gaussian fit : mean = %.2f,  std = %.2f' % (mu_rel, std_rel))
	ax3a.set_xlabel('Relative Error [$\sigma$]')
	ax3a.set_ylabel('Count [ ]')
	ax3a.legend()
	ax3a.set_xlim([-3, 3])
	ax3a.set_ylim([0, 10])
	fig3a.savefig(Simulation_multiple_directory + "Error_rel_hist_GS.png", dpi = 200)
	
	#Relative precision
	rel_prec = 1/2.*(yerr[1]+yerr[0])/np.abs(y)*100

	fig3c, ax3c = plt.subplots()
	#bins_tmp = np.linspace(0, 10, 50)
	ax3c.hist(rel_prec, bins=50, label = 'Simulations')
	mu_rel_prec, std_rel_prec =np.mean(rel_prec), np.std(rel_prec)
	p16=np.percentile(rel_prec, 16)
	p84=np.percentile(rel_prec, 84)
	ax3c.plot([mu_rel_prec, mu_rel_prec], [0, 10], 'r', linewidth=2, label = 'mean = %.2f' % mu_rel_prec)
	ax3c.plot([p16, p16], [0, 10], '--k', linewidth=2, label = '16th percentile = %.2f' % p16)
	ax3c.plot([p84, p84], [0, 10], '-.k', linewidth=2, label = '84th percentile = %.2f' % p84)
	ax3c.set_xlabel('Relative Precision [%]')
	ax3c.set_ylabel('Count [ ]')
	ax3c.legend()
	ax3c.set_xlim([0, 40])
	ax3c.set_ylim([0, 10])
	fig3c.savefig(Simulation_multiple_directory + "Relative_precision_hist_GS.png", dpi = 200)
	
	#Relative accuracy 
	rel_acc = error/truth_temp*100
	
	fig3d, ax3d = plt.subplots()
	ax3d.hist(rel_acc, bins=50, label = 'Simulations')
	mu_rel_acc, std_rel_acc =np.mean(rel_acc), np.std(rel_acc)
	p16=np.percentile(rel_acc, 16)
	p84=np.percentile(rel_acc, 84)
	ax3d.plot([mu_rel_acc, mu_rel_acc], [0, 25], 'r', linewidth=2, label = 'mean = %.2f' % mu_rel_acc)
	ax3d.plot([p16, p16], [0, 25], '--k', linewidth=2, label = '16th percentile = %.2f' % p16)
	ax3d.plot([p84, p84], [0, 25], '-.k', linewidth=2, label = '84th percentile = %.2f' % p84)
	ax3d.set_xlabel('Relative Accuracy [%]')
	ax3d.set_ylabel('Count [ ]')
	ax3d.legend()
	ax3d.set_xlim([-60, 40])
	ax3d.set_ylim([0, 20])
	fig3d.savefig(Simulation_multiple_directory + "Relative_accuracy_hist_GS.png", dpi = 200)
	
	#Chi2
	chi2 = error**2./(1/2.*yerr[1]+1/2.*yerr[0])**2.
	
	fig3e, ax3e = plt.subplots()
	ax3e.hist(chi2, bins=50, label = 'Simulations')
	mu_chi2, std_chi2 =np.mean(chi2), np.std(chi2)
	p16=np.percentile(chi2, 16)
	p84=np.percentile(chi2, 84)
	ax3e.plot([mu_chi2, mu_chi2], [0, 50], 'r', linewidth=2, label = 'mean = %.2f' % mu_chi2)
	ax3e.plot([p16, p16], [0, 50], '--k', linewidth=2, label = '16th percentile = %.2f' % p16)
	ax3e.plot([p84, p84], [0, 50], '-.k', linewidth=2, label = '84th percentile = %.2f' % p84)
	ax3e.set_xlabel('$\chi^2$ [ ]')
	ax3e.set_ylabel('Count [ ]')
	ax3e.legend()
	ax3e.set_xlim([0,8 ])
	ax3e.set_ylim([0, 50])
	fig3e.savefig(Simulation_multiple_directory + "Chi2_hist_GS.png", dpi = 200)
	
	
	# Summary stats and plot
	f = (number_pair-len(failed_sim)-len(silver_sample)-len(golden_sample))/float(number_pair) #n submitted / n total
	sum_chi2 = 1/(f*number_pair)*np.sum(chi2)
	sum_P = 1/(f*number_pair)*np.sum(rel_prec/100.)
	sum_A = 1/(f*number_pair)*np.sum(rel_acc/100.)
	X = 0
	for i in chi2 :
		if i<10 :
			X += 1;
	X = X/(number_pair-len(failed_sim)-len(silver_sample)-len(golden_sample))

	print('Golden sample :', f, sum_chi2, sum_P, sum_A, X)
	print('size of the sample :', number_pair-len(failed_sim)-len(silver_sample)-len(golden_sample))
	
	total_f   = np.append(total_f, f)
	total_chi2= np.append(total_chi2, sum_chi2)
	total_P   = np.append(total_P, sum_P)
	#total_chi2= np.append(total_chi2, np.log10(sum_chi2))
	#total_P   = np.append(total_P, np.log10(sum_P))
	total_A   = np.append(total_A, sum_A)
	total_X   = np.append(total_X, X)
	
	
	
	
	
	################################################
	##############  Recap file    ################## 
	################################################
	data ='sample,pair,truth,median,error+,error-,chi2,rel_prec,rel_acc\n'
	j=0
	for i in range(1,number_pair+1):
		if (i in failed_sim) :
			data += '-,' + str(i) + ',' + str(truth[i-1]) + ',-,-,-,-,-,-\n'
		else :
			data += 'F'
			if (i not in silver_sample):
				data += 'S'
			if (i not in golden_sample):
				data += 'G'
			data += (',' + str(i) + ',' + str(truth [i-1]) + ',' + str(median[i-1]) + ',' 
			+ str(error_up[i-1]) + ',' + str(error_down[i-1]) + ',' + str(all_chi2[j]) + ','
			+ str(all_P[j]) + ',' +  str(all_A[j]) +'\n')
			j+=1
	with open(Simulation_multiple_directory + 'Recap.csv', 'w') as f :
		f.write(data)
		f.close()
		
	summary = 'name, f, chi2, P, A, X\n'
	f, chi2, P, A, X = total_f, total_chi2, total_P, total_A, total_X
	for i in range(len(author)):
		summary += (str(author[i]) + ',' + str(f[i]) + ',' + str(chi2[i]) + ',' + str(P[i]) 
		+ ',' + str(A[i]) + ',' + str(X[i]) + '\n')
	with open(Simulation_multiple_directory + 'Full_Recap.csv', 'w') as _f :
		_f.write(summary)
		_f.close()
	
	
	
	################################################
	################## Final plot ################## 
	################################################	
	#Do the plot
	bx1=plt.subplot(4,4,1)
	rect = patches.Rectangle((0.3,0), 1, 0.03, facecolor='Gainsboro')
	#rect = patches.Rectangle((0.3,-2), 1, 2+np.log10(0.03), facecolor='Gainsboro')
	rect.set_alpha(0.3)
	bx1.add_patch(rect)
	rect = patches.Rectangle((0.5,0), 1, 0.03, facecolor='Grey')
	#rect = patches.Rectangle((0.5,-2), 1, 2+np.log10(0.03), facecolor='Grey')
	rect.set_alpha(0.3)
	bx1.add_patch(rect)
	for i in range(f.size):
	    bx1.scatter(f[i], P[i], color=color[i], marker = mark[i])
	plt.setp(bx1.get_xticklabels(), visible=False)
	bx1.set_ylabel('$P$')
	#bx1.set_ylabel('$\log10P$')

	bx2=plt.subplot(4,4,5,sharex=bx1)
	rect = patches.Rectangle((0.3,-0.03), 1, 0.06, facecolor='Gainsboro')
	rect.set_alpha(0.3)
	bx2.add_patch(rect)
	rect = patches.Rectangle((0.5,-0.03), 1, 0.06, facecolor='Grey')
	rect.set_alpha(0.3)
	bx2.add_patch(rect)
	for i in range(f.size):
	    bx2.scatter(f[i], A[i], color=color[i], marker = mark[i])
	plt.setp(bx2.get_xticklabels(), visible=False)
	bx2.set_ylabel('A')


	bx3=plt.subplot(4,4,6,sharey=bx2)
	rect = patches.Rectangle((0,-0.03), 0.03, 0.06, facecolor='Grey')
	rect.set_alpha(0.3)
	bx3.add_patch(rect)
	for i in range(f.size):
	    bx3.scatter(P[i], A[i], color=color[i], marker = mark[i])
	plt.setp(bx3.get_xticklabels(), visible=False)
	plt.setp(bx3.get_yticklabels(), visible=False)

	bx4=plt.subplot(4,4,9,sharex=bx1)
	rect = patches.Rectangle((0.3,0), 1, 1.5, facecolor='Gainsboro')
	rect.set_alpha(0.3)
	bx4.add_patch(rect)
	rect = patches.Rectangle((0.5,0), 1, 1.5, facecolor='Grey')
	rect.set_alpha(0.3)
	bx4.add_patch(rect)
	for i in range(f.size):
	    bx4.scatter(f[i], chi2[i], color=color[i], marker = mark[i])
	plt.setp(bx4.get_xticklabels(), visible=False)
	bx4.set_ylabel('$\chi^2$')
	#bx4.set_ylabel('$\log_{10}\chi^2$')

	bx5=plt.subplot(4,4,10,sharex=bx3,sharey=bx4)
	rect = patches.Rectangle((0,0), 0.03, 1.5, facecolor='Grey')
	rect.set_alpha(0.3)
	bx5.add_patch(rect)
	for i in range(f.size):
	    bx5.scatter(P[i], chi2[i], color=color[i], marker = mark[i])
	plt.setp(bx5.get_xticklabels(), visible=False)
	plt.setp(bx5.get_yticklabels(), visible=False)

	bx6=plt.subplot(4,4,11,sharey=bx4)
	rect = patches.Rectangle((-0.03,0), 0.06, 1.5, facecolor='Grey')
	rect.set_alpha(0.3)
	bx6.add_patch(rect)
	for i in range(f.size):
	    bx6.scatter(A[i], chi2[i], color=color[i], marker = mark[i])
	plt.setp(bx6.get_xticklabels(), visible=False)
	plt.setp(bx6.get_yticklabels(), visible=False)

	bx7=plt.subplot(4,4,13,sharex=bx1)
	for i in range(f.size):
	    bx7.scatter(f[i], X[i], color=color[i], marker = mark[i])
	bx7.set_xlabel('f')
	bx7.set_ylabel('X')

	bx8=plt.subplot(4,4,14,sharex=bx3,sharey=bx7)
	for i in range(f.size):
	    bx8.scatter(P[i], X[i], color=color[i], marker = mark[i])
	plt.setp(bx8.get_yticklabels(), visible=False)
	bx8.set_xlabel('P')
	#bx8.set_xlabel('$\log_{10}P$')

	bx9=plt.subplot(4,4,15,sharex=bx6,sharey=bx7)
	for i in range(f.size):
	    bx9.scatter(A[i], X[i], color=color[i], marker = mark[i])
	plt.setp(bx9.get_yticklabels(), visible=False)
	bx9.set_xlabel('A')
	
	bx10=plt.subplot(4,4,16,sharey=bx7)
	for i in range(f.size):
	    bx10.scatter(chi2[i], X[i], color=color[i], marker = mark[i])
	plt.setp(bx10.get_yticklabels(), visible=False)
	bx10.set_xlabel('$\chi^2$')
	#bx10.set_xlabel('$\log_{10}\chi^2$')

	#set ticks and limit
	bx1.set_xlim([-0.02,1.02])
	bx1.set_xticks([0.2, 0.4, 0.6, 0.8])
	bx1.set_ylim([0, 0.20])
	bx1.set_yticks([0.05, 0.10, 0.15])
	#bx1.set_ylim([-2,0])
	#bx1.set_yticks([-1.5, -1.0, -0.5])
	
	bx2.set_ylim([-0.05, 0.05])
	bx2.set_yticks([-0.03, -0.01, 0.01, 0.03])
	
	bx3.set_xlim([0, 0.20])
	bx3.set_xticks([0.05, 0.10, 0.15])
	#bx3.set_xlim([-2, 0])
	#bx3.set_xticks([-1.5, -1.0, -0.5])
	
	bx4.set_ylim([0, 2])
	bx4.set_yticks([0.5, 1, 1.5])
	#bx4.set_ylim([-1.5, 1.5])
	#bx4.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
	
	bx6.set_xlim([-0.05, 0.05])
	bx6.set_xticks([-0.03, 0, 0.03])

	bx7.set_ylim([0.90, 1.005])
	bx7.set_yticks([0.92, 0.94, 0.96, 0.98])
	
	bx10.set_xlim([0, 2])
	bx10.set_xticks([0.5, 1, 1.5])	
	#bx10.set_xlim([-1.5, 1.5])
	#bx10.set_xticks([-1.0, 0.0, 1.0])	

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.legend(author, ncol=2,  bbox_to_anchor=(1.4,4), frameon = False)
	plt.suptitle('Results for TDC1 rung3')
	plt.savefig(Simulation_multiple_directory + 'Final_Plot.png', dpi = 400)

	

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

