#!/urs/bin/env python

import os,sys
import numpy as np
import argparse as ap
import pycs
import matplotlib.pyplot as plt

def main(lensname,dataname,work_dir='./'):
	data_directory = work_dir + "data/"
	lc_fig_directory = './../Simulation/' + lensname + '_' + dataname + '/figure/light_curve/'
	
	### Change here 
	mag_shift =  1.19
	time_shift = 108.85
	
	if not os.path.exists(lc_fig_directory):
		os.mkdir(lc_fig_directory)
    
	mhjd, mag_A, magerr_A, mag_B, magerr_B = [],[],[],[],[]
	with open('./../data/' + lensname + '_' + dataname + '.rdb' ,'r') as f:
		Lines=f.readlines()
		linecount = 1
		for line in Lines :
			if linecount <=2 : 
				linecount+=1
			else :

				mhjd.append(float(line.split('\t')[0]))
				mag_A.append(float(line.split('\t')[1]))
				magerr_A.append(float(line.split('\t')[2]))
				mag_B.append(float(line.split('\t')[3]))
				magerr_B.append(float(line.split('\t')[4].replace('\n\r','')))
				linecount+=1
				
	mhjd, mag_A, magerr_A, mag_B, magerr_B = np.array(mhjd), np.array(mag_A), np.array(magerr_A), np.array(mag_B), np.array(magerr_B)		
	
	fig1, ax1 = plt.subplots()
	ax1.errorbar(mhjd, mag_A, magerr_A, marker = '.', markersize = 4, elinewidth = 1, linestyle='none', label = 'A')
	ax1.errorbar(mhjd, mag_B, magerr_B, marker = '.', markersize = 4, elinewidth = 1, linestyle='none', label = 'B')
	ax1.set_xlabel('Days [d]')
	ax1.set_ylabel('Magnitude [ ]')
	ax1.set_title(lensname + ' Light curves')
	ax1.legend()
	fig1.savefig(lc_fig_directory + "initial_lc.png", dpi = 200)
	
	
	fig3, ax3 = plt.subplots()
	ax3.errorbar(mhjd, mag_A, magerr_A, marker = '.', markersize = 4, elinewidth = 1, linestyle='none', label = 'A')
	ax3.errorbar(mhjd+time_shift, mag_B+mag_shift, magerr_B, marker = '.', markersize = 4, elinewidth = 1, linestyle='none', label = 'B')
	ax3.set_xlabel('Days [d]')
	ax3.set_ylabel('Magnitude [ ]')
	ax3.set_title(lensname + ' Light curves')
	ax3.legend()
	fig3.savefig(lc_fig_directory + "mag_and_time_shifted_lc.png", dpi = 200)
	
	
	



if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Fit spline and regdiff on the data.",
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

