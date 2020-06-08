# coding=utf-8

import sys 
import os
import argparse as ap
import subprocess
import itertools
import numpy as np
import py_compile

def main(name, name_type, number_pair = 1, work_dir = './'): #Make sure the directory in the data_dir has the same name
	config_directory = work_dir + "config/"
	multiple_config_directory = config_directory + "multiple/"
	data_directory = work_dir + "data/"
	guess_directory = data_directory + name + "/guess/"
	dataname = 'ECAM'
	Simulation_directory = work_dir + "Simulation/"
	Simulation_multiple_directory = Simulation_directory + "multiple/" + name + "_double/"
	if not os.path.exists(Simulation_directory + "multiple/"):
		print "I will create the multiple simulation directory for you ! "
		os.mkdir(Simulation_directory + "multiple/")
	if not os.path.exists(Simulation_multiple_directory):
		os.mkdir(Simulation_multiple_directory)
	
	### Open the initial guess folder with the name guess_name.txt in the data/name/guess directory
	### The folder has to be the same form as a csv file ie. name value \n name2 value 2 \n...
	### Take a random guess with a normal of mean=initial_guess, var=5
	guess = []
	with open(guess_directory + 'guess_' + name + '.txt','r') as f:
		Lines=f.readlines()
		for line in Lines :
			guess.append(np.random.normal(float(line.partition(' ')[2]), 5, 1)[0])

	

	### Open the multiple_config 
	with open(multiple_config_directory + 'config_multiple_' + name + '.py', 'r') as f:
		Lines=f.readlines()
		knotstep = Lines[5]
		preselection = Lines[6]
		ncopy = Lines[10]
		ncopypkls = Lines[11]
		nsim = Lines[14]
		nsimpkls = Lines[15]
		mlknopsteps = Lines[18]
		# Update the guess with the sign of the config_multiple file
		if (int(Lines[21][6:9])==-1):
			guess = [-x for x in guess]
		elif (int(Lines[21][6:9])!=1):
			print('ERROR : Make sure the sign of the config_multiple file is +1 or -1')
			sys.exit()	
			
	### Save the guesses
	with open(Simulation_multiple_directory + 'post_gaussian_guess.txt', 'w') as f :
		i=1
		for x in guess :
			f.write('Guess' + str(i) + ':' + str(x) + '\n')
			i+=1
		f.close()
	
	### Create the config files for each pairs ###
	lens_name = []	
	for i in range(1,number_pair+1):
		lens_name.append(name + '_' + name_type + '_' + 'pair%i'%i)
		try :
			print([sys.executable, '1_create_dataset.py', lens_name[i-1], dataname])
			subprocess.call([sys.executable, '1_create_dataset.py', lens_name[i-1], dataname])
			print("Sucessfully created the config file for " + lens_name[i-1] )
		except : 
			print("Error in script 1_create_dataset.py for" + lens_name[i-1])
			sys.exit()
		
		line_count = 0
		tmp = ''
		with open(config_directory + 'config_' + lens_name[i-1] + '_ECAM.py', 'r') as f :
			Lines=f.readlines()
			for line in Lines :
				line_count += 1
				if (line_count == 10) :
					tmp += 'full_lensname =\'' + lens_name[i-1] + '\'\n'
				elif (line_count == 14) :
					tmp += 'timeshifts = ut.convert_delays2timeshifts([%f])#give the estimated AB delay\n'%guess[i-1]
				elif (line_count == 29) :
					tmp += knotstep
				elif (line_count == 34) :
					tmp += preselection
				elif (line_count == 46) :
					tmp += ncopy
				elif (line_count == 47) :
					tmp += ncopypkls
				elif (line_count == 50) :
					tmp += nsim
				elif (line_count == 51) :
					tmp += nsimpkls
				elif (line_count == 64) :
					tmp += mlknopsteps
				else :
					tmp += line

		with open(config_directory + 'config_' + lens_name[i-1] + '_ECAM.py', 'w') as f :
			f.write(tmp)
			f.close()
		print('Done updating the file : ' + config_directory + 'config_' + lens_name[i-1] + '_ECAM.py')
		### update the pyc file aswell ###
		py_compile.compile(config_directory + 'config_' + lens_name[i-1] + '_ECAM.py')

	
	
	

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
