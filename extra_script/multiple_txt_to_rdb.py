import os
import argparse as ap
import itertools


def main(name, name_type, number_pair = 1, data_dir = '../data/'):
	txtfile = []
	rdbfile = []
	for i in range(1, number_pair+1) :
		txtfile.append(data_dir + name + '/' + name + '_' + name_type + '_' + 'pair%i'%i + '.txt')
		rdbfile.append(data_dir + name + '_' + name_type + '_' + 'pair%i'%i + '_ECAM.rdb')
	
	for i in range(number_pair) :
		line_skipped = 0
		data = ''
		with open(txtfile[i], 'r') as f:
			Lines = f.readlines()
			for line in Lines :
				line_skipped += 1
				if (line_skipped<=6) : continue
				data += line
		with open(rdbfile[i], 'w') as f:
			f.write("mhjd\tmag_A\tmagerr_A\tmag_B\tmagerr_B\n")
			f.write("====\t=====\t========\t=====\t========\n")
			f.write(data)
			f.close
	


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Reformat the txt file from the Time Delay Challenge to an usable rdb file.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_name = "name of the sample. Make sure the directory has the same name."
    help_name_type = "Type of the data ie double or quad"
    help_number_pair = "number of pair in the rung folder. Make sure the folder have the format name_pair0"
    help_data_dir = "name of the data directory"
    parser.add_argument(dest='name', type=str,
                        metavar='name', action='store',
                        help=help_name)
    parser.add_argument(dest='name_type', type=str,
                        metavar='name_type', action='store',
                        help=help_name_type)
    parser.add_argument(dest='number_pair', type=int,
                        metavar='number_pair', action='store',
                        help=help_number_pair)                    
    parser.add_argument('--dir', dest='data_dir', type=str,
                        metavar='', action='store', default='../data/',
                        help=help_data_dir)
    args = parser.parse_args()
    main(args.name, args.name_type, args.number_pair, data_dir=args.data_dir)
