import os
import argparse as ap
import pycs

def main(lensname,dataname, data_dir = '../data/'):
    rdbfile = data_dir + lensname + '_' + dataname+ '.rdb'

    with open(rdbfile, 'r') as f:
        header = f.readline()

    header = header.split('\t')
    lcs = []

    magerr = 5

    if "mag_A1" in header :
        lcs.append(pycs.gen.lc.rdbimport(rdbfile, 'A1', 'mag_A1', 'magerr_A1_%i'%magerr, dataname))
    if "mag_B" in header :
        lcs.append(pycs.gen.lc.rdbimport(rdbfile, 'B', 'mag_B', 'magerr_B_%i'%magerr, dataname))
    if "mag_C" in header :
        lcs.append(pycs.gen.lc.rdbimport(rdbfile, 'C', 'mag_C', 'magerr_C_%i'%magerr, dataname))
    if "mag_D" in header:
        lcs.append(pycs.gen.lc.rdbimport(rdbfile, 'D', 'mag_D', 'magerr_D_%i'%magerr, dataname))

    pycs.gen.util.multilcsexport(lcs, data_dir + lensname + '_' + dataname+ '_reformated.rdb')





if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Reformat the rdb file from COSMOULINE output to something readable by this pipeline. File must have the a name like OBJECT_DATANAME.rdb.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_lensname = "name of the lens to process"
    help_dataname = "name of the data set to process (Euler, SMARTS, ... )"
    help_data_dir = "name of the data directory"
    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument(dest='dataname', type=str,
                        metavar='dataname', action='store',
                        help=help_dataname)
    parser.add_argument('--dir', dest='data_dir', type=str,
                        metavar='', action='store', default='../data/',
                        help=help_data_dir)
    args = parser.parse_args()
    main(args.lensname, args.dataname, data_dir=args.data_dir)
