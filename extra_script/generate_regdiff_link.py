import os,sys
import argparse as ap
import pickle as pkl
import importlib



def main(lensname,dataname,work_dir='./'):
    os.chdir('..')
    main_path = os.getcwd()
    sys.path.append(work_dir + "config/")
    sys.path.append(work_dir)
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    for a,kn in enumerate(config.knotstep) :
        for  b, knml in enumerate(config.mlknotsteps):
            for c, opts in enumerate(config.optset):
                if config.simoptfctkw == "regdiff":
                    if a == 0 and b == 0:
                        kwargs = config.kwargs_optimiser_simoptfct[c]
                        destpath = os.path.join(main_path, config.lens_directory + config.combkw[a, b] + '/')

                        dir_link = os.path.join(destpath, "sims_%s_opt_%s" % (config.simset_copy, opts))
                        print "Dir link :", dir_link
                        pkl.dump(dir_link, open(os.path.join(config.lens_directory, 'regdiff_copies_link_%s.pkl' % kwargs['name']), 'w'))
                    else :
                        print "Please turn on your simoptfctkw to regdiff."
                        exit()


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Generate the regdiff link for compatibilty of old simulations with the new pipeline.",
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