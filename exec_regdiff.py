
import dill as pkl
import argparse as ap
import pycs, os

def main(pickle_name, is_copy, path):

    os.chdir(path)
    if is_copy :
        simset_copy, lcs, simoptfct, kwargs, opts, tsrand = pkl.load(pickle_name)
        pycs.sim.run.multirun(simset_copy, lcs, simoptfct, kwargs_optim=kwargs,
                          optset=opts, tsrand=tsrand)
    else :
        simset_mock, lcs, simoptfct, kwargs, opts, tsrand = pkl.load(pickle_name)
        pycs.sim.run.multirun(simset_mock, lcs, simoptfct, kwargs_optim=kwargs,
                              optset=opts, tsrand=tsrand, keepopt=True)


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="",
                               formatter_class=ap.RawTextHelpFormatter)
    help_pickle_path = ""
    help_is_copy = ""
    help_path = ""

    parser.add_argument(dest='pickle_path', type=str,
                        metavar='pickle_path', action='store',
                        help=help_pickle_path)
    parser.add_argument(dest='is_copy', type=bool,
                        metavar='is_copy', action='store',
                        help=help_is_copy)
    parser.add_argument(dest='path', type=str,
                        metavar='path', action='store',
                        help=help_path)

    args = parser.parse_args()
    main(args.pickle_path,args.is_copy,args.path)