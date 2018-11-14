import sys
import numpy as np
from inspect import getsource
from textwrap import dedent
import json
import pickle as pkl
import os
import pycs, copy


def proquest(askquestions):
	"""
	Asks the user if he wants to proceed. If not, exits python.
	askquestions is a switch, True or False, that allows to skip the
	questions.
	"""
	if askquestions:
		answer = raw_input("Tell me, do you want to go on ? (yes/no) ")
		if answer[:3] != "yes":
			sys.exit("Ok, bye.")
		print ""	# to skip one line after the question.

def getdelays(lcs):
	"""Function to obtain the time delay pairs given a list of light curves.
	The function return a list of delay between the pai of images and a list containing the name of the image pairs."""

	delay_pair = []
	delay_name = []

	for i,lci in enumerate(lcs):
		for j,lcj in enumerate(lcs):
			if i >= j :
				continue
			else:
				delay_name.append(lci.object + lcj.object)
				delay_pair.append(lcj.timeshift - lci.timeshift)

	delay_pair = np.asarray(delay_pair)
	delay_name = np.asarray(delay_name)

	return delay_pair,delay_name

def write_func_append(fn, stream, **kwargs):
	#function to write a function in a file, replacing the variable by their value
    fn_as_string = getsource(fn)
    for var in kwargs:
        fn_as_string = fn_as_string.replace(var, kwargs[var])

	fn_as_string = dedent(fn_as_string)

    stream.write('\n' + fn_as_string)

def generate_regdiffparamskw(pointdensity, covkernel, pow, amp, scale, errscale):
	out_kw = []
	for c in covkernel :
		for pts in pointdensity:
			for p in pow :
				for a in amp :
					for s in scale :
						for e in errscale :
							if covkernel == 'gaussian':  # no pow parameter
								out_kw.append("_pd%i_ck%s_amp%.1f_sc%i_errsc%i_" % (pts, c, a, s, e))
							else:
								out_kw.append("_pd%i_ck%s_pow%.1f_amp%.1f_sc%i_errsc%i_" % (
								pts, c, p, a, s, e))
	return out_kw

def read_preselected_regdiffparamskw(file):
	out_kw = []
	with open(file, 'r') as f :
		dic = json.load(f)
		for d in dic :
			if d['covkernel'] == 'gaussian':  # no pow parameter
				out_kw.append("_pd%i_ck%s_amp%.1f_sc%i_errsc%i_" % (d['pointdensity'], d['covkernel'], d['amp'], d['scale'], d['errscale']))
			else:
				out_kw.append("_pd%i_ck%s_pow%.1f_amp%.1f_sc%i_errsc%i_" %(d['pointdensity'], d['covkernel'], d['pow'],d['amp'], d['scale'], d['errscale']))
	return out_kw


def get_keyword_regdiff(pointdensity, covkernel, pow, amp, scale, errscale):
    kw_list = []
    for c in covkernel:
        for pts in pointdensity:
            for p in pow:
                for a in amp:
                    for s in scale:
                        for e in errscale:
                            kw_list.append({'covkernel':c, 'pointdensity':pts, 'pow':p, 'amp':a, 'scale':s, 'errscale':e})
    return kw_list

def get_keyword_regdiff_from_file(file):
	with open(file, 'r') as f :
		kw_list = json.load(f)
	return kw_list

def get_keyword_spline(kn):
    return {'kn' : kn}

def group_estimate(path_list, name_list, delay_labels, colors, sigma_thresh, new_name_marg, testmode = True):
    if testmode:
        nbins = 500
    else:
        nbins = 5000

    group_list = []
    medians_list = []
    errors_up_list = []
    errors_down_list = []
    if len(path_list) != len(name_list):
        raise RuntimeError("Path list and name_list should have he same lenght")
    for p, path in enumerate(path_list):
        if not os.path.isfile(path):
            print "Warning : I cannot find %s. I will skip this one. Be careful !" %path
            continue

        group = pkl.load(open(path, 'rb'))
        group.name = name_list[p]
        group_list.append(group)
        medians_list.append(group.medians)
        errors_up_list.append(group.errors_up)
        errors_down_list.append(group.errors_down)

    #build the bin list :
    medians_list = np.asarray(medians_list)
    errors_down_list = np.asarray(errors_down_list)
    errors_up_list = np.asarray(errors_up_list)
    binslist = []
    for i, lab in enumerate(delay_labels):
        bins = np.linspace(min(medians_list[:,i]) - 10 *min(errors_down_list[:,i]), max(medians_list[:,i]) + 10*max(errors_up_list[:,i]), nbins)
        binslist.append(bins)

    color_id = 0
    for g,group in enumerate(group_list):
        group.plotcolor = colors[color_id]
        group.binslist = binslist
        group.linearize(testmode=testmode)
        color_id += 1
        if color_id >= len(colors):
            print "Warning : I don't have enough colors in my list, I'll restart from the beginning."
            color_id = 0  # reset the color form the beginning


    combined = copy.deepcopy(pycs.mltd.comb.combine_estimates(group_list, sigmathresh=sigma_thresh, testmode=testmode))
    combined.linearize(testmode=testmode)
    combined.name = 'combined $\sigma = %2.2f$'%sigma_thresh
    combined.plotcolor = 'black'
    print "Final combination for marginalisation ", new_name_marg
    combined.niceprint()

    return group_list, combined

def convert_delays2timeshifts(timedelay):
    """
    Convert the time-delays you can measure by eye into time-shifts for the individual curve
    :param timedelay: list of time delays
    :return: list of timeshifts
    """
    timeshift=np.zeros(len(timedelay)+1)
    timeshift[1:]=timedelay
    return timeshift



