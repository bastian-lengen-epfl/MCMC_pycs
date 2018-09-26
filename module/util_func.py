import sys
import numpy as np
from inspect import getsource
from textwrap import dedent
import json

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
				out_kw.append("_pd%i_ck%s_pow%.1f_amp%.1f_sc%i_errsc%i_" %(d['pointdensity'], d['covkernel'], d['amp'],d['pow'], d['scale'], d['errscale']))
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