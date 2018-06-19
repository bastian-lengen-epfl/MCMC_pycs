#####################
#  Configuration file
#####################
import os, sys
import pycs
import numpy as np

askquestions = False
display = False

### optimisation function

# for now, we assume that we use the same function to draw the initial condition and run the optimiser.
optfctkw = "spl1" #function you used to optimise the curve at the 1st plase (in the script 2a), it should stay at spl1
simoptfctkw = "regdiff" #function you want to use to optimise the mock curves

# spl1
knotstep = [60,80] #give a list of the parameter you want

#regdiff params:
covkernel = 'matern'  # can be matern, pow_exp or gaussian
pointdensity = 2
pow = 2.2
amp = 0.5
scale = 200.0
errscale = 25.0

#initial guess :
timeshifts = [0,-2.1]
magshifts =  [0,-2.2]


# spldiff
pointdensity_spldiff = 4
knotstep_spldiff = 25

# dispersion
interpdist = 30

### Run parameters

## draw
# copies
ncopy = 10 #number of copy per pickle
ncopypkls = 1 #number of pickle

# mock
nsim = 10 #number of copy per pickle
nsimpkls = 1 #number of pickle
truetsr = 25.0  # Range of true time delay shifts when drawing the mock curves

## sim
run_on_copies = True
run_on_sims = True

tsrand = 25.0  # Random shift of initial condition for each simulated lc in [initcond-tsrand, initcond+tsrand]


### MICROLENSING ####
mltype = "splml"  # splml or polyml
mllist = [0, 1]  # Which lcs do you want to attach ml to ?
mlname = 'splml'
shotnoise_type = "mcres" #Select among [None, "magerrs", "res", "mcres", "sigma"]
forcen = False # False for Maidanak, True for the other, if true I doesn't use mlknotstep
mlknotsteps = [0,500]# 0 means no microlensing...
#Unused :
nmlspl = 2  #nb_knot - 1, used only if forcen == True
mlbokeps = 88 #  min spacing between ml knots, used only if forcen == True


###### TWEAK ML #####
#Noise generator for the mocks light curve :
tweakml_name = 'colored_noise_optim_PSO' #give a name to your tweakml, change the name if you change the type of tweakml
tweakml_type = 'colored_noise' #choose either colored_noise or PS_from_residuals
find_tweak_ml_param = False  #To let the program find the parameters for you
colored_noise_param = [[-1.9,0.1],[-2.,0.2]] #give your beta and sigma parameter for colored noise, used only if find_tweak_ml == False
PS_param_B = [2.0,2.0] #if you don't want the algorithm fine tune the high cut frequency (given in unit of Nymquist frequency)

#remember to use shotnoise = None for PS_from_residuals and 'magerrs' or 'mcres' for colored_noise

#if you chose to optimise the tweakml automatically, you might want to change this
optimiser = 'PSO' # choose between PSO, MCMC or GRID
max_core = None #None will use all the core available
n_curve_stat = 8 # Number of curve to compute the statistics on, (the larger the better but it takes longer...) .
n_particles = 8 #this is use only in PSO optimser
n_iter = 1 #number of iteration in PSO or MCMC
mpi = False # if you want to use MPI for the PSO

###### MARGINALISATION #########
# Chose the parameters you want to marginalise on :
name_marg = 'marginalisation_1' #choose a name for your marginalisation
opt_marg = ['spl','regdiff'] #choose the curve shifting technique to marginalize
kn_marg = [60,80]
knml_marg = [500,1000]
tweakml_name_marg = ['PS_noise_fix']

######MARGGINALISE MARGINALISATION #######
#choose here the marginalisation you want to combine :
name_marg_list = []





#TODO: implement a check function to assert that the ml parameters correspond to the mlname, if mlname already exists !

if optfctkw == "regdiff" or simoptfctkw == "regdiff":
	from pycs import regdiff

### Functions definition

def spl1(lcs, kn):
	spline = pycs.spl.topopt.opt_rough(lcs, nit=5)
	spline = pycs.spl.topopt.opt_fine(lcs, knotstep=kn, bokeps=kn/3.0, nit=5, stabext=100)
	return spline

def regdiff(lcs, kn = None): #knotstep is not used here but this is made to have the same number of argument as the other optimiser...
	return pycs.regdiff.multiopt.opt_ts(lcs, pd=pointdensity, covkernel=covkernel, pow=pow, amp=amp, scale=scale, errscale=errscale, verbose=True, method="weights")

def spldiff(lcs, kn):
	return pycs.spldiff.multiopt.opt_ts(lcs, pd=pointdensity, knotstep=kn, bokeps=kn/3.0)

rawdispersionmethod = lambda lc1, lc2 : pycs.disp.disps.linintnp(lc1, lc2, interpdist=interpdist)
dispersionmethod = lambda lc1, lc2 : pycs.disp.disps.symmetrize(lc1, lc2, rawdispersionmethod)
def disp(lcs):
	return pycs.disp.topopt.opt_full(lcs, rawdispersionmethod, nit=3, verbose=True)


def attachml(lcs,knml):
	if knml == 0 : #I do nothing if there is no microlensing to attach.
		return
	lcmls = [lcs[ind] for ind in mllist]
	knmlvec = [knml for ind in mllist]
	if mltype == 'splml':
		if forcen:
			for lcml, mlknotstep in zip(lcmls, knmlvec):
				pycs.gen.splml.addtolc(lcml, n=nmlspl, bokeps=mlbokeps)
		else:
			for lcml, mlknotstep in zip(lcmls, knmlvec):
				mlbokeps_ad = mlknotstep / 3.0   #maybe change this
				pycs.gen.splml.addtolc(lcml, knotstep=mlknotstep, bokeps=mlbokeps_ad)

	# polynomial microlensing
	if mltype == 'polyml':
		if len(mllist) != len(nparams):
			print "Give me enough nparams, one per lc in mllist!"

			sys.exit()
		if len(mllist) != len(autoseasonsgaps):
			print "Give me enough autoseasongaps, one per lc in mllist!"
			sys.exit()
		for ind, lcml in enumerate(lcmls):
			pycs.gen.polyml.addtolc(lcml, autoseasonsgap=autoseasonsgaps[ind], nparams=nparams[ind])


###### DON'T CHANGE ANYTHING BELOW THAT LINE ######

if optfctkw == "spl1":
	optfct = spl1
	splparamskw = ["ks%i" %knotstep[i] for i in range(len(knotstep))]

if optfctkw == "regdiff": # not used, small haxx to be able to execute 2 to check and 3 using the spl1 drawing
	optfct = regdiff
	splparamskw = "ks%i" % knotstep

if simoptfctkw == "spl1":
	simoptfct = spl1

if simoptfctkw == "regdiff":
	simoptfct = regdiff
	if covkernel == 'gaussian': # no pow parameter
		regdiffparamskw = "_pd%i_ck%s_amp%.1f_sc%i_errsc%i_" % (pointdensity, covkernel, amp, scale, errscale)
	else:
		regdiffparamskw = "_pd%i_ck%s_pow%.1f_amp%.1f_sc%i_errsc%i_" % (pointdensity, covkernel, pow, amp, scale, errscale)
		#TODO : make a grid out of this so that you can optimise everything at once !

data = os.path.join(module_directory+'pkl/', "%s_%s.pkl" % (lensname, dataname))

combkw = [["%s_%s_%s_ksml_%i" %(optfctkw, splparamskw[i], mlname,mlknotsteps[j]) for j in range(len(mlknotsteps))]for i in range(len(knotstep))]
combkw = np.asarray(combkw)

simset_copy = "copies_n%i" % (int(ncopy * ncopypkls))
simset_mock = "mocks_n%it%i_%s" % (int(nsim * nsimpkls), truetsr,tweakml_name)

if simoptfctkw == "regdiff":
	optset = simoptfctkw + regdiffparamskw + 't' + str(int(tsrand))
elif simoptfctkw == "spldiff":
	optset = simoptfctkw + spldiffparamskw + 't' + str(int(tsrand))
elif simoptfctkw == "disp":
	optset = simoptfctkw + dispparamskw + 't' + str(int(tsrand))
else:
	optset = simoptfctkw + 't' + str(int(tsrand))


