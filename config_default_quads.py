#####################
#  Configuration file
#####################
import os, sys
import pycs
import numpy as np
from module import util_func as ut

askquestions = False
display = False

### optimisation function

# for now, we assume that we use the same function to draw the initial condition and run the optimiser.
optfctkw = "spl1" #function you used to optimise the curve at the 1st plase (in the script 2a), it should stay at spl1
simoptfctkw = "spl1" #function you want to use to optimise the mock curves, currently support spl1 and regdiff

# spl1
knotstep = [35] #give a list of the parameter you want

#regdiff params:
covkernel = ['matern']  # can be matern, pow_exp or gaussian
pointdensity = [2]
pow = [2.2]
amp = [0.5]
scale = [200.0]
errscale = [25.0]

#initial guess :
timeshifts = [0,8.,0.0,-14]
magshifts =  [0,0,0,0]


# spldiff
pointdensity_spldiff = 4
knotstep_spldiff = 25

# dispersion
interpdist = 30

### Run parameters

## draw
# copies
ncopy = 10 #number of copy per pickle
ncopypkls = 20 #number of pickle

# mock
nsim = 20 #number of copy per pickle
nsimpkls = 20 #number of pickle
truetsr = 5.0  # Range of true time delay shifts when drawing the mock curves
tsrand = 5.0  # Random shift of initial condition for each simulated lc in [initcond-tsrand, initcond+tsrand]

## sim
run_on_copies = True
run_on_sims = True


### MICROLENSING ####
mltype = "splml"  # splml or polyml
mllist = [0,1,2,3]  # Which lcs do you want to attach ml to ?
mlname = 'splml'
forcen = False # False for Maidanak, True for the other, if true I doesn't use mlknotstep
mlknotsteps = [150]# 0 means no microlensing...
#Unused :
nmlspl = 2  #nb_knot - 1, used only if forcen == True
mlbokeps = 88 #  min spacing between ml knots, used only if forcen == True


###### TWEAK ML #####
#Noise generator for the mocks light curve, script 3a :
tweakml_name = 'PSO_PS_test' #give a name to your tweakml, change the name if you change the type of tweakml, avoid to have _opt_ in your name !
tweakml_type = 'PS_from_residuals' #choose either colored_noise or PS_from_residuals
shotnoise_type = None #Select among [None, "magerrs", "res", "mcres", "sigma"] You should have None for PS_from_residuals

find_tweak_ml_param = True #To let the program find the parameters for you, if false it will use the lines below :
colored_noise_param = [[-2.95,0.001],[-0.5,0.511],[-0.1,0.510],[-2.95,0.001]] #give your beta and sigma parameter for colored noise, used only if find_tweak_ml == False
PS_param_B = [[1.0],[1.0],[1.0],[1.0]] #if you don't want the algorithm fine tune the high cut frequency (given in unit of Nymquist frequency)


#remember to use shotnoise = None for PS_from_residuals and 'magerrs' or 'mcres' for colored_noise

#if you chose to optimise the tweakml automatically, you might want to change this
optimiser = 'PSO' # choose between PSO, MCMC or GRID or DIC
max_core = 8 #None will use all the core available
n_curve_stat =2 # Number of curve to compute the statistics on, (the larger the better but it takes longer... 16 or 32 are good, 8 is still OK) .
n_particles = 1 #this is use only in PSO optimser
n_iter = 1 #number of iteration in PSO or MCMC
mpi = False # if you want to use MPI for the PSO
grid = np.linspace(0.1,1,10) #this is use in the GRID optimiser
max_iter = 10 # this is used in the DIC optimiser, 10 is usually enough.


###### SPLINE MARGINALISATION #########
# Chose the parameters you want to marginalise on for the spline optimiser. Script 4b.
name_marg_spline = 'marginalisation_noise' #choose a name for your marginalisation
tweakml_name_marg_spline = ['PS_noise', 'colored_noise','colored_noise_magerrs']
knotstep_marg = knotstep #parameters to marginalise over, give a list or just select the same that you used above to marginalise over all the available parameters
mlknotsteps_marg = mlknotsteps

###### REGDIFF MARGINALISATION #########
# Chose the parameters you want to marginalise on for the regdiff optimiser. Script 4c.
name_marg_regdiff = 'marginalisation_regdiff'
tweakml_name_marg_regdiff = ['PS_noise']
auto_all = False #set this flag to True (recommanded) and it will use all the available regdiff simulation (all of the line below are ignored)
knotstep_marg_regdiff = knotstep
mlknotsteps_marg_regdiff = mlknotsteps
covkernel_marg = covkernel #parameters to marginalise over, give a list or just select the same that you used above to marginalise over all the available parameters
pointdensity_marg = pointdensity
pow_marg = pow
amp_marg = amp
scale_marg = scale
errscale_marg = errscale

#other parameteres for regdioff and spline marginalisation :
testmode = True
delay_labels = ["AB", "AC", "AD", "BC", "BD" , "CD"]
sigmathresh = 0   #0 is a true marginalisation, choose 1000 to take the most precise.

###### MARGGINALISE SPLINE AND REGDIFF TOGETHER #######
#choose here the marginalisation you want to combine in script 4d, it will also use the sigmathresh:
name_marg_list = ['marginalisation_1','marginalisation_2']
new_name_marg = 'marg_12'


#TODO: implement a check function to assert that the ml parameters correspond to the mlname, if mlname already exists !

if optfctkw == "regdiff" or simoptfctkw == "regdiff":
	from pycs import regdiff

### Functions definition

def spl1(lcs, **kwargs):
	# spline = pycs.spl.topopt.opt_rough(lcs, nit=5)
	spline = pycs.spl.topopt.opt_fine(lcs, knotstep=kwargs['kn'], bokeps=kwargs['kn']/3.0, nit=5, stabext=100)
	return spline

def regdiff(lcs, **kwargs): #knotstep is not used here but this is made to have the same number of argument as the other optimiser...
	return pycs.regdiff.multiopt.opt_ts(lcs, pd=kwargs['pointdensity'], covkernel=kwargs['covkernel'], pow=kwargs['pow'],
										amp=kwargs['amp'], scale=kwargs['scale'], errscale=kwargs['errscale'], verbose=True, method="weights")

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
	regdiffparamskw = ut.generate_regdiff_regdiffparamskw(pointdensity,covkernel, pow, amp, scale, errscale)

data = os.path.join(module_directory+'pkl/', "%s_%s.pkl" % (lensname, dataname))

combkw = [["%s_ks%i_%s_ksml_%i" %(optfctkw, knotstep[i], mlname,mlknotsteps[j]) for j in range(len(mlknotsteps))]for i in range(len(knotstep))]
combkw = np.asarray(combkw)

simset_copy = "copies_n%i" % (int(ncopy * ncopypkls))
simset_mock = "mocks_n%it%i_%s" % (int(nsim * nsimpkls), truetsr,tweakml_name)

if simoptfctkw == "regdiff":
	kwargs_optimiser_simoptfct = ut.get_keyword_regdiff(pointdensity, covkernel, pow, amp, scale, errscale)
	optset = [simoptfctkw + regdiffparamskw[i] + 't' + str(int(tsrand)) for i in range(len(regdiffparamskw))]
elif simoptfctkw == 'spl1':
	optset = [simoptfctkw + 't' + str(int(tsrand))]
else :
	print 'Error : I dont recognize your simoptfctkw, please use regdiff or spl1'
	sys.exit()


#TODO : code kwargs transmission to the optimiser