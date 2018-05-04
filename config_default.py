#####################
#  Configuration file
#####################
import os, sys
import pycs

lcs_label = ['A','B'] #nb of lightcurves
askquestions = True
display = True

### optimisation function

# for now, we assume that we use the same function to draw the initial condition and run the optimiser.
optfctkw = "spl1"
simoptfctkw = "spl1"

# spl1
knotstep = 100

#regdiff params:
covkernel = 'matern'  # can be matern, pow_exp or gaussian
pointdensity = 2
pow = 2.2
amp = 0.5
scale = 200.0
errscale = 25.0

#initial guess :
timeshifts = [0,0.0]
magshifts =  [0,-2.2]


# spldiff
pointdensity_spldiff = 4
knotstep_spldiff = 25

# dispersion
interpdist = 30

### Run parameters

## draw
# copies
ncopy = 10
ncopypkls = 1

# mock
nsim = 10
nsimpkls = 1
truetsr = 25.0  # Range of true time delay shifts when drawing the mock curves

## sim
run_on_copies = True
run_on_sims = True

tsrand = 25.0  # Random shift of initial condition for each simulated lc in [initcond-tsrand, initcond+tsrand]


### Microlensing
mlname = 'splml'

# general
mltype = "splml"  # splml or polyml
mllist = [0, 1]  # Which lcs do you want to attach ml to ?

forcen = False # False for Maidanak, True for the other
nmlspl = 2  #nb_knot - 1, used only if forcen == True
mlbokeps = 88 #  min spacing between ml knots, used only if forcen == True
# splml
mlknotsteps = [1000, 1000]#, used only if forcen == False

# polyml, not used only if polymol = 'polyml'
nparams = [1, 1]
autoseasonsgaps = [1000, 1000]

def mytweakml1(lcs):
	return pycs.sim.twk.tweakml(lcs, beta=0, sigma=0.0, fmin=1.0/50.0, fmax=0.2, psplot=False)
def mytweakml2(lcs):
	return pycs.sim.twk.tweakml(lcs, beta=0, sigma=0.0, fmin=1.0/50.0, fmax=0.2, psplot=False)
def mytweakml3(lcs):
	return pycs.sim.twk.tweakml(lcs, beta=-7.5, sigma=0.0000000005, fmin=1.0/50.0, fmax=0.2, psplot=False)

mytweakml = [mytweakml1,mytweakml2]
shotnoise_type = "mcres" #Select among [None, "magerrs", "res", "mcres", "sigma"]


#TODO: implement a check function to assert that the ml parameters correspond to the mlname, if mlname already exists !

if optfctkw == "regdiff" or simoptfctkw == "regdiff":
	from pycs import regdiff

### Functions definition

def spl1(lcs, kn):
	# spline = pycs.spl.topopt.opt_rough(lcs, nit=5)
	spline = pycs.spl.topopt.opt_fine(lcs, knotstep=kn, bokeps=knotstep/3.0, nit=5, stabext=100)
	return spline

def regdiff(lcs):
	return pycs.regdiff.multiopt.opt_ts(lcs, pd=pointdensity, covkernel=covkernel, pow=pow, amp=amp, scale=scale, errscale=errscale, verbose=True, method="weights")

def spldiff(lcs):
	return pycs.spldiff.multiopt.opt_ts(lcs, pd=pointdensity, knotstep=knotstep, bokeps=knotstep/3.0)

rawdispersionmethod = lambda lc1, lc2 : pycs.disp.disps.linintnp(lc1, lc2, interpdist=interpdist)
dispersionmethod = lambda lc1, lc2 : pycs.disp.disps.symmetrize(lc1, lc2, rawdispersionmethod)
def disp(lcs):
	return pycs.disp.topopt.opt_full(lcs, rawdispersionmethod, nit=3, verbose=True)


def attachml(lcs):
	lcmls = [lcs[ind] for ind in mllist]
	if mltype == 'splml':
		if forcen:
			for lcml, mlknotstep in zip(lcmls, mlknotsteps):
				pycs.gen.splml.addtolc(lcml, n=nmlspl, bokeps=mlbokeps)
		else:
			for lcml, mlknotstep in zip(lcmls, mlknotsteps):
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

# As there is only one parameter to tweak in spl1 or spl2, I add it directly in the combkw variable
if optfctkw == "spl1":
	optfct = spl1
	splparamskw = "ks%i" % knotstep

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

data = os.path.join('pkl', "%s_%s.pkl" % (lensname, dataname))

combkw = "%s_%s_%s_ksml_%i" % (optfctkw, splparamskw, mlname,mlknotsteps[0])

simset_copy = "copies_%s_n%i" % (dataname, int(ncopy * ncopypkls))
simset_mock = "mocks_%s_n%it%i" % (dataname, int(nsim * nsimpkls), truetsr)

if simoptfctkw == "regdiff":
	optset = simoptfctkw + regdiffparamskw + 't' + str(int(tsrand))
elif simoptfctkw == "spldiff":
	optset = simoptfctkw + spldiffparamskw + 't' + str(int(tsrand))
elif simoptfctkw == "disp":
	optset = simoptfctkw + dispparamskw + 't' + str(int(tsrand))
else:
	optset = simoptfctkw + 't' + str(int(tsrand))


