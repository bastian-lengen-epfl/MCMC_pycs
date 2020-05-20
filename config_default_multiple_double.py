###############################
# Multiple Configuration file #
###############################

### SPLINE PARAMETERS ###
knotstep = [45,55,65] #give a list of the parameter you want
preselection_file = 'config/preset_regdiff_ECAM.txt' #'config/preset_regdiff_ECAM.txt'

### RUN PARAMETERS #####
# copies
ncopy = 20 #number of copy per pickle
ncopypkls = 25 #number of pickle

# mock
nsim = 20 #number of copy per pickle
nsimpkls = 40 #number of pickle

### MICROLENSING ####
mlknotsteps = [200,300,400]# 0 means no microlensing...

### SIGN OF THE GUESS
sign = 1 # +1 with the D3CS & PyCS convention, -1 for the opposite.
