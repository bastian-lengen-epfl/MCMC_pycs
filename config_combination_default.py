#####################
#  Configuration file for combining data set
#####################

full_lensname =''
testmode = True #Set to False for more precision

combi_name = 'ECAM+C2+SMARTS'  #give a name to your combination
data_sets = ['ECAM','C2','SMARTS']  #select the data sets
marg_to_combine =['marginalisation_final' for sets in data_sets] #give the name of the marginalisation to combine
sigma_to_combine=[0. for sets in data_sets]  #give the corresponding sigma used for the marginalisation

#Additionnal spline marginalisation to plots (only if show_spline = True):
marg_to_combine_spline =['marginalisation_spline' for sets in data_sets] #give the name of the marginalisation to combine
sigma_to_combine_spline=[0.5 for sets in data_sets]

#Additionnal regdiff marginalisation to plots (only if show_spline = True):
marg_to_combine_spline =['marginalisation_regdiff' for sets in data_sets] #give the name of the marginalisation to combine
sigma_to_combine_spline=[0.5 for sets in data_sets]
