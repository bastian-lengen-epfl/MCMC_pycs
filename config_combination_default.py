#####################
#  Configuration file for combining data set
#####################
import sys, os
import importlib

full_lensname =''
delay_labels = ["AB"]
testmode = True #Set to False for more precision

combi_name = 'ECAM+C2+SMARTS'  #give a name to your combination
data_sets = ['ECAM','C2','SMARTS']  #select the data sets
marg_to_combine =['marginalisation_final' for sets in data_sets] #give the name of the marginalisation to combine
sigma_to_combine=[0. for sets in data_sets]  #give the corresponding sigma used for the marginalisation
sigma_thresh = 0.0 #sigme threshold for the PyCS-sum estimate, leave it to 0 for a true marginalisation

#Additionnal spline marginalisation to plots (only if show_spline = True):
marg_to_combine_spline =['marginalisation_spline' for sets in data_sets] #give the name of the marginalisation to combine
sigma_to_combine_spline=[0.5 for sets in data_sets]

#Additionnal regdiff marginalisation to plots (only if show_spline = True):
marg_to_combine_regdiff =['marginalisation_regdiff' for sets in data_sets] #give the name of the marginalisation to combine
sigma_to_combine_regdiff=[0.5 for sets in data_sets]
