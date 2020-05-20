import json

#ECAM ------------------------------------------------------------------------------------------------------------------
output_file_ECAM = '../config/preset_regdiff_ECAM.txt'
output_file_WFI = '../config/preset_regdiff_WFI.txt'
output_file_SMARTS = '../config/preset_regdiff_SMARTS.txt'
f = open(output_file_ECAM, 'w')
g = open(output_file_WFI, 'w')
h = open(output_file_SMARTS, 'w')

param_set1 = {
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.7,
"amp": 0.5,
"scale":200.0,
"errscale":20.0
}

param_set2 = {
"covkernel":'matern',
"pointdensity": 2,
"pow": 2.2,
"amp": 0.4,
"scale":200.0,
"errscale":25.0
}

param_set3 = {
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.5,
"amp": 0.4,
"scale":200.0,
"errscale":20.0
}

param_set4 = {
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.3,
"amp": 0.2,
"scale":200.0,
"errscale":5.0
}

param_set5 ={
"covkernel":'pow_exp',
"pointdensity": 2,
"pow": 1.8,
"amp": 0.3,
"scale":200.0,
"errscale":5.0
}

#WFI ------------------------------------------------------------------------------------------------------------------

param_set1_WFI = {
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.7,
"amp": 0.5,
"scale":200.0,
"errscale":20.0
}

param_set2_WFI = {
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.8,
"amp": 0.6,
"scale":150.0,
"errscale":15.0
}

param_set3_WFI ={
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.3,
"amp": 0.3,
"scale":150.0,
"errscale":10.0
}

param_set4_WFI ={
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.5,
"amp": 0.4,
"scale":250.0,
"errscale":25.0
}

param_set5_WFI ={
"covkernel":'pow_exp',
"pointdensity": 2,
"pow": 1.9,
"amp": 0.7,
"scale":250.0,
"errscale":25.0
}

#SMARTS ------------------------------------------------------------------------------------------------------------------

param_set1_SMARTS = {
"covkernel":'matern',
"pointdensity": 2,
"pow": 2.2,
"amp": 0.5,
"scale":200.0,
"errscale":25.0
}

param_set2_SMARTS = {
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.8,
"amp": 0.7,
"scale":200.0,
"errscale":25.0
}

param_set3_SMARTS ={
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.9,
"amp": 0.6,
"scale":200.0,
"errscale":20.0
}

param_set4_SMARTS ={
"covkernel":'matern',
"pointdensity": 2,
"pow": 1.3,
"amp": 0.3,
"scale":150.0,
"errscale":10.0
}

param_set5_SMARTS ={
"covkernel":'pow_exp',
"pointdensity": 2,
"pow": 1.7,
"amp": 0.7,
"scale":300.0,
"errscale":25.0
}

json.dump([param_set1,param_set2,param_set3,param_set4,param_set5], f)
json.dump([param_set1_WFI,param_set2_WFI,param_set3_WFI,param_set4_WFI,param_set5_WFI], g)
json.dump([param_set1_SMARTS,param_set2_SMARTS,param_set3_SMARTS,param_set4_SMARTS,param_set5_SMARTS], h)
f.close()
g.close()
h.close()
