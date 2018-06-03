
import pycs
import dill
from module import tweakml_PS_from_data as twk

execfile("config.py")

for i,kn in enumerate(knotstep) :
    for j, knml in enumerate(mlknotsteps):
        if tweakml_type == 'colored_noise':
            if find_tweak_ml_param == True :
                #do the optimisation here
                pass

            else :
                print "Colored noise : I will add the beta and sigma that you gave in input."
                tweak_ml_list = []
                for k in range(len(lcs_label)):
                    def tweakml_colored(lcs):
                        return pycs.sim.twk.tweakml(lcs, beta=colored_noise_param[k][0],
                                                       sigma=colored_noise_param[k][1], fmin=1.0 / 50.0, fmax=0.2, psplot=False)


                    tweak_ml_list.append(tweakml_colored)

                f = open(lens_directory + combkw[i,j] + '/tweakml_' + tweakml_name + '.dill','w')
                dill.dump(tweak_ml_list,f)


        elif tweakml_type == 'PS_from_residuals':
            if find_tweak_ml_param == True :
                #do the optimisation here
                pass

            else :
                print "Noise from Power Spectrum of the data : I use PS_param that you gave in input."
                tweak_ml_list = []
                for i in range(len(lcs_label)):
                    def tweakml_PS(lcs):
                        (_ , spline) = pycs.gen.util.readpickle(lens_directory + '%s/initopt_%s_ks%i_ksml%i.pkl' % (combkw[i,j], dataname, kn,knml))
                        return twk.tweakml_PS(lcs, spline, PS_param_B, f_min = 1/300.0,psplot=False, verbose = False, interpolation = 'linear')


                    tweak_ml_list.append(tweakml_PS)

                f = open(lens_directory + combkw[i, j]  + '/tweakml_' + tweakml_name + '.dill','w')
                dill.dump(tweak_ml_list,f )


