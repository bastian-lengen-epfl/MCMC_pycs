import matplotlib.pyplot as plt
import numpy as np
import pycs
import scipy.signal as sc

def tweakml_PS(lcs, spline, B, f_min = 1/300.0,psplot=False, save_figure_folder = None,  verbose = False, interpolation = 'linear', A_correction = 1.0):
    for l in lcs:
        if l.ml == None:
            print "WARNING, curve %s has no ML to tweak ! I won't tweak anything." % (str(l))
            continue
        # No ! Then we just add some flat spline
        # pycs.gen.splml.addtolc(l) # knotstep has no imortantce

        elif l.ml.mltype != "spline":
            print "WARNING, I can only tweak SplineML objects, curve %s has something else !  I won't tweak anything." % (str(l))
            continue

        name = "ML(%s)" % (l.object)
        ml_spline = l.ml.spline.copy()
        np.random.seed() #this is to reset the seed when using multiprocessing

        rls = pycs.gen.stat.subtract([l], spline)[0]
        target_std = pycs.gen.stat.resistats(rls)['std']
        target_nruns = pycs.gen.stat.resistats(rls)['nruns']

        x = rls.jds
        y = rls.mags
        n = len(x)
        start = x[0]
        stop = x[-1]
        span = stop - start
        sampling = span / n

        samples =  int(span) * 5  #number of samples you want in the generated noise, the final curve is interpolated from this
        if samples%2 ==1 :
            samples -= 1
        samplerate = 1 # don't touch this, add more sample if you want

        freqs_noise = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
        freqs_data = np.linspace(f_min, B* 1 / (sampling * 2.0), 10000)
        pgram = sc.lombscargle(x, y, freqs_data)

        if verbose :
            print "Time Span of your lightcurve : %i days"%span
            print "Average sampling of the curve [day] :", sampling
            print "Nymquist frequency [1/day]:", 1 / (sampling * 2.0)
            print "min max, lenght frequency of noise: ", np.min(freqs_noise), np.max(freqs_noise), len(freqs_noise)
            print "min max, lenght frequency of data: ", np.min(freqs_data), np.max(freqs_data), len(freqs_data)
            print "NUmber of samples generated :",samples

        band_noise = band_limited_noise_withPS(freqs_data, len(freqs_data)*pgram, samples=samples, samplerate=samplerate) #generate the noie with a PS from the data
        x_sample = np.linspace(start, stop, samples)

        noise_lcs_band = pycs.gen.lc.lightcurve()
        noise_lcs_band.jds = x_sample
        noise_lcs_band.mags = band_noise

        #Rescaling of the noise :
        generated_std = pycs.gen.stat.resistats(noise_lcs_band)['std']
        Amp = target_std / generated_std
        if verbose :
            print "required amplification :", Amp
            print "Additionnal A correction :", A_correction
        band_noise_rescaled = band_limited_noise_withPS(freqs_data, len(freqs_data)* Amp * pgram * A_correction, samples=samples, samplerate=samplerate)
        noise_lcs_rescaled = pycs.gen.lc.lightcurve()
        noise_lcs_rescaled.jds = x_sample
        noise_lcs_rescaled.mags = band_noise_rescaled

        #resampling of the generated noise :
        noise_lcs_resampled = interpolate(rls, noise_lcs_rescaled, interpolate=interpolation)
        if verbose :
            print "resampled :", pycs.gen.stat.resistats(noise_lcs_resampled)
            print "target : ", pycs.gen.stat.resistats(rls)


        source = pycs.sim.src.Source(ml_spline, name=name, sampling=span/float(samples))
        if len(noise_lcs_rescaled) != len(source.imags): #weird error can happen for some curves due to round error...
            print "Warning : round error somewhere, I will need to change a little bit the sampling of your source, but don't worry, I can deal with that."
            source.sampling = float(source.jdmax - source.jdmin) / float(len(noise_lcs_rescaled))
            source.ijds = np.linspace(source.jdmin, source.jdmax, float(len(noise_lcs_rescaled)))
            source.imags = source.inispline.eval(jds=source.ijds)

        source.imags += noise_lcs_rescaled.mags
        newspline = source.spline()
        l.ml.replacespline(newspline) # replace the previous spline with the tweaked one...

        if psplot :
            pgram_resampled = sc.lombscargle(noise_lcs_resampled.jds, noise_lcs_resampled.mags, freqs_data)
            fig4 = plt.figure(4)
            plt.plot(freqs_data, pgram, label='original')
            plt.plot(freqs_data, pgram_resampled, label='rescaled and resampled')
            plt.xlabel('Frequencies [1/days]')
            plt.ylabel('Power')
            plt.legend(loc='best')
            if save_figure_folder == None :
                plt.show()
                pycs.gen.stat.plotresiduals([[noise_lcs_resampled]])
                pycs.gen.stat.plotresiduals([[rls]])
            else :
                fig4.savefig(save_figure_folder + 'PS_plot_%s.png'%l.object)
                pycs.gen.stat.plotresiduals([[noise_lcs_resampled]], filename=save_figure_folder + 'resinoise_generated_%s.png'%l.object)
                pycs.gen.stat.plotresiduals([[rls]], filename=save_figure_folder + 'resinoise_original_%s.png'%l.object)




def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np + 1] *= phases
    f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def band_limited_noise_withPS(freqs, PS, samples=1024, samplerate=1):
    freqs_noise = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    PS_interp = np.interp(freqs_noise, freqs, PS, left=0., right=0.)

    f = np.ones(samples) * PS_interp
    return fftnoise(f)

def find_closest(a, x):
    return np.min(np.abs(x - a)), np.argmin(np.abs(x - a))

def interpolate(x1, x2, interpolate='nearest'):
    # take two light curve object and return the reasmpling of the second at the time of the first one

    if interpolate == 'nearest':
        new = x1.copy()
        new_mags = []
        for i, d in enumerate(x1.jds):
            distance, ind = find_closest(d, x2.jds)
            new_mags.append(x2.mags[ind])

        new.jds = x1.jds.copy()
        new.mags = np.asarray(new_mags)

        return new

    elif interpolate == 'linear':
        new = x1.copy()
        new.mags = np.interp(new.jds, x2.jds, x2.mags, left=0., right=0.)
        return new
