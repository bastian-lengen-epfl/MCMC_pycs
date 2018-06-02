import pycs.sim.src
import pycs.gen.stat
from numpy as np
import test_noise as tn
import scipy.signal as sc
import matplotlib.pyplot as plt

def tweakml_PS(lcs, A, B, f_min = 1/300.0,psplot=False, sampling=0.1, verbose = False):
    for l in lcs:
        if l.ml == None:
            print "WARNING, curve %s has no ML to tweak !" % (str(l))
            continue
        # No ! Then we just add some flat spline
        # pycs.gen.splml.addtolc(l) # knotstep has no imortantce

        elif l.ml.mltype != "spline":
            print "WARNING, I can only tweak SplineML objects, curve %s has something else !" % (str(l))
            continue

        name = "ML(%s)" % (l.object)
        spline = l.ml.spline.copy()
        rls = pycs.gen.stat.subtract(l, spline)

        n = len(rls.jds)
        span = rls.jds[-1] - rls.jds[0]
        sampling = span / n

        if verbose :
            print "Time Span of your lightcurve : %i days"%span
            print "Average sampling of the curve :", sampling
            print "Nymquist frequency :", 1 / (sampling * 2.0)


        freqs = np.linspace(f_min, 1 / (sampling * 2.0), 10000)
        pgram = sc.lombscargle(rls.jds, rls.mags, freqs) #compute the Lomb-Scargle Periodogram

        if psplot :
            plt.plot(1 / freqs, np.sqrt(4 * (pgram / n)))
            plt.xlabel('Period [days]')
            plt.xlabel('Power')
            plt.show()


        if psplot == True:
            psspline = pycs.sim.src.PS(source, flux=False)
            psspline.plotcolour = "black"
            psspline.calcslope(fmin=1 / 1000.0, fmax=1 / 100.0)

        source.addplaw2(beta=beta, sigma=sigma, fmin=fmin, fmax=fmax, flux=False, seed=None)


        source.name += "_twk"
        newspline =

        l.ml.replacespline(newspline)

        if psplot == True:
            psnewspline = pycs.sim.src.PS(source, flux=False)
            psnewspline.plotcolour = "red"
            psnewspline.calcslope(fmin=fmin, fmax=fmax)

            pycs.sim.src.psplot([psspline, psnewspline], nbins=50)









def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def find_closest(a, x):
    return np.min(np.abs(x-a)), np.argmin(np.abs(x-a))

def interpolate_nearest(x1,x2):
    # take two light curve object and return the reasmpling of the second at the time of the first one
    new = [pycs.gen.lc.lightcurve()]
    new_mags = []
    for i,d in enumerate(x1.jds) :
        distance, ind = find_closest(d, x2.jds)
        new_mags.append(x2.mags[ind])

    new[0].jds = x1.jds.copy()
    new[0].mags = np.asarray(new_mags)

    return new
