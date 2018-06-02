import pickle
import matplotlib.pyplot as plt
import numpy as np
import pycs
import scipy.signal as sc
import plot_functions as pltfct
import mcmc_function as fmcmc
import os


source ="pickle"
object = "HE0435"

picklepath = "./"+object+"/save/"

kntstp = 40
ml_kntstep =360
picklename ="opt_spl_ml_"+str(kntstp)+"-"+str(ml_kntstep) + "knt.pkl"
curve = 3

(lcs, spline) = pycs.gen.util.readpickle(picklepath + picklename)

rls = pycs.gen.stat.subtract(lcs, spline)
pycs.sim.draw.saveresiduals(lcs, spline)


sigma = pycs.gen.stat.mapresistats(rls)[curve]['std']

# pycs.gen.stat.plotresiduals([rls])

x = rls[curve].jds
y = rls[curve].mags
errors = rls[curve].magerrs
n = len(x)
span = x[-1] - x[0]
sampling = span / n

print "Span :", span
print "Sampling :", sampling
print "nymquist frequency :", 1/(sampling*2.0)

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



n = x.shape[0] # For normalization of the periodogram
freqs = np.linspace(1/300.0, 1/(sampling*3.0), 10000)
pgram = sc.lombscargle(x, y, freqs)

print "signal with period :", np.max(1/freqs), np.min(1/freqs)
print "frequency step :", freqs[1]-freqs[0]

plt.figure(1)
# plt.plot(freqs, np.sqrt(4*(pgram/normval)))
plt.plot(freqs, pgram)
plt.xlabel('Period [days]')
plt.xlabel('Power')


# noise = band_limited_noise(freqs[0], freqs[-1], samples=len(freqs), samplerate=1)
# noise = fftnoise(np.sqrt(2*n*pgram))
# noise = fftnoise(pgram)

x_sample = np.linspace(x[0],x[-1],len(freqs))

noise_lcs = [pycs.gen.lc.lightcurve()]
noise_lcs[0].jds = x_sample
noise_lcs[0].mags = noise
noise_lcs[0].magerrs = np.zeros(len(freqs))


dt = x_sample[1] - x_sample[0]
print "time step of the generated curve",dt

print "generated :",pycs.gen.stat.mapresistats(noise_lcs)
# pycs.gen.stat.plotresiduals([noise_lcs],filename='resinoise.png')
# pycs.gen.stat.plotresiduals([rls],filename='resinoise.png')

resampled = interpolate_nearest(rls[curve],noise_lcs[0])
resampled[0].magerrs =  np.zeros(len(resampled[0].jds))


print "resampled :", pycs.gen.stat.mapresistats(resampled)
print "original :", pycs.gen.stat.mapresistats(rls)[curve]
print len(resampled[0].mags)

pgram_noise = sc.lombscargle(x, resampled[0].mags, freqs)
# pgram_noise = sc.lombscargle(x, resampled[0].mags, freqs, normalize=True)
# plt.plot(freqs, np.sqrt(4*(pgram_noise/normval)),'r')
plt.plot(freqs, pgram_noise,'r')
plt.xlabel('Period [days]')
plt.xlabel('Power')


plt.figure(3)
plt.plot(freqs, pgram_noise-pgram)
plt.show()

pycs.gen.stat.plotresiduals([resampled])

# plt.show()

nruns = []
zruns = []
sigmas = []
A_vec = np.linspace(0.5,10,10)

# for A in A_vec:
#     noise_lcs[0].mags = A*noise
#     resampled = interpolate_nearest(rls[curve], noise_lcs[0])
#     nruns.append(pycs.gen.stat.mapresistats(resampled)[0]['nruns'])
#     zruns.append(pycs.gen.stat.mapresistats(resampled)[0]['zruns'])
#     sigmas.append(pycs.gen.stat.mapresistats(resampled)[0]['std'])
#
# print np.shape(A_vec)
# print np.shape(nruns)
#
# plt.figure(1)
# plt.plot(A_vec,nruns)
# plt.xlabel('A')
# plt.ylabel('nruns')
#
# plt.figure(2)
# plt.plot(A_vec,zruns)
# plt.xlabel('A')
# plt.ylabel('zruns')
#
# plt.figure(3)
# plt.plot(A_vec,sigmas)
# plt.xlabel('A')
# plt.ylabel('sigmas')
#
# plt.show()