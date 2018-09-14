import matplotlib.pyplot as plt
import numpy as np
import pycs
import scipy.signal as sc

source ="pickle"
object = "HE0435"
# object = "UM673_Euler"

picklepath = "./"+object+"/save/"

# kntstp = 60
kntstp = 40
# ml_kntstep =500
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
start= x[0]
stop = x[-1]
span = stop - start
sampling = span / n
interpolation = 'linear'

print "Span :", span
print "Sampling period:", sampling
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

def band_limited_noise_withPS(freqs, PS, samples=1024, samplerate=1):
    freqs_noise = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    PS_interp = np.interp(freqs_noise, freqs, PS, left=0., right=0.)

    f = np.ones(samples) * PS_interp
    return fftnoise(f)

def find_closest(a, x):
    return np.min(np.abs(x-a)), np.argmin(np.abs(x-a))

def interpolate(x1,x2, interpolate = 'nearest'):
    # take two light curve object and return the reasmpling of the second at the time of the first one

    if interpolate == 'nearest' :
        new = [x1.copy()]
        new_mags = []
        for i,d in enumerate(x1.jds) :
            distance, ind = find_closest(d, x2.jds)
            new_mags.append(x2.mags[ind])

        new[0].jds = x1.jds.copy()
        new[0].mags = np.asarray(new_mags)

        return new

    elif interpolate == 'linear' :
        new = [x1.copy()]
        new[0].mags = np.interp(new[0].jds, x2.jds, x2.mags, left=0., right=0.)
        # plt.figure(3)
        # plt.plot(new[0].jds,new[0].mags, label = 'fully sampled noise')
        # plt.plot(x2.jds,x2.mags,'r', label = 'resampled noise')
        # plt.title("resampling by linear interpolation")
        # plt.show()
        return new



samples = int(span) * 5
samplerate = 1

freqs_noise = np.abs(np.fft.fftfreq(samples, 1/samplerate))
freqs_data = np.linspace(1/300.0, 1/(sampling*2.0), 10000)
pgram = sc.lombscargle(x, y, freqs_data)

print "min max, lenght frequency of noise: ", np.min(freqs_noise),np.max(freqs_noise), len(freqs_noise)
print "min max, lenght frequency of data: ", np.min(freqs_data),np.max(freqs_data), len(freqs_data)

band_noise = band_limited_noise_withPS(freqs_data,len(freqs_data)*pgram, samples=samples, samplerate=samplerate)
x_sample = np.linspace(start,stop,samples)

noise_lcs_band = [pycs.gen.lc.lightcurve()]
noise_lcs_band[0].jds = x_sample
noise_lcs_band[0].mags = band_noise

print "original :", pycs.gen.stat.mapresistats(rls)[curve]
print "generated 1:",pycs.gen.stat.mapresistats(noise_lcs_band)
target_std = pycs.gen.stat.mapresistats(rls)[curve]['std']
target_nruns = pycs.gen.stat.mapresistats(rls)[curve]['nruns']
generated_std = pycs.gen.stat.mapresistats(noise_lcs_band)[0]['std']
Amp = target_std / generated_std
print "required amplification :",Amp

std = []
nruns = []
A_vec = np.linspace(1,Amp,50)

for i,A in enumerate(A_vec):
    band_noise_scan = band_limited_noise_withPS(freqs_data, len(freqs_data)*pgram, samples=samples, samplerate=samplerate)
    noise_lcs_scan = [pycs.gen.lc.lightcurve()]
    noise_lcs_scan[0].jds = x_sample
    noise_lcs_scan[0].mags = band_noise_scan
    # print "generated %i:"%i ,pycs.gen.stat.mapresistats(noise_lcs_scan)
    std.append(pycs.gen.stat.mapresistats(noise_lcs_scan)[0]['std'])
    nruns.append(pycs.gen.stat.mapresistats(noise_lcs_scan)[0]['nruns'])

band_noise_rescaled = band_limited_noise_withPS(freqs_data, len(freqs_data)*Amp*pgram, samples=samples, samplerate=samplerate)
noise_lcs_rescaled = [pycs.gen.lc.lightcurve()]
noise_lcs_rescaled[0].jds = x_sample
noise_lcs_rescaled[0].mags = band_noise_rescaled
print "Rescaled :",pycs.gen.stat.mapresistats(noise_lcs_rescaled)


plt.figure(1)
plt.plot(A_vec,std)
plt.xlabel("Amplification power spectrum")
plt.ylabel("std")

plt.figure(2)
plt.plot(A_vec,nruns)
plt.xlabel("Amplification power spectrum")
plt.ylabel("nruns")


#resampling :
noise_lcs_resampled = interpolate(rls[curve],noise_lcs_rescaled[0], interpolate=interpolation)

print "resampled 1:", pycs.gen.stat.mapresistats(noise_lcs_resampled)
print "target : ", pycs.gen.stat.mapresistats(rls)[curve]

#check if the periodogram of the resampled data looks the same :
pgram_generated = sc.lombscargle(noise_lcs_band[0].jds, noise_lcs_band[0].mags, freqs_data)
pgram_resampled = sc.lombscargle(noise_lcs_resampled[0].jds, noise_lcs_resampled[0].mags, freqs_data)
pgram_rescaled  = sc.lombscargle(noise_lcs_rescaled[0].jds, noise_lcs_rescaled[0].mags, freqs_data)

plt.figure(4)
plt.plot(freqs_data, pgram, label='original')
# plt.plot(freqs_data, pgram_generated, label='generated')
# plt.plot(freqs_data, pgram_rescaled, label='rescaled')
plt.plot(freqs_data, pgram_resampled, label='rescaled and resampled')
plt.xlabel('Frequencies [1/days]')
plt.ylabel('Power')
plt.legend(loc='best')
#
plt.show()


#now scan over the cutting frequencies :
B_vec = np.linspace(0.1,2.,50)
std = []
zruns=[]
nruns=[]
std_resampled = []
zruns_resampled=[]
nruns_resampled=[]
res_Bscan = []

for i,B in enumerate(B_vec):
    freqs_data_Bscan = np.linspace(1/300.0, B*1/(sampling*2.0), 10000)
    pgram_Bscan = sc.lombscargle(x, y, freqs_data_Bscan)

    band_noise_Bscan = band_limited_noise_withPS(freqs_data_Bscan,len(freqs_data_Bscan)*pgram_Bscan, samples=samples, samplerate=samplerate)
    x_sample = np.linspace(start,stop,samples)

    noise_lcs_Bscan = [pycs.gen.lc.lightcurve()]
    noise_lcs_Bscan[0].jds = x_sample
    noise_lcs_Bscan[0].mags = band_noise_Bscan

    generated_std = pycs.gen.stat.mapresistats(noise_lcs_Bscan)[0]['std']
    Amp = target_std / generated_std
    print "Amplification required f: ", Amp

    #redo it know that we know the amplitude :
    band_noise_Bscan = band_limited_noise_withPS(freqs_data_Bscan, len(freqs_data_Bscan)*Amp* pgram_Bscan, samples=samples, samplerate=samplerate)
    x_sample = np.linspace(start, stop, samples)

    noise_lcs_Bscan = [pycs.gen.lc.lightcurve()]
    noise_lcs_Bscan[0].jds = x_sample
    noise_lcs_Bscan[0].mags = band_noise_Bscan

    noise_lcs_Bscan_resampled = interpolate(rls[curve], noise_lcs_Bscan[0], interpolate=interpolation)
    noise_lcs_Bscan_resampled[0].magerrs = errors

    std.append(pycs.gen.stat.mapresistats(noise_lcs_Bscan)[0]['std'])
    nruns.append(pycs.gen.stat.mapresistats(noise_lcs_Bscan)[0]['nruns'])
    zruns.append(pycs.gen.stat.mapresistats(noise_lcs_Bscan)[0]['zruns'])

    std_resampled.append(pycs.gen.stat.mapresistats(noise_lcs_Bscan_resampled)[0]['std'])
    nruns_resampled.append(pycs.gen.stat.mapresistats(noise_lcs_Bscan_resampled)[0]['nruns'])
    zruns_resampled.append(pycs.gen.stat.mapresistats(noise_lcs_Bscan_resampled)[0]['zruns'])

    res_Bscan.append(noise_lcs_Bscan_resampled)

plt.figure(1)
plt.plot(B_vec,std)
plt.plot(B_vec,std_resampled, label = 'resampled')
plt.xlabel("Ratio to Nymquist frequency ")
plt.legend()
plt.ylabel("std")

plt.figure(2)
# plt.plot(B_vec,nruns)
plt.plot(B_vec,nruns_resampled, label = 'resampled')
plt.xlabel("Ratio to Nymquist frequency ")
plt.legend()
plt.ylabel("nruns")

plt.figure(8)
# plt.plot(B_vec,zruns)
plt.plot(B_vec,zruns_resampled, label = 'resampled')
plt.xlabel("Ratio to Nymquist frequency ")
plt.legend()
plt.ylabel("zruns")

#Now find the bes nruns :
ind = np.argmin(np.abs(nruns_resampled - target_nruns))

print "target : ", pycs.gen.stat.mapresistats(rls)[curve]
print "Best fit :", pycs.gen.stat.mapresistats(res_Bscan[ind])
print "Best fit for i=%i and B=%2.2f"%(ind,B_vec[ind])

plt.show()

best_lc = res_Bscan[ind]
pycs.gen.stat.plotresiduals([best_lc],filename='resinoise.png')
pycs.gen.stat.plotresiduals([rls],filename='resinoise.png')
plt.show()