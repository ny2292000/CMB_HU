import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Gaussian1DKernel

class fitClass:
    def __init__(self):
        self.n = 6
        self.gamma =0.01

    def six_peaks(self, t, *parkguess):
        'function of two overlapping peaks'
        p = np.zeros([1, len(t)])
        # self.gamma = parkguess[-1:][0]  # std of gaussian
        for i in np.arange(0, self.n * 3, 3):
            a0 = parkguess[i + 0] # peak height
            a1 = parkguess[i + 1] # Gaussian Center
            a2 = parkguess[i + 2] # std of gaussian
            # gamma = parkguess[i + 3]  # std of gaussian
            p += self.fitfun(t, a0, a1, a2)
        return p[0] * np.exp(-self.gamma * t)

    def fitfun(self, x, a0, a1, a2):
        return a0 * norm.pdf(x, loc=a1, scale=a2) * np.sqrt(2 * np.pi)

    def fitsine(self, t, *pars):
        'function of two overlapping peaks'
        A = pars[0]  # sin amplitude
        delta1 = pars[1]  # delta
        delta2 = pars[2]  # delta
        delta3 = pars[3]
        gaussianwidth = pars[4]
        dd = delta1 + delta2 * t + delta3 * t ** 2  # +delta4*t**3
        data_1D = A * np.sin(dd) ** 2
        gauss_kernel = Gaussian1DKernel(gaussianwidth)
        smoothed_data_gauss = convolve(data_1D, gauss_kernel) * np.exp(self.gamma * dd)
        return smoothed_data_gauss



mygauss = fitClass()
mygauss.n = 6
mygauss.gamma = 0.76E-3
xlim =1900
ell = np.load("./PG_data/ell.npy")
dll_SMICA_Clean = np.load("./PG_data/dll_SMICA_Clean.npy")
dll_SMICA_Clean -= dll_SMICA_Clean[xlim] - 0.005
dll_SMICA_Clean *= (2*ell+1)
parguess = np.array([43.80231637292005,224.90776924468133,86.20608331219438,
                     23.043501448288797,540.7059682253914,91.30382186875981,
                     28.6873242499867,822.9064942470347,94.88903529847492,
                     8.169152366216174,1433.4110032016874,87.44341486716048,
                     13.166892021233775,1137.7981648524428,92.40032186108301,
                     2.637071356491338,1773.4905857102524,85.18200984970403])

# plt.plot(ell, dll_SMICA_Clean)
# plt.plot(ell, mygauss.six_peaks(ell, *parguess), 'r-')
# plt.xlim([0, xlim])
# plt.ylim([0, None])
# plt.xlabel('Spherical Harmonic L')
# plt.ylabel('Intensity (arb. units)')
# plt.show()
popt, _ = curve_fit(mygauss.six_peaks, ell[0:xlim], dll_SMICA_Clean[0:xlim], parguess)
print("[")
print(*popt, sep=",")
print("]")
parguess = np.array(popt).reshape([mygauss.n, 3])
################################################################
################################################################
# parguess = np.array(popt[0:-1:]).reshape([5, 4])
# gamma= popt[-1:][0]
parguess = parguess[parguess[:, 1].argsort()]
widths = np.array([parguess[x, 2] for x in np.arange(5)])
centers = np.array([parguess[x, 1]+mygauss.gamma*widths[x]**2 for x in np.arange(5)])
freq = [centers[i + 1] - centers[i] for i in np.arange(mygauss.n-2)]
freq0= 2/np.pi*centers[0]
freqs = centers[1::] - centers[0:-1:]
amplt = np.array([parguess[x, 0]*np.exp(-mygauss.gamma*centers[x]) for x in np.arange(5)])
fitting1 = np.polyfit(centers, np.log(amplt), 1)
yy = np.exp(fitting1[1]) * np.exp(fitting1[0] * ell)
# First peak is at pi/2=l*Delta => Delta=pi/2/l
Wavelength =np.mean(freq)
gamma1 = fitting1[0]
fig, axis1 = plt.subplots()
# First peak is at pi/2=l*Delta => Delta=pi/2/l
axis1.scatter(centers, amplt)
axis2 = plt.twinx(axis1)
axis2.scatter(centers[1:],freq, c="r")
axis1.set_ylim([0,None])
axis1.plot(ell, yy)
axis1.set_xlim([0, xlim])
axis1.set_ylim([0, None])
axis2.set_ylim([0, None])
axis1.set_xlabel('Spherical Harmonic L')
axis1.set_ylabel('Intensity (arb. units)')
axis1.set_title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') % (
    Wavelength, gamma1))
axis1.legend(['Power Spectrum', 'Fitted Data'])
# plt.savefig('./img1/HighFreqDissipationFittedPowerSpectrum.png')
plt.show()

################################################################
################################################################
fig, axis1 = plt.subplots()
axis1.plot(ell, dll_SMICA_Clean)
for i in np.arange(mygauss.n):
    axis1.plot(ell, mygauss.fitfun(ell, *popt[i*3:i*3+3])*np.exp(-mygauss.gamma*ell), 'r-')
axis1.plot(ell, mygauss.six_peaks(ell, *popt), 'r-')
axis1.set_xlim([0, xlim])
axis1.set_ylim([0, None])
axis1.set_xlabel('Spherical Harmonic L')
axis1.set_ylabel('Intensity (arb. units)')
axis1.set_title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') %
          (Wavelength, gamma1))
plt.legend(['Power Spectrum', 'Fitted Data'])
plt.savefig('./img1/HighFreqFittedPowerSpectrum.png')
plt.show()


freq = [centers[i + 1] - centers[i] for i in np.arange(mygauss.n-2)]
freqdiff = [x / freq[0] for x in freq]
print(freq, freqdiff, widths)
print(amplt)
