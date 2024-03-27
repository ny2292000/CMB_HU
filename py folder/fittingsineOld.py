import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Gaussian1DKernel
import astropy
astropy.utils.data.clear_download_cache()

class fitClass:
    def __init__(self):
        self.n = 6
        self.gamma =0.01
        self.omega = np.pi/2/250

    def fitsine(self, t, *pars):
        'function of two overlapping peaks'
        A = pars[0]  # sin amplitude
        omega = np.pi/2/pars[1]
        delta1 = pars[2]  # delta
        delta2 = pars[3]
        delta3 = pars[4]
        gaussianwidth = pars[5]
        dd = delta1 * t + delta2 * t ** 2  #+delta3*t**3
        data_1D = A * np.sin(self.omega*t) ** 2
        gauss_kernel = Gaussian1DKernel(gaussianwidth)
        smoothed_data_gauss = convolve(data_1D, gauss_kernel) * np.exp(-self.gamma*t)
        return smoothed_data_gauss



mygauss = fitClass()
mygauss.gamma = 5E-4
mygauss.omega = np.pi/2/150
xlim =2048
ell = np.load("./PG_data/ell.npy")
dll_SMICA_Clean = np.load("./PG_data/dll_SMICA_Clean.npy")
dll_SMICA_Clean -= dll_SMICA_Clean[xlim] - 0.005


parguess = np.array([0.4481793425886116, 238.21713688487588, -0.0010317505177075378,
                     3.6784042048090425e-06, -1.1788894134411374e-09,
                     87.51969232795778, 53.49229832375512, 104.27784573919746, 0.002647926174294393])
plt.plot(ell, mygauss.fitsine(ell, *parguess), 'r-')
plt.xlim([0,1900])
plt.ylim([0,0.5])
plt.show()
popt, _ = curve_fit(mygauss.fitsine, ell[0:xlim], dll_SMICA_Clean[0:xlim], parguess)
print(*popt,sep = ", ")
################################################################
fig, axis1 = plt.subplots()
axis1.plot(ell, dll_SMICA_Clean)
axis1.plot(ell, mygauss.fitsine(ell, *popt), 'r-')
axis1.set_xlim([0, xlim])
axis1.set_ylim([0, 0.5])
axis1.set_xlabel('Spherical Harmonic L')
axis1.set_ylabel('Intensity (arb. units)')
axis1.set_title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') %
          (1, 1))
plt.legend(['Power Spectrum', 'Fitted Data'])
plt.savefig('./img1/HighFreqFittedPowerSpectrum.png')
plt.show()




