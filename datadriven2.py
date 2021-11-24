import matplotlib.pylab as plt
import healpy as hp
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import minimize

thishome = "/media/home/mp74207/GitHub/CMB_HU/Data SupernovaLBLgov/"
# https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/matrix_bpasscorr.html
# https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_1024_R2.02_full.fits

smicafiles = [
              # "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
              # "COM_CMB_IQU-sevem_2048_R3.01_full.fits",
              # "COM_CMB_IQU-smica_1024_R2.01_full.fits",
              "COM_CMB_IQU-smica_1024_R2.02_full.fits",
              # "COM_CMB_IQU-smica_2048_R3.00_full.fits",
              # "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits",
              # "LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits",
              # "LFI_SkyMap_044-BPassCorrected_1024_R3.00_full.fits",
              # "LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits",
              ]

def B_l(beam_arcmin, ls1):
    theta_fwhm = ((beam_arcmin / 60.0) / 180) * math.pi
    theta_s = theta_fwhm / (math.sqrt(8 * math.log(2)))
    return np.exp(-2 * (ls1 + 0.5) ** 2 * (math.sin(theta_s / 2.0)) ** 2)

def get_dl(fcolors, nside, beam_arc_min=5):
    cl_SMICA = hp.anafast(fcolors)
    ell = np.arange(len(cl_SMICA))
    pl = hp.sphtfunc.pixwin(nside=nside)
    dl_SMICA = cl_SMICA / (B_l(beam_arc_min, ell) ** 2 * pl ** 2)
    dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi))
    return cl_SMICA, dl_SMICA, ell

class fitClass:
    def __init__(self):
        self.t = []
        self.smica = []
        self.white_noise=[]

    def correctWN(self,x):
        return x[0]+x[1]*np.exp(x[2]*self.t)*self.white_noise

    def fitGN(self,x):
        err = np.sum((self.smica-self.correctWN(x))**2)
        return err

    def optimizeme(self, x0):
        x00 = minimize(self.fitGN, x0,
                       method='nelder-mead', options={'xatol': 1e-12, 'disp': True})
        err = x00.fun
        xx0 = x00.x
        return xx0, err

    def plotme(self,x0):
        plt.plot(self.t, self.smica, self.t, self.correctWN(x0))
        plt.xlim([0,None])
        plt.ylim([0,np.max(self.smica)])
        plt.show()


if __name__ == "__main__":
    diffmap = np.load("./img1/diffmap.npy")
    mygauss = fitClass()
    for f in smicafiles:
        planck_IQU_SMICA = hp.fitsfunc.read_map(thishome + f, dtype=np.float)
        if "1024" in f:
            nside=1024
        if "2048" in f:
            nside=2048
        mu_smica, sigma_smica = norm.fit(planck_IQU_SMICA)
        white_noise = np.ma.asarray(np.random.normal(0, sigma_smica, 12 * nside ** 2))
        cl_WHITE_NOISE, dll_WHITE_NOISE, ell = get_dl(white_noise, nside=nside, beam_arc_min=1)
        for beam_arc_min in [10]:
            cl_SMICA, dll_SMICA, ell = get_dl(planck_IQU_SMICA, nside=nside, beam_arc_min=beam_arc_min)
            mygauss.smica=dll_SMICA[ell>2000]
            mygauss.t = ell[ell>2000]
            mygauss.white_noise = dll_WHITE_NOISE[ell > 2000]
            x0 = np.array([ 0,  1.02461539e-2, -1.00693933e-03])
            mygauss.plotme(x0)
            xout, err = mygauss.optimizeme(x0)
            print(xout, err)
            mygauss.smica=dll_SMICA
            mygauss.t = ell
            mygauss.white_noise = dll_WHITE_NOISE
            adjWN = mygauss.correctWN(xout)
            ymax = np.max(dll_SMICA)
            plt.plot(ell, dll_SMICA, ell, adjWN)
            plt.legend([ "SMICA", "White_Noise"])
            plt.ylim([0,ymax])
            plt.xlim([0,None])
            plt.title(f + "\n"+ "beam_arc_min={}".format(beam_arc_min))
            plt.show()
            plt.plot(ell, dll_SMICA-adjWN)
            plt.legend([ "Noise-Free SMICA"])
            plt.ylim([0,0.6e-8])
            plt.xlim([0,2000])
            plt.title(f + "\n"+ "beam_arc_min={}".format(beam_arc_min))
            plt.show()



        # Laplace transform (t->s)
        y = (dll_SMICA-adjWN)[1:2048]
        t = ell[1:2048]
        y[y <= 0] = 0.0
        # Frequency domain representation
        samplingFrequency=2048
        # Create subplot
        figure, axis = plt.subplots(2, 1)
        plt.subplots_adjust(hspace=1)

        # Time domain representation for sine wave 1
        axis[0].set_title('Original CMB-HF Power Spectrum')
        axis[0].plot(t, y)
        axis[0].set_xlim([0,2048])
        axis[0].set_ylim([0, None])
        axis[0].set_xlabel('L')
        axis[0].set_ylabel('DL')
        # plt.show()
        # Time domain representation for sine wave 2
        fourierTransform = np.fft.fft(y)   # Normalize amplitude
        fourierTransform = fourierTransform[range(int(len(y) / 2))]  # Exclude sampling frequency
        tpCount = len(y)
        values = np.arange(int(tpCount / 2))
        timePeriod = tpCount / samplingFrequency
        frequencies = np.fft.fftshift(values / timePeriod)
        periods = 1/frequencies
        # Frequency domain representation
        axis[1].set_title('Fourier transform depicting the frequency components')
        axis[1].plot(frequencies, abs(fourierTransform))
        axis[1].set_xlabel('Frequencies')
        axis[1].set_ylabel('Amplitude')
        axis[1].set_xlim(0,None)
        axis[1].set_ylim(0, 1e-7)
        plt.show()
        a=1
