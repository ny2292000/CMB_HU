import matplotlib.pylab as plt
import healpy as hp
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import sici, sinc
import pandas as pd
from astropy.convolution import convolve, Gaussian1DKernel

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

def createDensityOfStates(kmax):
    df = np.zeros([kmax,1])
    for k in np.arange(2,kmax+1):
        for l in np.arange(k):
            for m in np.arange(-l,l+1):
                df[l]+=1.0
    np.save("./densityOfStates.npy",df)


class fitClass:
    def __init__(self):
        self.t = []
        self.smica = []
        self.white_noise=[]
        self.func=self.correctWN
        self.n = 6

    def six_peaks(self, ell, *parkguess):
        'function of two overlapping peaks'
        p = np.zeros([1, len(ell)])
        # gamma = parkguess[-1:][0]
        for i in np.arange(0, self.n * 3, 3):
            a0 = parkguess[i + 0] # peak height
            a1 = parkguess[i + 1] # Gaussian Center
            a2 = parkguess[i + 2] # std of gaussian
            gamma = parkguess[i + 3]  # std of gaussian
            p += self.fitfun(ell, a0, a1, a2, gamma)
        return p[0]

    def fitfun(self, x, a0, a1, a2, gamma):
        return a0 * norm.pdf(x, loc=a1, scale=a2) * np.exp(-gamma * x) * np.sqrt(2 * np.pi)

    def correctWN(self,x):
        return x[0] * np.exp(x[1] * self.t) * self.white_noise
        # return x[0]+x[1]*np.exp(x[2]*self.t)*self.white_noise

    def sins_quared(self,x):
        phase =  self.t/x[3]
        data_1D = x[0]*sici( phase )[0]
        box_kernel = Gaussian1DKernel(x[1])
        smoothed_data_box = convolve(data_1D, box_kernel) * np.exp(x[2] * phase)
        return smoothed_data_box

    def fitGN(self,x):
        err = np.sum((self.smica-self.func(x))**4*1E20)
        return err

    def optimizeme(self, x0):
        x00 = minimize(self.fitGN, x0,
                       method='nelder-mead', options={'xatol': 1e-18, 'disp': True,'maxiter':10000})
        err = x00.fun
        xx0 = x00.x
        return xx0, err

    def plotme(self,x0, ymax =None):
        plt.figure()
        plt.plot(self.t, self.smica)
        plt.plot(self.t, self.func(x0), 'r-')
        # plt.xlim([0, 2500])
        # plt.ylim([0, ymax])
        plt.show()

    def plotmeSix(self,x0):
        plt.plot(self.t, self.smica, self.t, self.func(x0))
        plt.xlim([0,None])
        plt.ylim([0,np.max(self.smica)])
        plt.show()


if __name__ == "__main__":
    # createDensityOfStates(1000)
    densityOfStates = np.load("./densityOfStates.npy")
    plt.plot(densityOfStates)
    plt.show()
    beam_arc_min =10
    diffmap = np.load("./img1/diffmap.npy")
    f = "COM_CMB_IQU-smica_1024_R2.02_full.fits"
    planck_IQU_SMICA = hp.fitsfunc.read_map(thishome + f, dtype=float)
    nside=1024
    mu_smica, sigma_smica = norm.fit(planck_IQU_SMICA)
    white_noise = np.ma.asarray(np.random.normal(0, sigma_smica, 12 * nside ** 2))
    cl_WHITE_NOISE, dll_WHITE_NOISE, ell = get_dl(white_noise, nside=nside, beam_arc_min=1)
    cl_SMICA_clean, dll_SMICA_clean, ell = get_dl(diffmap, nside=nside, beam_arc_min=beam_arc_min)
    ###################################################
    mygauss = fitClass()
    mygauss.smica=dll_SMICA_clean[ell>2500]
    mygauss.t = ell[ell>2500]
    mygauss.white_noise = dll_WHITE_NOISE[ell > 2500]
    x0 = np.array([2e-2, 8.17916807e-04])
    mygauss.plotme(x0)
    mygauss.func = mygauss.correctWN
    xout, err = mygauss.optimizeme(x0)
    print(xout, err)
    ###################################################
    mygauss.smica=dll_SMICA_clean
    mygauss.t = ell
    mygauss.white_noise = dll_WHITE_NOISE
    adjWN = mygauss.correctWN(xout)
    ###################################################
    ymax = np.max(dll_SMICA_clean)
    plt.plot(ell, dll_SMICA_clean, ell, adjWN)
    plt.legend([ "SMICA", "White_Noise"])
    plt.ylim([0,ymax])
    plt.xlim([0,None])
    plt.title(f + "\n"+ "beam_arc_min={}".format(beam_arc_min))
    plt.show()
    ###################################################
    dll_SMICA_Clean_NoNoise = (dll_SMICA_clean - adjWN)*np.exp(-ell*xout[1])
    ymax = np.max(dll_SMICA_Clean_NoNoise)
    plt.plot(ell, dll_SMICA_Clean_NoNoise)
    plt.legend([ "Noise-Free SMICA HF"])
    plt.ylim([0,ymax])
    plt.xlim([0,None])
    plt.title(f + "\n"+ "beam_arc_min={}".format(beam_arc_min))
    plt.show()
    ###################################################
    dll_SMICA_Clean_NoNoise = dll_SMICA_Clean_NoNoise[0:2500]
    ell = ell[0:2500]
    dll_SMICA_Clean_NoNoise -= dll_SMICA_Clean_NoNoise[-1:]
    dll_SMICA_Clean_NoNoise[dll_SMICA_Clean_NoNoise<0.0]=0.0
    ymax = np.max(dll_SMICA_Clean_NoNoise)
    plt.plot(ell, dll_SMICA_Clean_NoNoise)
    plt.legend([ "Noise-Free SMICA HF"])
    plt.ylim([0,ymax])
    plt.xlim([0,ell[-1:]])
    plt.title(f + "\n"+ "beam_arc_min={}".format(beam_arc_min))
    plt.show()
    ###################################################
    # x[0] * (1 + x[1] * sici(np.pi / 2 / x[5] * self.t)[1] ** 2) * np.exp(
    #     x[2] + x[3] * self.t + x[4] * sici(np.pi / 2 / x[5] * self.t)[1])
    # initpoint=100
    # x0 = np.array([1.2e-08, 70, -0.30, 162])
    # mygauss.func = mygauss.sins_quared
    # nn = len(dll_SMICA_Clean_NoNoise)
    # mygauss.smica = dll_SMICA_Clean_NoNoise[initpoint::]
    # mygauss.t = ell[initpoint:nn].astype(float)
    # np.save("./correctedSMica.npy", mygauss.smica)
    # np.save("./ell.npy", mygauss.t)
    # xout, err = mygauss.optimizeme(x0)
    # print("[", *xout,"]", sep=", " )
    # mygauss.plotme(xout)
    # initpoint=1
    # mygauss.smica = dll_SMICA_Clean_NoNoise[initpoint::]
    # mygauss.t = ell[initpoint:len(mygauss.smica)+initpoint].astype(float)
    # # mygauss.plotme(xout, 5e-9)
    # x0 = np.array([4.607637045207382e-09, 43.12499999999282, -0.26441579664751114, 72.0360578427234 ])
    # mygauss.plotme(x0, 5e-9)
    # #####################################################################################
    # #####################################################################################
    # initpoint=1
    # x0 = np.array([4.607637045207382e-09, 43.12499999999282, -0.26441579664751114, 72.0360578427234 ])
    # mygauss.smica = dll_SMICA_Clean_NoNoise[initpoint::]
    # mygauss.t = ell[initpoint:len(mygauss.smica)+initpoint].astype(float)
    # mygauss.plotme(x0, 5e-9)
    #####################################################################################
    #####################################################################################
    mygauss.n = 5
    # parguess = np.array([6.79053092e-05, 3.60285741e+02, 9.05805670e+01 #, 1.88936178e-02
    #                     , 7.83077145e-01, 7.07591900e+02, 8.39559884e+01 #, 2.63757619e-02
    #                     , 5.57640782e-06, 8.54146768e+02, 1.01730999e+02 #, 5.65945326e-03
    #                     , 2.20551261e-04, 1.18322835e+03, 8.43704492e+01 #, 8.72663725e-03
    #                     , 7.37379920e-10, 1.39338263e+03, 9.14057363e+01, 1.02587790e-03])

    parguess = np.array([3.28642213e-001, 1.36471759e+004, 1.34022432e+002,-4.49356565e-262,
                         3.23557240e-003, 5.50822821e+002, 3.61985065e+001, 1.14115079e-002,
                         1.32199607e-003, 8.00821927e+002, 1.15782930e+002, 2.05724969e-003,
                         5.34256601e-002, 1.16050790e+006, 2.17787263e+002, 9.80182678e+001,
                         1.02257029e-007, 1.40000000e+003, 1.20000000e+002, 2.05724969e-003])

    # parguess = np.array([ 0.00561422844, 250.488235, 120.0045112, 0.00205724969,
    #                       0.0032355724, 550.822821, 115.793998, 0.00205724969,
    #                       0.00323507187, 800.821927, 115.78293, 0.00205724969,
    #                       4.99454352e-07, 1200.643373, 90.0461827, 0.00205724969,
    #                       1.02257029e-07, 1400, 120, 0.00205724969
    #                       ])
    popt, err = curve_fit(mygauss.six_peaks, ell, dll_SMICA_Clean_NoNoise, parguess)
    print("[", *popt, "]" , sep=", ")
    parguess = np.array(popt).reshape([5, 4])
    parguess = parguess[parguess[:, 1].argsort()]
    centers = np.array([parguess[x, 1] for x in np.arange(5)])
    freqs = centers[1::] - centers[0:-1:]
    amplt = np.array([parguess[x, 0]* np.exp(-parguess[x, 3] * x) for x in np.arange(5)])
    fitting1 = np.polyfit(centers, np.log(amplt), 1)
    yy = np.exp(fitting1[1]) * np.exp(fitting1[0] * ell)
    # First peak is at pi/2=l*Delta => Delta=pi/2/l
    delta = np.pi / 2 / centers[0]
    gamma1 = fitting1[0] / delta

    ################################################################
    ################################################################
    fig, axis1 = plt.subplots()
    axis1.plot(ell, dll_SMICA_Clean_NoNoise)
    for i in np.arange(5):
        axis1.plot(ell, mygauss.fitfun(ell, *parguess[i, :]), 'r-')
    axis1.plot(ell, mygauss.six_peaks(ell, *popt), 'r-')
    axis1.set_xlim([0, 2000])
    axis1.set_ylim([0, None])
    axis1.set_xlabel('Spherical Harmonic L')
    axis1.set_ylabel('Intensity (arb. units)')
    axis1.set_title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') %
              (delta, gamma1))
    plt.legend(['Power Spectrum', 'Fitted Data'])
    plt.savefig('./img1/HighFreqFittedPowerSpectrum.png')
    plt.show()

    fig, axis1 = plt.subplots()
    fitting1 = np.polyfit(centers, np.log(amplt), 1)
    yy = np.exp(fitting1[1]) * np.exp(fitting1[0] * ell)
    # First peak is at pi/2=l*Delta => Delta=pi/2/l
    plt.scatter(centers, amplt)
    plt.plot(ell, yy)
    plt.xlim([0, 2000])
    # plt.ylim([-200, 2000])
    plt.xlabel('Spherical Harmonic L')
    plt.ylabel('Intensity (arb. units)')
    plt.title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') % (
        delta, gamma1))
    plt.legend(['Power Spectrum', 'Fitted Data'])
    # plt.savefig('./img1/HighFreqDissipationFittedPowerSpectrum.png')
    plt.show()
