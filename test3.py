from scipy.special import sici
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import healpy as hp
import math
from astropy.convolution import convolve, Gaussian1DKernel
from parameters import *


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

def optimizeme(x0):
    x00 = minimize(sins_quared, x0,
                   method='nelder-mead', options={'xatol': 1e-16, 'disp': True, 'maxiter': 10000})
    err = x00.fun
    xx0 = x00.x
    return xx0, err

def denoise(x0,t,white_noise):
    data_1D = x0[6] * np.exp(x0[7] * t) * white_noise
    return data_1D

def sins_quaredOld(x0,t,white_noise):
    data_1D = x0[0]*np.exp(-x0[1]*t +  x0[2]*(1+x0[3]*np.exp(-x0[4]*t))* sici(t/52)[0]) \
              + 0.0025 * np.exp(0.00152 * t) * white_noise
    return data_1D

def sins_quared(x0,t,white_noise):
    gauss_kernel = Gaussian1DKernel(x0[0])
    ampl = np.sin(t*np.pi/2/x0[1])
    amplloss = -t/x0[2] + x0[3]*ampl
    smoothed_data_gauss = convolve(amplloss, gauss_kernel)
    integraloflostamps= np.cumsum(smoothed_data_gauss)
    totalampl =ampl-amplloss+integraloflostamps
    return x0[4]*np.exp(totalampl) + x0[5] * np.exp(0.00152 * t) * white_noise +x0[6]

def newerr(x, smica, ell, white_noise):
    err = smica - denoise(x,ell, white_noise)
    err = np.sum(err ** 2)*1e5
    print(x,err)
    return err

def olderr(x, smica, ell, white_noise):
    err = smica - sins_quared(x,ell, white_noise)
    err = np.sum(err ** 2)*1e20
    print(x,err)
    return err

def plotme(x0,t,smica, white_noise, ymax =None):
    plt.figure()
    plt.plot(t, smica)
    plt.plot(t, sins_quared(x0,t,white_noise), 'r-')
    plt.xlim([0, 2500])
    plt.ylim([0, ymax])
    plt.show()

if __name__=="__main__":
    # x = np.linspace(1, np.pi * 30, 1000)
    # y = sici(x)[0]**0.5
    # plt.plot(x, y)
    # plt.show()
    # smica_corrected =np.load("./correctedSMica.npy")
    # smica = np.load("./correctedSMica.npy")
    # ell = np.load("./ell.npy")
    nside = 1024
    f = "COM_CMB_IQU-smica_1024_R2.02_full.fits"
    SMICA = hp.fitsfunc.read_map(thishome + f, dtype=float)
    cl_SMICA, dl_SMICA, ell = get_dl(SMICA, nside=nside, beam_arc_min=10)
    mu_smica, sigma_smica = norm.fit(planck_IQU_SMICA)
    white_noise = np.ma.asarray(np.random.normal(0, sigma_smica, 12 * nside ** 2))
    cl_WHITE_NOISE, dll_WHITE_NOISE, ell = get_dl(white_noise, nside=nside, beam_arc_min=1)
    initpoint = 0
    # x0 = np.array([2.45e-10, 0.0013147043750211843, 2, 9.9, 0.09])
    x0 = np.array([14.604600706014246, 280.44215883100895, 384.1382320984859, 0.07320732751359314,
                   -9.363313943700361e-10, 0.0018918624189458758, 1.4821245799470557e-09])
    smica_short = dl_SMICA[initpoint::] #* 1E9
    dll_WHITE_NOISE_short =dll_WHITE_NOISE[initpoint::]
    # yhat = savgol_filter(smica_short, 51, 3)
    # yhat_max=np.max(yhat)
    # yhat_min=np.min(yhat)
    # yhat= (yhat-yhat_min)+ (yhat_max-yhat_min)*1E-9
    t = ell[initpoint::]
    x00 = minimize(olderr, x0, args=(smica_short, t, dll_WHITE_NOISE_short ),
                   method='nelder-mead', options={'xatol': 1e-17, 'disp': True, 'maxiter': 10000})
    err = x00.fun
    xx0 = x00.x
    print("[", *xx0, "]", sep=", ")
    plotme(xx0, t, smica_short,dll_WHITE_NOISE_short, ymax=None)
    initpoint = 0
    dll_WHITE_NOISE_short = dll_WHITE_NOISE[initpoint::]
    smica_short = dl_SMICA[initpoint::]
    t = ell[initpoint::]
    # [0.00402876 0.00136283]
    x0 = np.array([14.604600706014246, 280.44215883100895, 384.1382320984859, 0.07320732751359314,
                   -9.363313943700361e-10, 0.0018918624189458758, 1.4821245799470557e-09])
    y =  sins_quared(x0,t,dll_WHITE_NOISE_short)
    plt.plot(t, smica_short,t,y)
    plt.show()
