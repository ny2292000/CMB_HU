import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
import healpy as hp
from astropy.convolution import convolve, Gaussian1DKernel
import astropy
from parameters import *
from scipy.optimize import minimize
astropy.utils.data.clear_download_cache()

def getWN(x, ell, white_noise):
    return  x[0]+x[1]*np.exp(-x[2] * ell) * white_noise
    # return -0.06424533434828383 + x[1] * np.exp(-x[2] * ell) * white_noise

def correctSMICA(x, ell, SMICA):
    #dissipation
    return  (np.exp(x[2] * ell)/x[1] * (SMICA-x[0]))

def getexpcoef(x, ell, SMICA, white_noise):
    return np.sum((SMICA - getWN(x, ell, white_noise))**2)

def plotNewWhiteNoise(x, ell,white_noise, SMICA, ymax=None):
    white_noise0 =  getWN(x, ell, white_noise)
    plt.plot(ell,SMICA,ell,white_noise0)
    plt.xlim([0,None])
    plt.ylim([0,ymax])
    plt.show()

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
        self.n = 6
        self.gamma =0.01
        self.omega = np.pi/2/250

    def fitsine(self, t, *pars):
        'function of two overlapping peaks'
        A = pars[0]  # sin amplitude
        gaussianwidth = pars[1]
        shift = pars[2]
        gamma = pars[3]
        delta1 = np.pi/2/pars[4]  # delta
        delta2 = pars[5]  # delta
        delta3 = pars[6]
        B = pars[7]
        dd = delta1 + delta2 * t + delta3 * t ** 2
        data_1D = (B * np.cos(dd * (t + shift)) ** 2 + A * np.sin(dd*(t+shift))) ** 2 * np.exp(gamma * dd* (t+shift))
        gauss_kernel = Gaussian1DKernel(gaussianwidth)
        smoothed_data_gauss = convolve(data_1D, gauss_kernel)
        return smoothed_data_gauss

    def sinOfk(self,t, peak=280):
        kmax = np.max(t)
        a = pd.DataFrame(np.zeros([2*kmax,1]), columns=["val"])
        for k in t:
            b = np.sin(np.pi/2/peak*k)**2
            for l in np.arange(k):
                for m in np.arange(-l,l+1):
                    a.loc[l+np.abs(m),"val"]+=b
        a = a/a.max()
        return a.values

if __name__=="__main__":
    beam_arc_min=5
    white_noise = np.ma.asarray(np.random.normal(0, sigma_smica, 12 * 1024 ** 2))
    diffmap = np.load("./img1/diffmap.npy")
    cl_SMICA, dll_SMICA, ell = get_dl(planck_IQU_SMICA, nside=1024, beam_arc_min=beam_arc_min)
    cl_SMICA_DIFF, dll_SMICA_DIFF, ell = get_dl(diffmap, nside=1024, beam_arc_min=beam_arc_min)
    cl_WHITE_NOISE, dll_WHITE_NOISE, ell = get_dl(white_noise, nside=1024, beam_arc_min=beam_arc_min)
    plt.plot(ell,dll_SMICA,ell, dll_SMICA_DIFF, ell,dll_WHITE_NOISE/300)
    plt.legend([ "SMICA", "SMICA_HF_Diff", "White_Noise"])
    plt.xlim([0,3000])
    plt.title("beam_arc_min={}".format(beam_arc_min))
    plt.ylim([0,None])
    plt.show()
    x01 = np.array([1.88721404e-01, 7.39444768e-06, 3.33566483e-03, 1.0])
    # x01 = np.array([0.17364976, 0.07587949, 0.00374772])
    dll_SMICA_DIFF*=1E8
    a=1800
    x00 = minimize(getexpcoef, x01, args=(ell[a::], dll_SMICA_DIFF.squeeze()[a::], dll_WHITE_NOISE.squeeze()[a::]),
                   method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    err = x00.fun
    xx0 = x00.x
    # xx0 =  np.array([5.45318980e-02,  5.13947533e-05, -3.46415531e-03, -5.24338388e-11])
    print(xx0)
    xlim = 2000
    plotNewWhiteNoise(xx0, ell, dll_WHITE_NOISE, dll_SMICA_DIFF, ymax=0.6)
    dll_SMICA_Clean = getWN(xx0, ell, dll_WHITE_NOISE)-dll_SMICA_DIFF
    # dll_SMICA_Clean0 = correctSMICA(xx0, ell, dll_SMICA_Clean)
    plt.plot(ell,dll_SMICA_Clean)
    plt.xlim([0,3000])
    plt.ylim([0,None])
    plt.show()
    np.save("./PG_data/dll_SMICA_Clean.npy",dll_SMICA_Clean)
    # dll_SMICA_Clean = np.load("./PG_data/dll_SMICA_Clean.npy")

    mygauss = fitClass()
    mygauss.gamma = -2.1E-3
    mygauss.omega = np.pi/2/140

    alms = hp.map2alm(diffmap, lmax=1024)


    gamma = 3e-3
    alpha=5e6
    plt.semilogy(ell, dll_SMICA, ell, alpha*dll_WHITE_NOISE*np.exp(-gamma*ell), ell,1E-8*(dll_SMICA_Clean+0.05))
    plt.ylim([1e-20,1E-7])
    plt.xlim(0,None)
    plt.show()
    alms2 = np.abs(alms)**2*1E12
    gauss_kernel = Gaussian1DKernel(3)
    smoothed_data_gauss = convolve(alms2, gauss_kernel)
    # plt.plot(alms2)
    # plt.plot(np.abs(smoothed_data_gauss)) #* ell[1:1176]
    plt.plot(ell, dll_SMICA_Clean)
    # plt.plot(mygauss.sinOfk(ell))
    # plt.plot(cl_SMICA_Clean[1:]* ell[1:1024])
    plt.xlim([0,400])
    plt.ylim([0,None])
    plt.show()
    parguess = np.array([12, 10.93337053709095,
                         -145.629475034814906, -0.12,
                         300, 1E-06, -4.8E-10, 12])
    plt.figure()
    plt.plot(ell, mygauss.fitsine(ell, *parguess), 'r-')
    plt.plot(ell, dll_SMICA_Clean)
    plt.xlim([0,xlim])
    plt.ylim([0,None])
    plt.show()
    popt, _ = curve_fit(mygauss.fitsine, ell[0:xlim], dll_SMICA_Clean[0:xlim], parguess)
    print(*popt,sep = ", ")
    ################################################################
    fig, axis1 = plt.subplots()
    axis1.plot(ell, dll_SMICA_Clean)
    axis1.plot(ell, mygauss.fitsine(ell, *popt), 'r-')
    axis1.set_xlim([0, xlim])
    axis1.set_ylim([0, None])
    axis1.set_xlabel('Spherical Harmonic L')
    axis1.set_ylabel('Intensity (arb. units)')
    axis1.set_title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') %
              (1, 1))
    plt.legend(['Power Spectrum', 'Fitted Data'])
    plt.savefig('./img1/HighFreqFittedPowerSpectrum.png')
    plt.show()




