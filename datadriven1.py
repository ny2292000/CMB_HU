import matplotlib.pylab as plt
import healpy as hp
import numpy as np
import math
from scipy.stats import norm
thishome = "/media/home/mp74207/GitHub/CMB_HU/Data SupernovaLBLgov/"
# https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/matrix_bpasscorr.html
# https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_1024_R2.02_full.fits

smicafiles = ["COM_CMB_IQU-nilc_2048_R3.00_full.fits",
              "COM_CMB_IQU-sevem_2048_R3.01_full.fits",
              "COM_CMB_IQU-smica_1024_R2.01_full.fits",
              "COM_CMB_IQU-smica_1024_R2.02_full.fits",
              "COM_CMB_IQU-smica_2048_R3.00_full.fits",
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

if __name__ == "__main__":
    for f in smicafiles:
        planck_IQU_SMICA = hp.fitsfunc.read_map(thishome + f, dtype=np.float)
        if "1024" in f:
            nside=1024
        if "2048" in f:
            nside=2048
        (mu_smica, sigma_smica) = norm.fit(planck_IQU_SMICA)
        for beam_arc_min in [5, 10]:
            white_noise = np.ma.asarray(np.random.normal(0, sigma_smica, 12 * nside ** 2))
            diffmap = np.load("./img1/diffmap.npy")
            cl_SMICA, dll_SMICA, ell = get_dl(planck_IQU_SMICA, nside=nside, beam_arc_min=beam_arc_min)
            cl_WHITE_NOISE, dll_WHITE_NOISE, ell = get_dl(white_noise, nside=nside, beam_arc_min=0)
            ymaxWN = np.max(dll_WHITE_NOISE)
            ymax = np.max(dll_SMICA)
            plt.plot(ell, dll_SMICA, ell, dll_WHITE_NOISE / ymaxWN * ymax)
            plt.legend([ "SMICA", "White_Noise"])
            plt.ylim([0,ymax])
            plt.xlim([0,None])
            plt.title(f + "\n"+ "beam_arc_min={}".format(beam_arc_min))
            plt.show()