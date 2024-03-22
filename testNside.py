
from numpy.random import rand
from lib3 import *
from parameters import *

pl = hp.sphtfunc.pixwin(1024)

white_noise = np.ma.asarray(np.random.normal(0, 0.0001, 12 * 1024 ** 2))
planck_IQU_SMICA = hp.fitsfunc.read_map("./Data SupernovaLBLgov/COM_CMB_IQU-smica_1024_R2.02_full.fits", dtype=float)
planck_theory_cl = np.loadtxt(
    "./Data SupernovaLBLgov/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt", dtype=float)
cl_SMICA = hp.anafast(planck_IQU_SMICA, lmax=1024)
ell = np.arange(len(cl_SMICA))

# Deconvolve the beam and the pixel window function
dl_SMICA = cl_SMICA / (B_l(10.0, ell) ** 2 * pl[0:1025] ** 2)
dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi)) / 1E-12
cl_white = hp.anafast(white_noise, lmax=1024)
dl_white = (ell * (ell + 1) * cl_white / (2 * math.pi)) / 1E-12

# plotSMICA_aitoff(planck_IQU_SMICA)
# plotSMICAHistogram(planck_IQU_SMICA)
# plot_WhiteNoise(white_noise)
# plotWhiteNoiseHistogram(white_noise)
# plot_CL(ell, dl_SMICA, planck_theory_cl,dl_white)

# We check the orthogonality of the spherical harmonics:
# Si (l,m) =! (l',m') the inner product must be zero
Y = lambda l, m, theta, phi: sp.sph_harm(m, l, phi, theta)
f = lambda theta, phi: Y(4, 3, theta, phi)
g = lambda theta, phi: Y(4, 2, theta, phi)





if __name__=="__main__":
    image = planck_IQU_SMICA
    nside = 1024
    mm = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside=nside, ipix=mm)
    nsidearray ={0}
    nsidearray.update( set([ int(x) for x in np.geomspace(1, 400, 100)]) )
    a=datetime.now()
    print(a)
    df = getspectrumL([image],theta, phi, nsidearray)
    df.to_csv("./powerspectrum.txt")
    b=datetime.now()
    print(b, (b-a).seconds)