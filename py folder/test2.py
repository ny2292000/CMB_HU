import healpy as hp
import matplotlib.pylab as plt
import mpmath as mp
import numpy as np
from matplotlib import cm
from pyshtools.expand.spharm_functions import spharm as pysh_spherharm

nside3D = 48
ksi, theta, phi = hp.pix2vec(nside=nside3D, ipix=np.arange(hp.nside2npix(nside=nside3D)))
(ksi, theta, phi) = (ksi+0.23, theta+1.4, phi+0.44)
cosksi = np.cos(ksi)
sinksi = np.sin(ksi)

def get_SPH(lmax, theta, phi):
    df = np.zeros([2, (lmax + 1) * (lmax + 2) // 2, len(theta)])
    for i, (t, p) in enumerate(zip(theta, phi)):
        df[:, :, i] = pysh_spherharm(lmax=lmax, theta=t, phi=p, packed=True, kind="real", degrees=False)
    return df


def get_HSPH(k, l, m, theta, phi, cosksi,sinksi, kmax, df=None):
    def get_LM(l, m, df):
        if m >= 0:
            ind = (l + 1) * (l + 2) - l + m
            return df[0, ind, :]
        if m < 0:
            ind = (l + 1) * (l + 2) - l + m
            return df[1, ind, :]
    b = (-1) ** k * np.sqrt(
        2 * (k + 1) / np.pi * mp.factorial(k - l) * 2 ** (2 * l) * mp.factorial(l) ** 2 / mp.factorial(k + l + 1))
    A = np.zeros(cosksi.shape)
    for i, x in enumerate(cosksi):
        A[i] = b * mp.gegenbauer(1 + l, k - l, x)
    if df is None or k > kmax:
        df = get_SPH(kmax, theta, phi)
    B = get_LM(l, m, df)
    fcolors = A * B * sinksi**l
    plot_aitoff(fcolors)
    return fcolors, df

def functionG(k, l, cosksi):
    b = (-1) ** k * np.sqrt(
        2 * (k + 1) / np.pi * mp.factorial(k - l) * 2 ** (2 * l) * mp.factorial(l) ** 2 / mp.factorial(k + l + 1))
    A = np.zeros(cosksi.shape)
    for i, x in enumerate(cosksi):
        A[i] = b * mp.gegenbauer(1 + l, k - l, x)
    print("ended {}_{}".format(k, l))
    return ((1 + l, k - l), A)


def plot_aitoff(fcolors, cmap=cm.RdBu_r):
    hp.mollview(fcolors.squeeze(), cmap=cmap)
    hp.graticule()
    plt.show()


def plot_orth(fcolors):
    hp.orthview(fcolors, min=-1, max=1, title='Raw WMAP data', unit=r'$\Delta$T (mK)')
    plt.show()


def plot_aitoff_df(l, m, df, cmap=cm.RdBu_r):
    def get_LM(l, m, df):
        if m >= 0:
            ind = (l + 1) * (l + 2) - l + m
            return df[0, ind, :]
        if m < 0:
            ind = (l + 1) * (l + 2) - l + m
            return df[1, ind, :]

    fcolors = get_LM(l, m, df)
    plot_aitoff(fcolors, cmap=cmap)
    plot_orth(fcolors)


if __name__ == "__main__":
    kmax = 10
    k = 9
    l = 7
    m = -2
    df = None
    fcolors, df = get_HSPH(k, l, m, theta, phi, cosksi,sinksi, kmax, df=df)
    plot_aitoff_df(l, m, df)