import os
import warnings
from os import listdir
from os.path import isfile, join

import cupy as cu
import matplotlib.pylab as plt
import pygame
from PIL import Image
from pyshtools.legendre import legendre
from scipy.optimize import minimize
from scipy.special import eval_gegenbauer

from lib3 import *
from parameters import *

warnings.filterwarnings("ignore")

pygame.mixer.init()
sound = pygame.mixer.Sound('/usr/share/sounds/freedesktop/stereo/phone-incoming-call.oga')


def B_l(beam_arcmin, ls1):
    theta_fwhm = ((beam_arcmin / 60.0) / 180) * math.pi
    theta_s = theta_fwhm / (math.sqrt(8 * math.log(2)))
    return np.exp(-2 * (ls1 + 0.5) ** 2 * (math.sin(theta_s / 2.0)) ** 2)


class HYPER:
    def __init__(self, nside3D, sigma_smica, planck_IQU_SMICA, k1, kmax,
                 lambda_k=0, lambda_l=0, lambda_m=0):
        self.G = {}
        self.P = np.zeros([1, 1])
        self.nside3D = nside3D
        self.sigma_smica = sigma_smica
        self.xx, self.yy, self.zz = hp.pix2vec(nside=nside3D, ipix=np.arange(hp.nside2npix(nside3D)))
        self.k1 = k1
        self.kmax = kmax
        self.sinksi = np.zeros([1, 1])
        self.phi = np.zeros([1, 1])
        self.lambda_k, self.lambda_l, self.lambda_m = (lambda_k, lambda_l, lambda_m)
        self.SMICA = self.normalizeFColors(planck_IQU_SMICA, sigma_smica)
        self.SMICA_LR = self.SMICA
        self.df = []
        self.my_klm_dict = {}
        self.createDFMapping(k1, kmax)
        self.change_SMICA_resolution(kmax, nside3D, sigma_smica, doit=True)
        self.change_HSH_center(self.lambda_k, self.lambda_l, self.lambda_m, k1, kmax, doit=True)

    def change_SMICA_resolution(self, kmax, nside3D, sigma, doit=False):
        if (self.kmax, self.nside3D, self.sigma_smica) == (kmax, nside3D, sigma) and not doit:
            return
        filename = "./img1/SMICA_{}.png".format(nside3D)
        title = "SMICA_nside3D={}".format(nside3D)
        self.SMICA_LR = self.change_resolution(self.SMICA, sigma, nside3D=nside3D,
                                               title=title, filename=filename, plotme=True, save=True).astype(
            np.float32)

    def factorMP(self, k, l, m):
        m = np.abs(m)
        a = (-1) ** m * np.sqrt((2 * l + 1) / (2 * np.pi) * mp.factorial(l - m) / mp.factorial(l + m))
        b = (-1) ** k * np.sqrt(
            2 * (k + 1) / np.pi * mp.factorial(k - l) / mp.factorial(k + l + 1) * 2 ** (2 * l) * mp.factorial(l) ** 2)
        c = np.float64(a * b)
        return c

    def normalizeFColors(self, fcolors, sigma_smica):
        (mu, sigma) = norm.fit(fcolors)
        if sigma == 0.0:
            return fcolors
        return sigma_smica / sigma * (fcolors - mu)

    def getSpectralFiles(self, mypath):
        return [join(mypath, f) for f in sorted(listdir(mypath)) if isfile(join(mypath, f))]

    def creategiff(self, kmax, nside3D):
        giffiles = self.getSpectralFiles("./img1")
        giffiles = [x for x in giffiles if "aitoff_" in x]
        images = []
        for ff in giffiles:
            with Image.open(ff).convert("RGB") as img:
                images.append(img)
        images[0].save('./img1/CMB_{}_{}.gif'.format(kmax, nside3D),
                       save_all=True, append_images=images[1:], optimize=False, duration=1000, loop=0)

    def creategiffMem(self, x):
        # err, prime, results, fig
        images = [y[3] for y in x]
        titles = [y[1] for y in x]
        errs = [y[0] for y in x]
        for i, xx in enumerate(images):
            (lk, ll, lm) = [x for x in titles[i]]
            err = errs[i]
            filename = "./img1/aitoff_{}_{}_{}_{}_{}.png".format(kmax, np.round(lk, 2), np.round(ll, 2),
                                                                 np.round(lm, 2),
                                                                 np.round(err, 3))
            # draw the canvas, cache the render
            xx.seek(0)
            im = Image.open(xx)
            im.save(filename)

    def change_resolution(self, fcolors, sigma1, nside3D=1024, filename=None, title=None, save=False, plotme=None):
        # noinspection PyUnresolvedReferences
        SMICA_alms = hp.map2alm(fcolors, lmax=nside3D)
        fcolors = hp.alm2map(SMICA_alms, nside=nside3D)
        (mu, sigma) = norm.fit(fcolors)
        if sigma != 0.0:
            fcolors = (fcolors - mu) / sigma * sigma1
        fcolors = np.expand_dims(fcolors, axis=1)
        self.plot_aitoff(fcolors.squeeze(), kk=0, ll=0, mm=0, err=0, filename=filename, title=title, save=save,
                         plotme=plotme)
        return fcolors

    def plotHistogram(self, fcolors, nside3D, kmax, plotme=False):
        (mu, sigma) = norm.fit(fcolors)
        n, bins, patch = plt.hist(fcolors, 600, density=1, facecolor="r", alpha=0.25)
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y)
        plt.xlim(mu - 4 * sigma, mu + 4 * sigma)
        plt.xlabel("Temperature/K")
        plt.ylabel("Frequency")
        plt.title(r"Histogram {}_{} HU Modeling of Planck SMICA Map ".format(kmax, nside3D), y=1.08)
        plt.savefig("./img1/Histogram_{}_{}.png".format(kmax, nside3D), dpi=300)
        if plotme:
            plt.show()

    def plot_aitoff(self, fcolors, kk, ll, mm, err, filename=None, title=None, plotme=False, save=False):
        # noinspection PyUnresolvedReferences
        if title is None:
            title = "{}_{}_{}_{}_{}.png".format(kmax, np.round(kk, 1),
                                                np.round(ll, 1), np.round(mm, 1),
                                                np.round(err * 1, 3))
        hp.mollview(fcolors.squeeze(), title=title, min=-2 * self.sigma_smica,
                    max=2 * self.sigma_smica, unit="K", cmap=cm.RdBu_r)
        hp.graticule()
        if save:
            plt.savefig(filename, format='png', dpi=72)
        if plotme:
            plt.show()
        plt.close(plt.gcf())

    def pickledict(self, test_dict, filename='./errarray.pickle'):
        import pickle
        with open(filename, 'wb') as handle:
            pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def unpickledict(self, filename='./errarray.pickle'):
        import pickle
        with open(filename, 'rb') as handle:
            text_dict = pickle.load(handle)
        return text_dict

    def plot_CL_From_Image(self, fcolors, planck_theory_cl, xmax=3000):
        cl_SMICA = hp.anafast(fcolors)
        ell = np.arange(len(cl_SMICA))
        themax = np.max(planck_theory_cl[10:3000, 1])
        planck_theory_cl[:, 1] = planck_theory_cl[:, 1] / themax
        pl = hp.sphtfunc.pixwin(nside=1024)
        # Deconvolve the beam and the pixel window function
        dl_SMICA = cl_SMICA / (B_l(10.0, ell) ** 2 * pl ** 2)
        dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi))
        dl_SMICA = dl_SMICA / np.max(dl_SMICA)

        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111)
        ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
        ax.plot(planck_theory_cl[:, 0], planck_theory_cl[:, 1], ell * np.pi, dl_SMICA)
        ax.set_ylabel('$\ell$')
        ax.set_title("Angular Power Spectra")
        ax.legend(loc="upper right")
        #     ax.set_yscale("log")
        ax.set_xlim(10, xmax)
        plt.ylim(1E-10, 1)
        ax.grid()
        plt.show()
        return cl_SMICA, dl_SMICA, ell

    def get_dl(self, fcolors):
        cl_SMICA = hp.anafast(fcolors)
        ell = np.arange(len(cl_SMICA))
        pl = hp.sphtfunc.pixwin(nside=self.nside3D)
        # Deconvolve the beam and the pixel window function
        dl_SMICA = cl_SMICA / (B_l(10.0, ell) ** 2 * pl ** 2)
        dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi))
        dl_SMICA = dl_SMICA / np.max(dl_SMICA)
        return dl_SMICA, ell

    def plotdf(self, n):
        for i in n:
            self.df = self.df / np.expand_dims(self.df.std(axis=1), axis=1) * self.sigma_smica
            hp.mollview(self.df[i, :].squeeze(), title="{}".format(i), min=-2 * self.sigma_smica,
                        max=2 * self.sigma_smica, unit="K", cmap=cm.RdBu_r)
            hp.graticule()
            plt.show()

    def plot_ONLY_CL_From_Image(self, fcolors, xmax=30, filename=None):
        dl_SMICA_HU, ell = self.get_dl(fcolors.squeeze())
        dl_SMICA, ell = self.get_dl(self.SMICA_LR.squeeze())
        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111)
        ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
        ax.plot(ell * np.pi, dl_SMICA, ell * np.pi, dl_SMICA_HU)
        ax.set_ylabel('$\ell$')
        ax.set_title("Angular Power Spectra {}_{}".format(self.kmax, self.nside3D))
        ax.legend(loc="upper right")
        ax.set_xlim(1, xmax)
        plt.ylim(1E-10, 1)
        ax.grid()
        plt.savefig("./img1/AngularPowerSpectra_{}_{}.png".format(kmax, nside3D), dpi=300)
        plt.show()
        return dl_SMICA, dl_SMICA_HU, ell

    def plotSH(self, k1, kmax, l, m, ll, lm):
        self.reset_PGKmax(k1, kmax, self.lambda_k, ll, lm)
        fcolors = self.spherharm(l, m)
        hp.mollview(fcolors.squeeze(), title="{}_{}_{}_{}".format(l, m, ll, lm), cmap=cm.RdBu_r)
        hp.graticule()
        plt.show()
        return fcolors

    def project4D3d_by_k(self, k1, kmax, lk, ll, lm, filename=None, title=None, plotme=False, save=False):
        self.calc_all_hyperharm(k, k1, kmax, nside3D, lk, ll, lm)
        self.df = (self.df - np.expand_dims(self.df.mean(axis=1), axis=1)) / np.expand_dims(self.df.std(axis=1),
                                                                                            axis=1) * self.sigma_smica
        C = self.df.dot(self.SMICA_LR)
        B = self.df.dot(self.df.T)
        results = np.linalg.solve(B, C)
        newmap = results.T.dot(self.df)
        mu, sigma = norm.fit(newmap)
        newmap = (newmap - mu) / sigma * self.sigma_smica
        err = (1.0 - np.correlate(newmap.squeeze(), self.SMICA_LR.squeeze()) * 1e4)[0]
        if plotme or save:
            if filename is None:
                filename = "./img1/aitoff_{}_{}_{}_{}_{}.png".format(kmax, np.round(self.lambda_k, 3),
                                                                     np.round(self.lambda_l, 3),
                                                                     np.round(self.lambda_m, 3),
                                                                     np.round(err * 1, 3))
            self.plot_aitoff(newmap, self.lambda_k, self.lambda_l,
                             self.lambda_m, err, title=title, filename=filename, plotme=plotme, save=save)
        return results, newmap, err

    def project4D3d(self, k1, kmax, lk, ll, lm, filename=None, title=None, plotme=False, save=False):
        self.calc_hyperharmlong(k1, kmax, lk, ll, lm)
        self.df = (self.df - np.expand_dims(self.df.mean(axis=1), axis=1)) / np.expand_dims(self.df.std(axis=1),
                                                                                            axis=1) * self.sigma_smica
        C = self.df.dot(self.SMICA_LR)
        B = self.df.dot(self.df.T)
        results = np.linalg.solve(B, C)
        newmap = results.T.dot(self.df)
        mu, sigma = norm.fit(newmap)
        newmap = (newmap - mu) / sigma * self.sigma_smica
        err = (1.0 - np.correlate(newmap.squeeze(), self.SMICA_LR.squeeze()) * 1e4)[0]
        if plotme or save:
            if filename is None:
                filename = "./img1/aitoff_{}_{}_{}_{}_{}.png".format(kmax, np.round(self.lambda_k, 3),
                                                                     np.round(self.lambda_l, 3),
                                                                     np.round(self.lambda_m, 3),
                                                                     np.round(err * 1, 3))
            self.plot_aitoff(newmap, self.lambda_k, self.lambda_l,
                             self.lambda_m, err, title=title, filename=filename, plotme=plotme, save=save)
        return results, newmap, err

    def project4D3d_err(self, k1, kmax, lk, ll, lm):
        self.calc_hyperharmlong(k1, kmax, lk, ll, lm)
        self.df = (self.df - np.expand_dims(self.df.mean(axis=1), axis=1)) / np.expand_dims(self.df.std(axis=1),
                                                                                            axis=1) * self.sigma_smica
        C = self.df.dot(self.SMICA_LR)
        B = self.df.dot(self.df.T)
        results = np.linalg.solve(B, C)
        newmap = results.T.dot(self.df)
        mu, sigma = norm.fit(newmap)
        newmap = (newmap - mu) / sigma * self.sigma_smica
        err = (1.0 - np.correlate(newmap.squeeze(), self.SMICA_LR.squeeze()) * 1E4)[0]
        return err

    def fit_K_range(self, k1, k2, nside3D, x0):
        filename = "./img1/{}_{}_SMICA.png".format(kmax, nside3D)
        title = "kmax={}_nside3D={}={}_SMICA".format(kmax, nside3D)
        self.SMICA_LR = self.change_resolution(self.SMICA, sigma_smica, nside=nside3D,
                                               lmax=nside3D, title=title, filename=filename, plotme=True).astype(
            np.float32)
        results, fcolors, ls, err = self.project4D3d(k1, k2, x0[0], x0[1], x0[2], plotme=True)
        self.plotHistogram(fcolors, nside3D, kmax)
        self.plot_ONLY_CL_From_Image(fcolors.squeeze(), xmax=10 * k2)
        newSMICA = self.SMICA - fcolors
        self.SMICA_LR = self.change_resolution(newSMICA, 1, nside=nside3D, max=nside3D,
                                               title=title, filename=filename, plotme=True).astype(np.float32)
        return results.x

    def calcStuff(self, alpha):
        err = 0.0
        results = 0.0
        prime = alpha[0]
        kmax = alpha[1]
        lk = prime[0]
        ll = prime[1]
        lm = prime[2]
        start_time = time()
        try:
            results, fcolors, ls, err = self.project4D3d(kmax, lk, ll, lm, plotme=True)
        except Exception as exception_object:
            print('Error with getting map for: {0}_{0}_{0}'.format(lk, ll, lm))
        stop_time = time()
        print(alpha, err)
        return [start_time, stop_time, err, prime, results]

    def mindict(self, test_dict):
        # Using min() + list comprehension + values()
        # Finding min value keys in dictionary
        temp = min(test_dict.values())
        res = [key for key in test_dict if test_dict[key] == temp]
        return res[0], test_dict[res[0]]

    def createDFMapping(self, k1, kmax):
        self.my_klm_dict = {}
        idd = 0
        for k in np.arange(k1, kmax + 1):
            for l in np.arange(1, k):
                for m in np.arange(-l, l + 1):
                    self.my_klm_dict[(k, l, m)] = idd
                    idd += 1
        self.df = np.zeros([idd, len(self.xx), ], dtype=np.float32)

    def searchPlot(self, lambda_k, lambda_l, lambda_m, k1, kmax, plotme=False, save=False):
        ind = []
        err = 0.0
        results = 0.0
        df = 0.0
        for lk in lambda_k:
            for ll in lambda_l:
                for lm in lambda_m:
                    start_time = time()
                    try:
                        results, fcolors, err = self.project4D3d(k1, kmax, lk, ll, lm, plotme=plotme, save=save)
                    except Exception as aa:
                        print('Error with getting map for: {}_{}_{}'.format(lk, ll, lm))
                    stop_time = time()
                    ind.append([start_time, stop_time, err, (lk, ll, lm), results])
        return ind, df

    def calcError(self, x, k1, kmax):
        t1 = datetime.now()
        err = self.project4D3d_err(k1, kmax, [x[0]], [x[1]], [x[2]])
        print(x, err, (datetime.now() - t1).microseconds)
        return err

    def find_neighborhood(self, n, k1, kmax, nside3D, k0=0, l0=0, m0=0, plot=False, save=False):
        self.change_SMICA_resolution(kmax, nside3D, self.sigma_smica)
        lambda_k = np.linspace(k0, 2 * np.pi + k0, n)
        lambda_l = np.linspace(l0, 2 * np.pi + l0, n)
        lambda_m = np.linspace(m0, 2 * np.pi + m0, n)
        errarray, df = self.searchPlot(lambda_k, lambda_l, lambda_m, k1, kmax, plotme=plot, save=save)
        self.creategiff(kmax, nside3D)
        minvalue = 10000
        indmin = 0
        for x in errarray:
            if x[2] < minvalue:
                minvalue = x[2]
                indmin = x[3]
        print(minvalue, indmin)
        x0 = np.array(indmin)
        return errarray, x0

    def reset_PGKmax(self, lk, ll, lm, k1, kmax):
        self.createDFMapping(k1, kmax)
        self.change_HSH_center(lk, ll, lm, k1, kmax)

    def save_PGKmax(self, lk, ll, lm, k1, kmax, nside3D):
        filename_P = "./PG_data/P_{}_{}_{}_{}_{}".format(kmax, nside3D, np.round(lk, 3),
                                                         np.round(ll, 3),
                                                         np.round(lm, 3))
        filename_G = "./PG_data/G_{}_{}_{}_{}_{}".format(kmax, nside3D, np.round(lk, 3),
                                                         np.round(ll, 3),
                                                         np.round(lm, 3))
        np.save(filename_P, self.P, allow_pickle=True)
        np.save(filename_G, self.G, allow_pickle=True)

    def getMaxKmaxNside(self, mypath="./PG_data", prefix="P"):
        filelist = [join(mypath, f) for f in sorted(listdir(mypath)) if
                    isfile(join(mypath, f)) and f.startswith(prefix)]

        info = [f.split("_") for f in filelist]
        numbs = []
        for f in info:
            dd = []
            for g in f:
                bb = [int(x) for x in g if x.isdigit()]
                cc = []
                a = 0
                for x in bb:
                    a = int(10 * a) + x
                cc.append(a)
                print(cc)
                dd.append(cc)
            numbs.append(dd)
        numbs = np.array(numbs)
        return numbs.squeeze().T[1, :].max(), numbs.squeeze().T[2, :].max()

    def load_PGKmax(self, lk, ll, lm, k1, kmax, nside3D):
        oldfilename_P = "./PG_data/P_{}_{}_{}_{}_{}.pny".format(kmax, nside3D, np.round(lk, 3),
                                                                np.round(ll, 3),
                                                                np.round(lm, 3))
        oldfilename_G = "./PG_data/G_{}_{}_{}_{}_{}.pny".format(kmax, nside3D, np.round(lk, 3),
                                                                np.round(ll, 3),
                                                                np.round(lm, 3))
        if os.path.exists(oldfilename_G):
            self.G = np.load(oldfilename_G, allow_pickle=True)
        if os.path.exists(oldfilename_P):
            self.G = np.load(oldfilename_P, allow_pickle=True)

    def change_HSH_center(self, lk, ll, lm, k1, kmax, doit=False):
        if (self.lambda_k, self.lambda_l, self.lambda_m, self.k1, self.kmax) == (lk, ll, lm, k1, kmax) and not doit:
            return
        self.lambda_k = lk
        self.lambda_l = ll
        self.lambda_m = lm
        z = self.zz + lk
        y = self.yy + ll
        x = self.xx + lm
        cosksi = np.cos(z)
        self.sinksi = np.sin(z)
        self.costheta = np.cos(y)
        gegenbauer_keys = [(k, l) for (k, l, m) in self.my_klm_dict.keys() if m >= 0]
        self.phi = x
        self.G = {(k, l): eval_gegenbauer(1 + l, k - l, cosksi) for (k, l) in gegenbauer_keys}
        self.P = np.zeros([self.xx.shape[0], kmax + 1, kmax + 1])
        for i, x in enumerate(self.costheta):
            self.P[i, :, :] = legendre(kmax, x)

    def calc_hyperharmlong(self, k1, kmax, lk, ll, lm):
        self.reset_PGKmax(lk, ll, lm, k1, kmax, )
        for (k, l, m), id in self.my_klm_dict.items():
            self.df[id, :] = (-1) ** k * self.sinksi ** l * self.G[(k, l)] * self.spherharm(l, m)

    def calc_all_hyperharm(self, k, k1, kmax, nside3D, lk, ll, lm):
        if nside3D != self.nside3D:
            self.xx, self.yy, self.zz = hp.pix2vec(nside=nside3D, ipix=np.arange(hp.nside2npix(nside3D)))
        if (k1, kmax) != (self.k1, self.kmax):
            self.k1 = k1
            self.kmax = kmax
            if k > kmax:
                print("k passed kmax")
                return

        z = self.zz + lk
        y = self.yy + ll
        x = self.xx + lm
        cosksi = np.cos(z)
        self.sinksi = np.sin(z)
        self.costheta = np.cos(y)
        self.phi = x
        self.G = {(k, l): eval_gegenbauer(1 + l, k - l, cosksi) for l in np.arange(k + 1)}
        self.P = np.zeros([self.xx.shape[0], k + 1, k + 1])
        for i, x in enumerate(self.costheta):
            self.P[i, :, :] = legendre(k, x)
        # Calculate all eigenfunctions for k
        self.df[k - k1, :] = np.zeros(self.xx.shape)
        for l in np.arange(k + 1):
            sphermL = np.zeros(self.xx.shape)
            for m in np.arange(-l, l + 1):
                sphermL = sphermL + self.spherharm(l, m)
            self.df[k - k1, :] = self.df[k, :] + (-1) ** k * self.sinksi ** l * \
                                 self.G[(k, l)] * sphermL

    def spherharm(self, l, m):
        a = (-1) ** np.abs(m) / np.sqrt(4 * np.pi)
        b = 0
        if m > 0:
            b = a * cu.asnumpy(cu.array(self.P[:, l, np.abs(m)]) * np.cos(m * cu.array(self.phi)))
        if m == 0:
            b = a * self.P[:, l, np.abs(m)]
        if m < 0:
            b = a * cu.asnumpy(cu.array(self.P[:, l, np.abs(m)]) * np.sin(np.abs(m) * cu.array(self.phi)))
        return b

    def cleanup(self, mypath, prefix):
        filelist = [join(mypath, f) for f in sorted(listdir(mypath)) if
                    isfile(join(mypath, f)) and f.startswith(prefix)]
        for f in filelist:
            if os.path.exists(f):
                os.remove(f)
            else:
                print("The file does not exist")


if __name__ == "__main__":
    n = 5
    nside3D = 64
    k1 = 2
    kmax = 7
    k0, l0, m0 = (0.1, 0.2, 0.3)  # 299
    # these three indices are related to position within the hyperspherical hypersurface.  Don't confuse them with the
    x0 = (1.3566370614359173, 3.969911184307752, 0.3)
    mypath = "/media/home/mp74207/GitHub/CMB_HU/sphericalharmonics/Big_KLM"
    myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, k1, kmax)
    #################################################################
    # Find Neighborhood
    findneighborhood = True
    if findneighborhood:
        errarray, x0 = myHyper.find_neighborhood(n, k1, kmax, nside3D, k0, l0, m0, plot=False, save=True)
        myHyper.pickledict(errarray, filename='./errarray.pickle')
        x0 = minimize(myHyper.calcError, x0, args=(k1, kmax), method='nelder-mead',
                      options={'xatol': 1e-4, 'disp': True})
        np.save("./img1/x0_{}_{}.npy".format(kmax, nside3D), x0.x, allow_pickle=True)
        (k, l, m) = x0.x
        filename = "./img1/Best_{}_{}__{}_{}_{}_{}.png".format(kmax, nside3D, np.round(k, 1),
                                                               np.round(l, 1), np.round(m, 1),
                                                               np.round(x0.fun * 1, 4))
        title = "Best_{}_{}__{}_{}_{}_{}".format(kmax, nside3D, np.round(k, 1),
                                                 np.round(l, 1), np.round(m, 1),
                                                 np.round(x0.fun * 1, 4))
        results, newmap, err = myHyper.project4D3d(k1, kmax, k, l, m, filename=filename, title=title, plotme=True,
                                                   save=True)
        sound.play()
        myHyper.cleanup(mypath="./img1", prefix="aitoff")
        np.save("./img1/results_{}_{}.npy".format(kmax, nside3D), results, allow_pickle=True)
        # Final Result for nside3D=48, kmax=15
        # final_simplex: (array([[1.3286744, 5.27623442, 0.30042731],
        #                        [1.3286744, 5.27623442, 0.30042731],
        #                        [1.3286744, 5.27623442, 0.30042731],
        #                        [1.3286744, 5.27623442, 0.30042731]]),
        #                 array([-1.22970653, -1.22967672, -1.22966576, -1.22966123]))
        # fun: -1.2297065258026123
        #     message: 'Optimization terminated successfully.'
        #     nfev: 296
        #     nit: 113
        # status: 0
        # success: True
        # x: array([1.3286744, 5.27623442, 0.30042731])
    #################################################################
    else:
        x0 = [1.3286744, 5.27623442, 0.30042731]
        (lk, ll, lm) = x0
        t0 = datetime.now()
        kmax = 80
        nside3D = 32
        # myHyper.reset_PGKmax(lk, ll, lm, k1=2, kmax=9, nside3D=32, save=True)
        # myHyper.save_PGKmax( lk, ll, lm, k1, kmax, nside3D)
        g_kmax, g_nside3D = myHyper.getMaxKmaxNside(mypath="./PG_data", prefix="G")
        myHyper.load_PGKmax(lk, ll, lm, k1, g_kmax, g_nside3D)
        print((datetime.now() - t0).seconds)
        myHyper.change_SMICA_resolution(kmax, nside3D, myHyper.sigma_smica)
        myHyper.createDFMapping(k1, kmax)
        results, fcolors, err = myHyper.project4D3d(k1, kmax, x0[0], x0[1], x0[2], plotme=True, save=True)
        myHyper.plotHistogram(fcolors.squeeze(), nside3D, kmax)
        dl_SMICA, dl_SMICA_HU, ell = myHyper.plot_ONLY_CL_From_Image(fcolors.squeeze(), xmax=10 * kmax)
