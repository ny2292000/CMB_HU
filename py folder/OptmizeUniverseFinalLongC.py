import operator
import os
import warnings
from os import listdir
from os.path import isfile, join

import matplotlib.pylab as plt
from PIL import Image
from pyshtools.legendre import legendre
from scipy.optimize import minimize
from scipy.special import eval_gegenbauer

from lib3 import *
from parameters import *

warnings.filterwarnings("ignore")
import mpmath as mp
from enum import Enum
from time import sleep
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Gaussian1DKernel


class fitClass:
    def __init__(self):
        self.n = 6

    def six_peaks(self, t, *parkguess):
        'function of two overlapping peaks'
        p = np.zeros([1, len(t)])
        for i in np.arange(0, self.n * 4, 4):
            a0 = parkguess[i + 0] # peak height
            a1 = parkguess[i + 1] # Gaussian Center
            a2 = parkguess[i + 2] # std of gaussian
            gamma = parkguess[i + 3]  # std of gaussian
            p += self.fitfun(t, a0, a1, a2, gamma)
        return p[0]

    def fitfun(self, x, a0, a1, a2, gamma):
        return a0 * norm.pdf(x, loc=a1, scale=a2) * np.exp(-gamma * x) * np.sqrt(2 * np.pi)



def sindelta(t, *pars):
    'function of two overlapping peaks'
    A = pars[0]  # sin amplitude
    delta1 = pars[1]  # delta
    delta2 = pars[2]  # delta
    delta3 = pars[3]
    delta4 = pars[4]
    gamma = pars[5]  # gamma exponential damping term
    h0 = pars[6]
    nbits = pars[7]
    dd = delta1 + delta2 * t + delta3 * t ** 2  # +delta4*t**3
    data_1D = h0 * A * np.sin(dd) ** 2 * np.exp(gamma * dd)
    gauss_kernel = Gaussian1DKernel(45)
    smoothed_data_gauss = convolve(data_1D, gauss_kernel)
    return smoothed_data_gauss


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
plt.rcParams.update(params)


class Color(Enum):
    MINIMIZEPOSITION = 1
    FINDNEIGHBORHOOD = 2
    EVALUATE_DF_AT_POSITION = 3
    FINDBESTFORKRANGE = 4
    OPTIMIZE_SPECTRUM = 5
    CREATE_GAUSSIAN_BACKGROUND = 6
    WORK_86_128 = 7
    OPTIMIZE_SMICA_BACKGROUND = 8


def get_chopped_range(lmax, n=20, lmin=1):
    llist = sorted(list({int(np.round(x, 0)) for x in np.linspace(lmin, lmax, n)}))
    return llist


def sinxx(xx):
    y = np.zeros(xx.shape)
    for i, x in enumerate(xx):
        if x == 0:
            y[i] = 1.0
        else:
            y[i] = np.sin(x) / x
    return y


def optimzeSpectraC(df, x0, opt=False, kkk=None):
    def erroC(x, df):
        k = df.k - 2
        dk = x[0] * k + x[1] * k ** 2 + x[2] * k ** 3
        dl = x[3] * df.l + x[4] * df.l ** 2
        dm = x[5] * np.abs(df.m) + x[6] * df.m ** 2
        df["population"] = x[7] + dk + dl + dm
        dfG = pd.DataFrame(df.groupby(['k', "l"])["CL", "population"].apply(lambda x: x.astype(float).mean()),
                           columns=["CL", "population"])
        dfG["CL"] /= dfG["CL"].max()
        err = np.sum((dfG.population - dfG.CL) ** 2)
        print(err)
        return err

    if opt:
        if kkk:
            df = df[(df.k == kkk[0]) * (df.l == kkk[1])]
        xout = minimize(erroC, x0, args=(df), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        print("errC", xout.fun, xout.x)
        return xout.x, df
    else:
        x = x0
    ##########################################################
    if kkk:
        df = df[(df.k == kkk[0]) * (df.l == kkk[1])]
        plt.plot(df.m, df.coeff)
        plt.show()
    k = df.k - 2
    dk = x[0] * k + x[1] * k ** 2 + x[2] * k ** 3
    dl = x[3] * df.l + x[4] * df.l ** 2
    dm = x[5] * np.abs(df.m) + x[6] * df.m ** 2
    df["population"] = x[7] + dk + dl + dm
    dfG = pd.DataFrame(df.groupby(['k', "l"])["CL", "population"].apply(lambda x: x.astype(float).mean()),
                       columns=["CL", "population"])
    dfG["CL"] /= dfG["CL"].max()
    dfG["difference"] = (dfG.CL - dfG.population) / dfG.population
    dfG["zero"] = 0.0
    dfG["one"] = 1.0
    ##########################################################
    ax = dfG.CL.plot(legend=False)
    # dfG.population.plot( ax=ax)

    plt.xlabel('Hyperspherical Harmonic k')
    plt.title('Power Spectrum C')
    plt.ylabel("Hyperspherical Harmonic Mode Population")
    # plt.ylim(0, 2)
    plt.show()


def plot_aitoff(fcolors, cmap=cm.RdBu_r):
    hp.mollview(fcolors.squeeze(), cmap=cmap)
    hp.graticule()
    plt.show()

def plot_aitoff_df(l, m, df, cmap=cm.RdBu_r):
    def plot_orth(fcolors):
        sigma = np.std(fcolors)
        hp.orthview(fcolors, min=-2 * sigma, max=2 * sigma, title='Raw WMAP data', unit=r'$\Delta$T (mK)')
        plt.show()

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

def B_l(beam_arcmin, ls1):
    theta_fwhm = ((beam_arcmin / 60.0) / 180) * math.pi
    theta_s = theta_fwhm / (math.sqrt(8 * math.log(2)))
    return np.exp(-2 * (ls1 + 0.5) ** 2 * (math.sin(theta_s / 2.0)) ** 2)

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def chunks(lst, n, args=None):
    """Yield successive n-sized chunks from lst."""
    maxn = len(lst)
    for i in range(0, maxn, n):
        nextstep = i + n
        if nextstep > maxn:
            nextstep = maxn
        if args is not None:
            yield (i, args, lst[i:nextstep])
        else:
            yield (i, lst[i:nextstep])


def chg2ang(alpha):
    residue = alpha % (2 * np.pi)
    return int(residue * 180 / np.pi)

def newerr(x, SMICA_LR, newmap):
    newnewmap = SMICA_LR.squeeze() - x[1] * newmap.squeeze() - x[0]
    err = np.std(newnewmap)
    return err


def olderr(x, dl_SMICA, ell):
    # err = np.std(dl_SMICA- x[0]*np.sin(x[1]*ell)**2/ell**2+x[2]*ell**x[3])
    err = dl_SMICA - x[0] - x[1] * np.exp(ell ** x[2]) - x[3] * np.sin(ell * x[4]) ** 2 / (x[4] * ell) ** 2
    err = np.sum(err * err)
    print(err, x)
    return err


def functionG(a):
    l = a[0][0] - 1
    k = a[0][1] + l
    cosksi = a[1]
    A = np.array([eval_gegenbauer(1 + l, k - l, x) for x in cosksi])
    print(a[0], "done")
    return a[0], A


class HYPER:
    def __init__(self, nside3D, sigma_smica, planck_IQU_SMICA, karray,
                 lambda_k=0.0, lambda_l=0.0, lambda_m=0.0, bandwidth=256,
                 loadpriorG=False, savePG=False, longG=False):
        self.G = pd.DataFrame()
        self.loadpriorG = loadpriorG
        self.savePG = savePG
        self.extra_G_keys = {}
        self.extra_P_keys = {}
        self.longG = longG
        self.nside3D = nside3D
        self.bandwidth = bandwidth
        self.df = np.zeros([1, 1])
        self.results = {}
        self.sigma_smica = sigma_smica
        self.xx, self.yy, self.zz = hp.pix2vec(nside=nside3D, ipix=np.arange(hp.nside2npix(nside3D)))
        self.k1 = min(karray)
        self.kmax = max(karray)
        self.karray = karray
        self.mypath = "/media/home/mp74207/GitHub/CMB_HU/PG_data"
        self.G_filename = os.path.join(self.mypath, "G_{}_{}_{}_{}.npy".format(self.nside3D,
                                                                                  chg2ang(lambda_k),
                                                                                  chg2ang(lambda_l),
                                                                                  chg2ang(lambda_m)))
        self.P_filename = os.path.join(self.mypath, "P_{}_{}_{}_{}_".format(self.nside3D,
                                                                               chg2ang(lambda_k),
                                                                               chg2ang(lambda_l),
                                                                               chg2ang(lambda_m)))
        # self.G_filename = "./PG_data/G_64_492_495_709.npy"
        if loadpriorG:
            if os.path.exists(self.G_filename):
                print("loading from ", self.G_filename)
                self.G = pd.read_pickle(self.G_filename)
        self.sinksi = np.zeros([1, 1])
        self.cosksi = np.zeros([1, 1])
        self.phi = np.zeros([1, 1])
        self.x = np.zeros([1, 1])
        self.y = np.zeros([1, 1])
        self.z = np.zeros([1, 1])
        self.costheta = np.zeros([1,1])
        self.lambda_k, self.lambda_l, self.lambda_m = (lambda_k, lambda_l, lambda_m)
        self.SMICA = self.normalizeFColors(planck_IQU_SMICA, sigma_smica)
        self.SMICA_LR = self.SMICA
        self.newmap = np.zeros(self.SMICA.shape)
        self.change_SMICA_resolution(nside3D, doit=True, bandwidth=bandwidth)
        self.change_HSH_center(self.lambda_k, self.lambda_l, self.lambda_m, self.karray,
                               self.nside3D, loadpriorG=loadpriorG, doit=True, savePG=savePG)

    def change_SMICA_resolution(self, nside3D, doit=False, bandwidth=256):
        if (self.nside3D == nside3D) and not doit:
            return
        filename = "./img1/SMICA_{}_{}.png".format(nside3D, bandwidth)
        title = "SMICA_nside3D={}_{}".format(nside3D, bandwidth)
        self.SMICA_LR = self.change_resolution(self.SMICA, nside3D=nside3D, bandwidth=bandwidth,
                                               title=title, filename=filename,
                                               plotme=True, save=True)

    def factorMP(self, k, l, m):
        m = np.abs(m)
        a = (-1) ** m * np.sqrt((2 * l + 1) / (2 * np.pi) * mp.factorial(l - m) / mp.factorial(l + m))
        b = (-1) ** k * np.sqrt(
            2 * (k + 1) / np.pi * mp.factorial(k - l) / mp.factorial(k + l + 1) * 2 ** (2 * l) * mp.factorial(l) ** 2)
        c = float(a * b)
        return c

    def normalizeFColors(self, fcolors, sigma_smica):
        (mu, sigma) = norm.fit(fcolors)
        if sigma == 0.0:
            return fcolors
        return sigma_smica / sigma * (fcolors - mu)

    def getSpectralFiles(self, mypath):
        return [join(mypath, f) for f in sorted(listdir(mypath)) if isfile(join(mypath, f))]

    def creategiff(self, kmax, nside3D, mypath="./img1", prefix="aitoff_", filename="CMB"):
        giffiles = self.getSpectralFiles(mypath)
        giffiles = [x for x in giffiles if prefix in x]
        images = []
        for ff in giffiles:
            with Image.open(ff).convert("RGB") as img:
                images.append(img)
        fname = os.path.join(mypath, filename) + '_{}_{}.gif'.format(kmax, nside3D)
        images[0].save(fname, save_all=True, append_images=images[1:], optimize=False, duration=1000, loop=0)

    def creategiffMem(self, x):
        # err, prime, results, fig
        images = [y[3] for y in x]
        titles = [y[1] for y in x]
        errs = [y[0] for y in x]
        for i, xx in enumerate(images):
            (lk, ll, lm) = [x for x in titles[i]]
            err = errs[i]
            filename = "./img1/aitoff_{}_{}_{}_{}.png".format(kmax, chg2ang(lk), chg2ang(ll), chg2ang(lm))
            # draw the canvas, cache the render
            xx.seek(0)
            im = Image.open(xx)
            im.save(filename)

    def change_resolution(self, fcolors, nside3D=1024, filename=None,
                          title=None, save=False, plotme=None, bandwidth=256):
        # noinspection PyUnresolvedReferences
        SMICA_alms = hp.map2alm(fcolors, lmax=bandwidth)
        fcolors = hp.alm2map(SMICA_alms, nside=nside3D)
        (mu, sigma) = norm.fit(fcolors)
        if sigma != 0.0:
            fcolors = (fcolors - mu) / sigma * self.sigma_smica
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

    def plot_aitoff(self, fcolors, kk, ll, mm, err, filename=None, title=None, plotme=False, save=False, nosigma=True):
        # noinspection PyUnresolvedReferences
        plt.clf()
        if title is None:
            title = "{}_{}_{}_{}".format(kmax, chg2ang(kk), chg2ang(ll), chg2ang(mm))

        f = plt.figure()
        if nosigma:
            hp.mollview(fcolors.squeeze(), title=title, min=-2 * self.sigma_smica,
                        max=2 * self.sigma_smica, unit="K", cmap=cm.RdBu_r)
        else:
            mu, sigma = norm.fit(fcolors.squeeze())
            hp.mollview(fcolors.squeeze(), title=title, min=mu - 2 * sigma,
                        max=mu + 2 * sigma, unit="K", cmap=cm.RdBu_r)
        hp.graticule()
        if save:
            plt.savefig(filename, format='png')
        if plotme:
            plt.show()
        f.clear()
        plt.close(f)

    def plot_Single_CL_From_Image(self, fcolors, nside, ymin=1E-6, ymax=1, xmax=300, log=False):
        cl_SMICA, dl_SMICA, ell = self.get_dl(fcolors, nside)
        dl_SMICA = dl_SMICA / np.max(dl_SMICA)
        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111)
        ax.plot(ell, dl_SMICA)
        ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
        ax.set_ylabel('$\ell$')
        ax.set_title("Angular Power Spectra")
        ax.legend(loc="upper right")
        if log:
            ax.set_yscale("log")
        ax.set_xlim(1, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid()
        plt.show()
        return cl_SMICA, dl_SMICA, ell

    def plot_CL_From_Image(self, fcolors, nside, planck_theory_cl, xmax=3000, ymax=1.0):
        cl_SMICA, dl_SMICA, ell = self.get_dl(fcolors, nside)
        dl_SMICA = dl_SMICA / np.max(dl_SMICA)
        planck_theory_cl[:, 1] = planck_theory_cl[:, 1] / np.max(planck_theory_cl[10:3000, 1])
        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111)
        ax.plot(planck_theory_cl[:, 0], planck_theory_cl[:, 1])
        ax1 = plt.twinx(ax)
        ax1.plot(ell, dl_SMICA)
        ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
        ax.set_ylabel('$\ell$')
        ax.set_title("Angular Power Spectra")
        ax.legend(loc="upper right")
        # ax1.set_yscale("log")
        ax.set_xlim(10, xmax)
        ax1.set_xlim(10, xmax)
        # ax1.set_ylim(1E-4, ymax)
        ax1.grid()
        plt.show()
        return cl_SMICA, dl_SMICA, ell

    def get_dl(self, fcolors, nside):
        cl_SMICA = hp.anafast(fcolors)
        ell = np.arange(len(cl_SMICA))
        pl = hp.sphtfunc.pixwin(nside=nside)
        # Deconvolve the beam and the pixel window function
        dl_SMICA = cl_SMICA / (B_l(10.0, ell) ** 2 * pl ** 2)
        dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi))
        return cl_SMICA, dl_SMICA, ell

    def plot_ONLY_CL_From_Image(self, fcolors, nside, smica, nsidesmica, xmax=30):
        cl_SMICA, dl_SMICA_HU, ell = self.get_dl(fcolors.squeeze(), nside=nside)
        cl_SMICA, dl_SMICA, ell1 = self.get_dl(smica.squeeze(), nside=nsidesmica)
        dl_SMICA_HU /= np.max(dl_SMICA_HU[0:len(ell) // 2])
        dl_SMICA /= np.max(dl_SMICA[0:len(ell1) // 2])
        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111)
        ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
        line, = ax.plot(ell1, dl_SMICA)
        line.set_label('SMICA')
        ax1 = plt.twinx(ax)
        line1, = ax1.plot(ell, dl_SMICA_HU)
        line1.set_label('HU')
        ax1.set_ylabel('$\ell$')
        ax1.set_title("Angular Power Spectra {}_{}".format(self.kmax, self.nside3D))
        ax1.legend(loc="upper right")
        ax1.set_xlim(1, xmax)
        plt.ylim(1E-10, 1)
        ax1.grid()
        plt.savefig("./img1/AngularPowerSpectra_{}_{}.png".format(kmax, nside3D), dpi=300)
        plt.show()
        return dl_SMICA, dl_SMICA_HU, ell

    def plotSH(self, l, m, ll, lm, pp):
        fcolors = self.spherharmm(l, m, self.phi, pp)
        hp.mollview(fcolors.squeeze(), title="{}_{}_{}_{}".format(l, m, chg2ang(ll), chg2ang(lm)), cmap=cm.RdBu_r)
        hp.graticule()
        plt.show()
        return fcolors

    def plotNewMap(self, newmap, err, filename=None, title=None, plotme=False, save=False, nosigma=True):
        err0 = np.round(err, 3)
        if filename is None:
            filename = "./img1/aitoff_{}_{}_{}__{}_{}_{}.png".format(self.kmax, self.nside3D, err0,
                                                                     chg2ang(self.lambda_k),
                                                                     chg2ang(self.lambda_l),
                                                                     chg2ang(self.lambda_m))
        if title is None:
            title = "{}_{}_{}__{}_{}_{}".format(self.kmax, self.nside3D, err0, chg2ang(self.lambda_k),
                                                chg2ang(self.lambda_l),
                                                chg2ang(self.lambda_m))
        self.plot_aitoff(newmap, self.lambda_k, self.lambda_l,
                         self.lambda_m, err0, title=title, filename=filename, plotme=plotme, save=save, nosigma=nosigma)

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
            results, fcolors, err = self.project4D3d(kmax)
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

    def calcError(self, x, karray):
        t1 = datetime.now()
        self.change_HSH_center(x[0], x[1], x[2], karray, nside3D, loadpriorG=False, doit=True, savePG=False)
        _, _, err = self.project4D3d(karray)
        print(x, err, (datetime.now() - t1).microseconds)
        return err

    def calcErrorDF(self, x):
        err = np.sum((np.dot(x.T, self.df[:, 4:]) - self.SMICA_LR.squeeze()) ** 2) * 1E6
        return err

    def cleanup(self, mypath, prefix):
        filelist = [join(mypath, f) for f in sorted(listdir(mypath)) if
                    isfile(join(mypath, f)) and f.startswith(prefix)]
        for f in filelist:
            if os.path.exists(f):
                os.remove(f)
            else:
                print("The file does not exist")

    # def get_gegenbauerAsyn(self):
    #     number_of_workers = 10
    #     multiindex = pd.MultiIndex.from_tuples(self.extra_G_keys(), names=('k', 'l'))
    #     columns = list(np.arange(len(self.cosksi)))
    #     if len(self.extra_G_keys()) == 0:
    #         return
    #     extraG = pd.DataFrame(data=np.zeros([len(multiindex), len(columns)]), index=multiindex, columns=columns)
    #     extra_G_keys = ((x,self.cosksi) for x in self.extra_G_keys())
    #     with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
    #         responses = executor.map(functionG, extra_G_keys)
    #     for response in responses:
    #         k = response[0][0]
    #         l = response[0][1]
    #         extraG.loc[(1+l,k-l),:] = response[1]
    #         print("transferred", k,l)
    #     return extraG

    def get_gegenbauerAsyn(self):
        multiindex = pd.MultiIndex.from_tuples(self.extra_G_keys(), names=('k', 'l'))
        columns = list(np.arange(len(self.cosksi)))
        if len(self.extra_G_keys()) == 0:
            return
        extraG = pd.DataFrame(data=np.zeros([len(multiindex), len(columns)]), index=multiindex, columns=columns)
        for key in self.extra_G_keys():
            a, extraG.loc[key, :] = functionG((key, self.cosksi))
            print(a, "transferred")
        return extraG

    def change_HSH_center(self, lk, ll, lm, karray, nside, doit=False, loadpriorG=False, savePG=False):
        kmax = max(karray)
        if not doit:
            array_number_match = len(self.karray) == len(karray)
            if array_number_match:
                array_match = np.sum(self.karray == karray) != 0
            else:
                array_match = False
            if (self.lambda_k, self.lambda_l, self.lambda_m, self.kmax) == (lk, ll, lm, kmax) \
                    and array_number_match and array_match:
                return
        self.loadpriorG = loadpriorG
        self.savePG = savePG
        if not loadpriorG:
            self.G = pd.DataFrame()
        self.nside3D = nside
        self.xx, self.yy, self.zz = hp.pix2vec(nside=nside, ipix=np.arange(hp.nside2npix(nside)))
        self.lambda_k = lk
        self.lambda_l = ll
        self.lambda_m = lm
        self.z = self.zz + lk
        self.y = self.yy + ll
        self.x = self.xx + lm
        self.cosksi = np.cos(self.z)
        self.sinksi = np.sin(self.z)
        self.costheta = np.cos(self.y)
        self.phi = self.x
        G = {}
        listofk = sorted(karray)
        for k in listofk:
            for l in np.arange(1, k):
                if (1 + l, k - l) not in self.G.index:
                    G[1 + l, k - l] = 1
        if len(G) != 0:
            self.extra_G_keys = G.keys
            if self.G.shape[0] == 0:
                self.G = self.get_gegenbauerAsyn()
            else:
                self.G = pd.concat([self.G, self.get_gegenbauerAsyn()], axis=0)
        if self.savePG:
            self.G.to_pickle(self.G_filename)
        # kjump = 20
        # ibeginning=0
        # for i in np.arange(kjump, self.kmax, kjump):
        #     newPfile = self.P_filename + "{}.npy".format(i)
        #     if not os.path.exists(newPfile):
        #         pp = get_legendre((0,np.arange(ibeginning,i), self.costheta))[1].squeeze()
        #         np.save(newPfile,pp)
        #         ibeginning=i

    # def get_P(self, l, costheta):
    #     pp = np.zeros([0, kmax + 1, kmax + 1])
    #     lencos = len(costheta)
    #     number_of_workers = 10
    #     mychunks = chunks(costheta, lencos // number_of_workers, l)
    #     P = {}
    #     with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
    #         responses = executor.map(get_legendre, mychunks)
    #     for response in responses:
    #         i = response[0]
    #         P[i] = response[1]
    #     for key in np.sort(list(P.keys())):
    #         pp = np.concatenate([pp, P[key]], axis=0)
    #     return pp

    def createGaussianBackground(self, x0, nside, delta, karray):
        xx, yy, zz = hp.pix2vec(nside=nside, ipix=np.arange(hp.nside2npix(nside)))
        x, y, z = tuple(map(operator.add, (xx, yy, zz), x0))
        cosksi = np.cos(z)
        sinksi = np.sin(z)
        costheta = np.cos(y)
        phi = x
        ############################################################
        ############################################################
        # random amplitudes
        wavefunction = np.zeros(xx.shape)
        t1 = datetime.now()
        print((datetime.now() - t1).seconds)
        df = np.array([len(karray), len(xx)])
        self.kmax = np.max(karray)
        self.calc_hyperharmnano0(karray)

        ############################################################
        return wavefunction

    def project4D3d_1(self, karray):
        self.kmax = np.max(karray)
        self.calc_hyperharmnano1(karray)
        C = np.dot(self.df[:, 4:], self.SMICA_LR)
        B = np.dot(self.df[:, 4:], self.df[:, 4:].T)
        results = np.linalg.solve(B, C)

        self.newmap = np.dot(results.T, self.df[:, 4:])
        mu, sigma = norm.fit(self.newmap)
        self.newmap = (self.newmap - mu) / sigma * self.sigma_smica
        err = (1.0 - np.correlate(self.newmap.squeeze(), self.SMICA_LR.squeeze()) * 1e4)[0]
        return results, self.newmap, err

    def project4D3d_0(self, karray):
        self.kmax = np.max(karray)
        self.calc_hyperharmnano2(karray)
        C = np.dot(self.df[:, 4:], self.SMICA_LR)
        B = np.dot(self.df[:, 4:], self.df[:, 4:].T)
        results = np.linalg.solve(B, C)

        self.newmap = np.dot(results.T, self.df[:, 4:])
        mu, sigma = norm.fit(self.newmap)
        self.newmap = (self.newmap - mu) / sigma * self.sigma_smica
        err = (1.0 - np.correlate(self.newmap.squeeze(), self.SMICA_LR.squeeze()) * 1e4)[0]
        return results, self.newmap, err

    def calc_hyperharmnano0(self, karray):
        nnn = 20
        kmax = np.max(karray)
        npoints = len(self.xx)
        jj = 0
        for k in karray:
            llist = sorted(list(set(int(np.round(kk, 0)) for kk in np.geomspace(1, k - 1, nnn))))
            for l in llist:
                jj += 1
        self.df = np.zeros([jj, 4 + npoints])
        pp = np.array([legendre(kmax - 1, x, csphase=-1) for x in self.costheta])
        jj = 0
        for k in karray:
            llist = sorted(list(set(int(np.round(kk, 0)) for kk in np.geomspace(1, k - 1, nnn))))
            jj = self.calc_hyperharmnano1(k, llist, pp, jj)

    def calc_hyperharmnano1(self, k, llist, pp, jj):
        ##############################################################
        ##############################################################
        newG = 0
        for ii, l in enumerate(llist):
            if (1 + l, k - l) not in self.G.index:
                aa, self.G[1 + l, k - l] = functionG(((1 + l, k - l), self.cosksi))
                newG += 1
            if (1 + l, k - l) not in self.G.index:
                print("missing", 1 + l, k - l)
            a = self.sinksi ** l * self.G.loc[(1 + l, k - l), :].values
            a1 = (-1) ** k * np.sqrt(2 * (k + 1) / np.pi * mp.factorial(k - l)
                                     * 2 ** (2 * l) * mp.factorial(l) ** 2 / mp.factorial(k + l + 1))
            a1 = float(a1)
            mlist = sorted(list(set(int(np.round(kk, 0)) for kk in np.linspace(-l, l, len(llist)))))
            b = np.zeros(self.xx.shape)
            for m in mlist:
                b += a * a1 * self.spherharmm(l, m, self.phi, pp[:, l, np.abs(m)])
            c = b.std()
            mu = b.mean()
            if c != 0:
                self.df[jj, 0] = k
                self.df[jj, 1] = l
                self.df[jj, 2] = 0
                self.df[jj, 3] = c
                self.df[jj, 4::] = b / c
                jj += 1
            else:
                print("failed ", k, l)
        print(jj, self.df.shape)
        if newG != 0:
            self.G.to_pickle(self.G_filename)
        return jj

    def calc_hyperharmnano2(self, karray):
        npoints = len(self.xx)
        nnn = 20
        G = {}
        listofk = sorted(karray)
        lmax = np.max(karray)
        pp = np.array([legendre(lmax, x, csphase=-1) for x in self.costheta])
        ##############################################################
        gg = 0
        for k in listofk:
            llist = sorted(list(set(int(np.round(kk, 0)) for kk in np.geomspace(1, k - 1, nnn))))
            for ii, l in enumerate(llist):
                mlist = sorted(list(set(int(np.round(kk, 0)) for kk in np.linspace(-l, l, nnn))))
                for m in mlist:
                    gg += 1
        self.df = np.zeros([gg, 4 + npoints])
        ##############################################################
        jj = 0
        for k in listofk:
            llist = sorted(list(set(int(np.round(kk, 0)) for kk in np.geomspace(1, k - 1, nnn))))
            for ii, l in enumerate(llist):
                if self.longG:
                    b = np.zeros(self.xx.shape)
                if (1 + l, k - l) not in self.G.index:
                    G[1 + l, k - l] = 1.0
                    print(k, l)
                    continue
                if (1 + l, k - l) not in self.G.index:
                    print("missing", 1 + l, k - l)
                a = self.sinksi ** l * self.G.loc[(1 + l, k - l), :].values
                a1 = (-1) ** k * np.sqrt(2 * (k + 1) / np.pi * mp.factorial(k - l)
                                         * 2 ** (2 * l) * mp.factorial(l) ** 2 / mp.factorial(k + l + 1))
                a1 = float(a1)
                mlist = sorted(list(set(int(np.round(kk, 0)) for kk in np.linspace(-l, l, nnn))))
                for m in mlist:
                    b = a * a1 * self.spherharmm(l, m, self.phi, pp[:, l, np.abs(m)])
                    c = b.std()
                    if c != 0:
                        self.df[jj, 0] = k
                        self.df[jj, 1] = l
                        self.df[jj, 2] = m
                        self.df[jj, 3] = c
                        self.df[jj, 4::] = b / c
                        jj += 1
                    else:
                        print("failed ", k, l, m)
        if len(G.keys()) != 0:
            self.extra_G_keys = G.keys
            print(G.keys())
            print("DO IT AGAIN")
        print(jj, gg, self.df.shape)

    def spherharmm(self, l, m, phi, pp):
        mm = np.abs(m)
        if m > 0:
            return pp * np.cos(mm * phi)
        if m == 0:
            return pp
        if m < 0:
            return pp * np.sin(mm * phi)

    def get_df_size(self, karray):
        npoints = len(self.xx)
        kmax = np.max(karray)
        jj = 0
        G = {}
        files = sorted([x for x in self.getSpectralFiles(self.mypath) if "/PG_data/P_{}_{}_{}_{}".format(self.nside3D,
                                                                                                         chg2ang(
                                                                                                             self.lambda_k),
                                                                                                         chg2ang(
                                                                                                             self.lambda_l),
                                                                                                         chg2ang(
                                                                                                             self.lambda_m)) in x])
        karray0 = [eval(x.replace(self.mypath, "").replace(".npy", "").split("_")[-1:][0]) for x in files]
        filesMap = {x: y for (x, y) in zip(karray0, files)}
        listofk = [x for x in sorted(list(filesMap.keys())) if x <= kmax]
        for k in listofk:
            pp = np.load(filesMap[k])
            llist = [np.count_nonzero(x, axis=0) for x in pp[0, :, :].squeeze()]
            pp = None
            llist = [x - 1 for x in llist if x < k + 1 and x > 1]
            for ii, l in enumerate(llist):
                if (1 + l, k - l) not in self.G.index:
                    G[1 + l, k - l] = 1.0
                for m in np.arange(-l, l + 1):
                    if not self.longG:
                        jj += 1
                if self.longG:
                    jj += 1
        self.df = np.zeros([jj, 4 + npoints])
        self.extra_G_keys = G.keys

    def plot_orth(self, fcolors):
        sigma = np.std(fcolors)
        hp.orthview(fcolors, min=-2 * sigma, max=2 * sigma, title='Raw WMAP data', unit=r'$\Delta$T (mK)')
        plt.show()

    def plot_aitoff_df(self, l, m, phi, pp=None, cmap=cm.RdBu_r):
        if pp is None:
            pp = np.array([legendre(l, x) for x in self.costheta])
        fcolors = self.spherharmm(l=l, m=m, phi=phi, pp=pp)
        sigma = np.std(fcolors)
        title = "{}_{}".format(l, m)
        hp.mollview(fcolors.squeeze(), title=title, min=-2 * sigma,
                    max=2 * sigma, unit="K", cmap=cm.RdBu_r)
        hp.graticule()
        self.plot_orth(fcolors)

    def optimizeNewMap(self, newmap0, SMICA_LR, xx0, nside3D, bandwidth, nosigma=True, mypath="./PG_data/"):
        newmap = xx0[1] * newmap0 + xx0[0]
        mu, sigma = norm.fit(newmap)
        newmap -= mu
        diffmap = SMICA_LR - newmap.squeeze()
        mu, sigma = norm.fit(diffmap)
        diffmap -= mu
        restoredmap = newmap + diffmap
        err = (1.0 - np.correlate(newmap, SMICA_LR) * 1e4)[0]
        return newmap, diffmap, restoredmap, err

    def matchSMICA(self, a, newmap):
        diffmap_1024 = a * self.SMICA.squeeze() + newmap * (1 - a)
        diffmap_1024 = (diffmap_1024 - np.mean(diffmap_1024)) / np.std(diffmap_1024) * self.sigma_smica
        myHyper.plotNewMap(diffmap_1024, err=0.0, plotme=True, title=str(a))


if __name__ == "__main__":
    myHyper = None
    # History
    # x0 = [2.11999902, 4.84227338, 4.95303838] # -1.4133892288576106 Best_48_48__121_277_283 June02
    # x0 = [4.14376593, 1.09484514, 1.89120155]
    # x0 = [4.92463563, 4.95336686, 0.8165214728204138]
    # x0 = [4.82349567, 4.88348719, 0.84566045]  # -3.442960596169308 9876
    # x0 = [2.13302707, 5.08554078, 4.34175881]  # -0.958704 [122.21, 291.38, 248.76]
    # 2.13302707, 5.08554078, 4.34175881 -2.037992624771841 Best_29_64__122_291_248
    # x0 = [4.92463563, 4.95336686, 7.09970678] #-0.26296188908503493 61732 Best_20_48__4.9_5.0_7.1_-0.2629
    # x0 = [4.92463563, 4.95336686, 7.09970678]  # -0.41013304308658216 Best_20_48__4.9_5.0_7.1_-0.4101
    # x0 = [2.0943951023931953, 2.0943951023931953, 0.0]
    #################################################################
    # Evidence of Base Spectrum independent - Two walkabouts led to the same position
    # x0 = [4.1887902047863905, 1.0471975511965976, 2.0943951023931953] # -1.3261511132450026
    # x0 = [4.14376593, 1.09484514, 1.89120155]  # -1.37423412
    # x0 = [4.1436085, 1.09556579, 1.94038023] - 1.3343069931542146
    # x0 = [4.15976051, 1.10351324, 1.98582181] #-1.392411053733134
    #################################################################
    #################################################################
    #################################################################
    #################################################################
    # Position of our Universe (error = -1.3924651085864181 for base 2:11)
    # x0 = [4.15976051, 1.10351324, 1.98582181] # in radians
    # x0 = [4.92463563, 4.95336686, 0.8165214728204138]  # [282.16, 283.81, 46.78]
    # x0 = [4.96195883, 4.95304836, 0.81803705] # [284.3, 283.79, 46.87]
    # x0 = (2.0943951023931953, 4.886921905584122, 4.886921905584122)
    # x0 = [2.11999902, 4.84227338, 4.95303838] # ksi, theta, phi = [121.47, 277.44, 283.79] value= -0.957923 June02
    # x0 = [4.14376593, 1.09484514, 1.89120155]
    # x0 = [2.0943951023931953, 5.235987755982988, 4.1887902047863905pt-]  # -0.9512365749096219 June 09 [120.0, 300.0, 240.0]
    # x0 = [2.13302806, 5.0855885,  4.21869283] # -0.958704 [122.21, 291.38, 241.71]
    # x0 = [2.13302707, 5.08554078, 4.34175881] # -0.958704 [122.21, 291.38, 248.76]
    # x0 = [4.82349567, 4.88348719, 0.84566045]  # -3.4429605 [276.37, 279.8, 48.45]
    # x0 = [4.92463563, 4.95336686, 7.09970678]  # In degrees  [282.16, 283.81, 46.78]
    x0 = [4.92463563, 4.95336686, 0.8165214728204138]  # In degrees  [282.16, 283.81, 46.78]
    # Earth Position
    y0 = [np.round((xx + 1) / np.pi * 180, 2) for xx in x0]
    print("Earth Position", y0)
    # In degrees  [238.34, 63.23, 113.78]
    # In degrees [120.0, 300.0, 240.0]
    todo = [Color.EVALUATE_DF_AT_POSITION, Color.OPTIMIZE_SMICA_BACKGROUND, Color.OPTIMIZE_SPECTRUM]
    if Color.EVALUATE_DF_AT_POSITION in todo:
        (lk, ll, lm) = x0

        print("evaluated at {}".format(x0))
        # Create karray
        # karray = list(np.arange(2, 49))
        # karray = list(np.arange(2, 10))

        # karray = list(np.arange(20, 80, 20))
        karray = list(np.arange(2, 50))
        karray = np.array(sorted(list(set(karray)))).astype(int)

        nside3D = 64
        bandwidth = 2048
        # kmax=20064
        # karray += sorted(list(set([int(k) for k in np.geomspace(49,kmax,20)])))
        # karray = np.array(sorted(list(set(karray)))).astype(int)
        print(len(karray))
        kmax = max(karray)
        #################################
        if myHyper is None:
            myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray,
                            lk, ll, lm, loadpriorG=True, savePG=True, bandwidth=bandwidth, longG=False)
        #################################################################
        results, newmap0, err = myHyper.project4D3d_0(karray)
        print(lk, ll, lm, err)
        #########################################################################
        np.save("./img1/SMICA_LR.npy", myHyper.SMICA_LR)
        #################################################################
        #################################################################
        x01 = np.array([0.00161687, 0.31435231])
        x00 = minimize(newerr, x01, args=(myHyper.SMICA_LR.squeeze(), newmap0.squeeze()),
                       method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        err = x00.fun
        xx0 = x00.x
        newmap, diffmap, restoredmap, err = myHyper.optimizeNewMap(newmap0.squeeze(), myHyper.SMICA_LR.squeeze(),
                                                                   xx0=xx0,
                                                                   nside3D=nside3D, bandwidth=bandwidth, nosigma=True)
        myHyper.plotHistogram(newmap.squeeze(), nside3D, kmax, plotme=True)
        #################################################################
        #################################################################
        #################################################################
        #########################################################################
        filename = "./img1/SingleBest_{}_{}_{}__{}_{}_{}.png".format(kmax, nside3D, bandwidth, chg2ang(lk),
                                                                     chg2ang(ll), chg2ang(lm))
        title = "Best_{}_{}_{}_{}__{}_{}_{}".format(kmax, nside3D, bandwidth, err, chg2ang(lk),
                                                    chg2ang(ll), chg2ang(lm))

        myHyper.plotNewMap(newmap, err, filename=filename, title=title, plotme=True, save=True)
        #########################################################################
        newmap_1024 = myHyper.change_resolution(newmap, nside3D=1024, bandwidth=64).squeeze()
        mu, sigma = norm.fit(newmap_1024)
        newmap_1024 = (newmap_1024 - mu) / sigma * myHyper.sigma_smica
        x01 = np.array([0.00072128, 0.47630094])
        x00 = minimize(newerr, x01, args=(myHyper.SMICA.squeeze(), newmap_1024.squeeze()),
                       method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        err = x00.fun
        xx0 = x00.x
        diffwmap_1024 = myHyper.SMICA.squeeze() - xx0[1] * newmap_1024.squeeze()

        filename = "./img1/DiffMap_{}_{}_{}__{}_{}_{}.png".format(kmax, nside3D, bandwidth, chg2ang(lk),
                                                                  chg2ang(ll), chg2ang(lm))
        title = "DiffMap_{}_{}_{}_{}__{}_{}_{}".format(kmax, nside3D, bandwidth, err, chg2ang(lk),
                                                       chg2ang(ll), chg2ang(lm))

        myHyper.plotNewMap(diffwmap_1024, err, filename=filename, title=title, plotme=True, save=True)
        np.save("./img1/newmap.npy", newmap)
        np.save("./img1/diffmap.npy", diffwmap_1024)
        ###############################################################
        ###############################################################
        ###############################################################
        print(lk, ll, lm, err)
        #################################################################
        np.save("./PG_data/resultsBest_{}_{}_{}".format(kmax, nside3D, bandwidth), results, allow_pickle=True)
        #################################################################
        #################################################################
    if Color.OPTIMIZE_SMICA_BACKGROUND in todo:
        cl_SMICA, dl_SMICA, ell = myHyper.plot_Single_CL_From_Image(diffwmap_1024, nside=1024, xmax=2000, ymax=0.5,
                                                                    log=False)
        x01 = np.array([5.88621273e-02, 9.60331959e-11, 3.89846976e-01, 1, 1])
        aa = 2000
        x = ell[aa:].squeeze()
        y = dl_SMICA[aa:].squeeze()
        x00 = minimize(olderr, x01, args=(y, x),
                       method='nelder-mead', options={'xatol': 1e-6, 'disp': True})
        err = x00.fun
        xx0 = x00.x
        yy = xx0[0] + xx0[1] * np.exp(x ** xx0[2]) + xx0[3] * np.sin(x * xx0[4]) ** 2 / (xx0[4] * x) ** 2
        dll_SMICA_Clean = dl_SMICA - xx0[0] - xx0[1] * np.exp(ell ** xx0[2])
        # a0 = pars[0]  # Amplitude
        # a1 = pars[1]  # center
        # a2 = pars[2]  # std
        # parguess = [326.07961712,   240.5495938,    130.86283279,
        #             185.2806624, 511.71738065,   106.82439558,
        #             654.09021179,   778.65904957, 195.83549305,
        #             50.093336,    1114.17415044,   102.05325759,
        #             105.70709484,  1272.23029976,   184.05430212,
        #             -1185.72051889, 666.00139292,   367.6208938 ]

        mygauss = fitClass()
        mygauss.n=5
        parguess = np.array([1.13141056e+02, 2.54610864e+02, 8.41771175e+01, 4.92402013e-03,
                             9.80458369e+02, 5.95410382e+02, 8.96299871e+01, 7.51841770e-03,
                             2.60752925e+01, 8.22194747e+02, 9.18280463e+01, 7.75142569e-04,
                             4.60311565e+01, 1.14621194e+03, 8.47451004e+01, 2.12699957e-03,
                             4.00537143e-02, 1.41213361e+03, 6.14342868e+01, -2.42490682e-03])
        # plt.plot(ell, mygauss.six_peaks(ell, *parguess), 'r-')
        # plt.show()
        # np.save("./PG_data/ell.npy", ell)
        # np.save("./PG_data/dll_SMICA_Clean.npy", dll_SMICA_Clean)
        popt, _ = curve_fit(mygauss.six_peaks, ell[0:2000], dll_SMICA_Clean[0:2000], parguess)
        print(popt)
        parguess = np.array(popt).reshape([5, 4])
        gamma= popt[-1:][0]
        parguess = parguess[parguess[:, 1].argsort()]
        centers = np.array([parguess[x, 1] for x in np.arange(5)])
        freqs = centers[1::] - centers[0:-1:]
        amplt = np.array([parguess[x, 0] for x in np.arange(5)])
        fitting1 = np.polyfit(centers, np.log(amplt), 1)
        yy = np.exp(fitting1[1]) * np.exp(fitting1[0] * ell)
        # First peak is at pi/2=l*Delta => Delta=pi/2/l
        delta = np.pi / 2 / centers[0]
        gamma1 = fitting1[0] / delta

        ################################################################
        ################################################################
        fig, axis1 = plt.subplots()
        axis1.plot(ell, dll_SMICA_Clean)
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
        ################################################################
        ################################################################
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
        freq = [centers[i + 1] - centers[i] for i in np.arange(5)]
        freqdiff = [x / freq[0] for x in freq]
        print(freq, freqdiff)
        print(amplt)
        ################################################################
        ################################################################
        fitting2 = np.polyfit(centers[0:-1:] * delta, freq, 2)
        plt.scatter(centers, amplt)
        plt.plot(ell, yy)
        plt.xlim([0, 2000])
        plt.ylim([0, None])
        plt.xlabel('Spherical Harmonic L')
        plt.ylabel('Intensity (arb. units)')
        plt.title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') % (
            delta, gamma))
        plt.legend(['Power Spectrum', 'Fitted Data'])
        plt.savefig('./img1/FreqFitteditted.png')
        plt.show()
        ################################################################
        ################################################################
        #    A = pars[0]  # sin amplitude
        # delta = pars[1]  # delta
        # gamma = pars[2]  #  gamma exponential damping term
        # h0 = pars[3]
        parguess = [1.75703174e+00,
                    -3.18905547e-01, 1.08942482e-02, -1.51470042e-05, 6.98842799e-09,
                    -7.17548356e-01, 5.52935906e-01, 2.00000000e+01]
        popt, _ = curve_fit(sindelta, ell[0:2000], dll_SMICA_Clean[0:2000], parguess)
        print(popt)
        parguess = np.array(popt)
        plt.figure()
        plt.plot(ell, dll_SMICA_Clean)
        plt.plot(ell, sindelta(ell, *popt), 'r-')
        plt.xlim([0, 2000])
        plt.ylim([0, None])
        plt.xlabel('Spherical Harmonic L')
        plt.ylabel('Intensity (arb. units)')
        plt.title("Modeling High-Fequency CMB Power Spectrum")
        plt.legend(['Power Spectrum', 'Fitted Data'])
        plt.savefig('./img1/HighFreqFittedPowerSpectrum.png')
        plt.show()
        ################################################################
        ################################################################
        aaaa = 1
    if Color.FINDNEIGHBORHOOD in todo:
        # these three indices are related to position within the hyperspherical hypersurface.  Don't confuse them with the
        # quantum numbers k,l,m
        x0 = [0.0, 0.0, 0.0]
        (lambda_k, lambda_l, lambda_m) = x0
        # Create karray
        nside3D = 48
        bandwidth = 64
        karray = list(np.arange(2, 10))
        print(len(karray))
        kmax = max(karray)
        #################################
        if myHyper is None:
            myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray,
                            lambda_k, lambda_l, lambda_m, loadpriorG=False,
                            savePG=False, bandwidth=bandwidth)
        else:
            myHyper.savePG = False
            myHyper.loadpriorG = False
        #################################################################
        # these three indices are related to position within the hyperspherical hypersurface.  Don't confuse them with the
        # quantum numbers k,l,m
        errarray = []
        olderr = 1110.0
        results = 0.0
        x00 = []
        nside3D = 48
        bandwidth = 48
        n = 7
        myHyper.change_SMICA_resolution(nside3D, doit=False, bandwidth=bandwidth)
        for lk in np.linspace(0, 2 * np.pi, n):
            for ll in np.linspace(0, 2 * np.pi, n):
                for lm in np.linspace(0, 2 * np.pi, n):
                    start_time = time()
                    try:
                        myHyper.change_HSH_center(lk, ll, lm, karray, nside3D, loadpriorG=False, doit=True,
                                                  savePG=False)
                        results, fcolors, err = myHyper.project4D3d(karray)
                        myHyper.plotNewMap(fcolors, err, filename=None, title=None, plotme=False, save=True)
                        if olderr > err:
                            olderr = err
                            err0 = np.round(err, 3)
                            filename = "./img1/Bestf_{}_{}_{}__{}_{}_{}.png".format(myHyper.kmax, myHyper.nside3D,
                                                                                    err0,
                                                                                    chg2ang(lk),
                                                                                    chg2ang(ll),
                                                                                    chg2ang(lm))
                            myHyper.plotNewMap(fcolors, err, filename=filename, title=None, plotme=True, save=True)
                            x00.append((lk, ll, lm, err))
                            np.save("./img1/x0_{}_{}.npy".format(kmax, nside3D), x00[-1:], allow_pickle=True)
                            print(lk, ll, lm, err)
                    except Exception as aa:
                        print('Error with getting map for: {}_{}_{}'.format(lk, ll, lm))
                    stop_time = time()
        myHyper.creategiff(kmax, nside3D, mypath="./img1", prefix="aitoff_", filename="CMB")
        sleep(10)
        myHyper.creategiff(kmax, nside3D, mypath="./img1", prefix="Bestf", filename="BEST")
        sleep(10)
        myHyper.cleanup(mypath="./img1", prefix="aitoff_")
        # myHyper.cleanup(mypath="./img1", prefix="Bestf")
        print("Best Position = ", x00[-1:])
        bestposition = "./img1/x0_{}_{}.npy".format(kmax, nside3D)
        np.save("./img1/x0_{}_{}.npy".format(kmax, nside3D), x00[-1:], allow_pickle=True)
        np.save("./img1/x00_{}_{}.npy".format(kmax, nside3D), x00, allow_pickle=True)
    #################################################################
    if Color.CREATEMAPOFUNIVERSE in todo:
            # these three indices are related to position within the hyperspherical hypersurface.  Don't confuse them with the
            # quantum numbers k,l,m
            x0 = [0.0, 0.0, 0.0]
            (lambda_k, lambda_l, lambda_m) = x0
            # Create karray
            nside3D = 48
            bandwidth = 64
            karray = list(np.arange(2, 10))
            print(len(karray))
            kmax = max(karray)
            #################################
            if myHyper is None:
                myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray,
                                lambda_k, lambda_l, lambda_m, loadpriorG=False,
                                savePG=False, bandwidth=bandwidth)
            else:
                myHyper.savePG = False
                myHyper.loadpriorG = False
            #################################################################
            # these three indices are related to position within the hyperspherical hypersurface.  Don't confuse them with the
            # quantum numbers k,l,m
            errarray = []
            olderr = 1110.0
            results = 0.0
            x00 = []
            nside3D = 48
            bandwidth = 48
            n = 7
            myHyper.change_SMICA_resolution(nside3D, doit=False, bandwidth=bandwidth)
            for lk in np.linspace(0, 2 * np.pi, n):
                for ll in np.linspace(0, 2 * np.pi, n):
                    for lm in np.linspace(0, 2 * np.pi, n):
                        start_time = time()
                        try:
                            myHyper.change_HSH_center(lk, ll, lm, karray, nside3D, loadpriorG=False, doit=True,
                                                      savePG=False)
                            results, fcolors, err = myHyper.project4D3d(karray)
                            myHyper.plotNewMap(fcolors, err, filename=None, title=None, plotme=False, save=True)
                            if olderr > err:
                                olderr = err
                                err0 = np.round(err, 3)
                                filename = "./img1/Bestf_{}_{}_{}__{}_{}_{}.png".format(myHyper.kmax, myHyper.nside3D,
                                                                                        err0,
                                                                                        chg2ang(lk),
                                                                                        chg2ang(ll),
                                                                                        chg2ang(lm))
                                myHyper.plotNewMap(fcolors, err, filename=filename, title=None, plotme=True, save=True)
                                x00.append((lk, ll, lm, err))
                                np.save("./img1/x0_{}_{}.npy".format(kmax, nside3D), x00[-1:], allow_pickle=True)
                                print(lk, ll, lm, err)
                        except Exception as aa:
                            print('Error with getting map for: {}_{}_{}'.format(lk, ll, lm))
                        stop_time = time()
            myHyper.creategiff(kmax, nside3D, mypath="./img1", prefix="aitoff_", filename="CMB")
            sleep(10)
            myHyper.creategiff(kmax, nside3D, mypath="./img1", prefix="Bestf", filename="BEST")
            sleep(10)
            myHyper.cleanup(mypath="./img1", prefix="aitoff_")
            # myHyper.cleanup(mypath="./img1", prefix="Bestf")
            print("Best Position = ", x00[-1:])
            bestposition = "./img1/x0_{}_{}.npy".format(kmax, nside3D)
            np.save("./img1/x0_{}_{}.npy".format(kmax, nside3D), x00[-1:], allow_pickle=True)
            np.save("./img1/x00_{}_{}.npy".format(kmax, nside3D), x00, allow_pickle=True)
    #################################################################
    if Color.MINIMIZEPOSITION in todo:
        # Best Position = [(2.0943951023931953, 4.886921905584122, 4.886921905584122, -0.9543073989083581)]
        # x0= x00[-1:][0][0:3]
        # x0 = (2.0943951023931953, 4.886921905584122, 4.886921905584122)
        # x0 = [2.11999902, 4.84227338, 4.95303838]  # ksi, theta, phi = [121.47, 277.44, 283.79]
        # x0 = [2.0943951023931953, 5.235987755982988, 4.1887902047863905] # -0.9512365749096219 June 09
        # x0 = [2.13302707, 5.08554078, 4.34175881]  # -0.958704 [122.21, 291.38, 241.71] June9 Minimized
        # x0 = [4.92463563, 4.95336686, 0.8165214728204138]
        # x0 = [4.82349567, 4.88348719, 0.84566045] #- 3.3740886193323982  696721
        # x0 = [4.82349567, 4.88348719, 0.84566045] #-3.442960596169308 9876
        # x0 = [4.92463563, 4.95336686, 0.8165214728204138]  # [282.16, 283.81, 46.78]
        x0 = [4.96195883, 4.95304836, 0.81803705]
        nside3D = 48
        bandwidth = 64
        (lambda_k, lambda_l, lambda_m) = x0
        # Create karray
        karray = list(np.arange(2, 10))
        print(len(karray))
        kmax = max(karray)
        #################################
        if myHyper is None:
            myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray,
                            lambda_k, lambda_l, lambda_m, loadpriorG=False, savePG=False, bandwidth=bandwidth)
        else:
            myHyper.loadpriorG = False
            myHyper.savePG = False
        #################################
        print(x0)
        x0 = minimize(myHyper.calcError, x0, args=(karray), method='nelder-mead',
                      options={'xatol': 1e-4, 'disp': True})
        np.save("./img1/x0_{}_{}.npy".format(kmax, nside3D), x0.x, allow_pickle=True)
        x0 = x0.x
        (lk, ll, lm) = x0
        print("minimized at {}".format(x0))
        filename = "./img1/Best_{}_{}_{}__{}_{}_{}.png".format(kmax, nside3D, bandwidth, chg2ang(lk), chg2ang(ll),
                                                               chg2ang(lm))
        title = "Best_{}_{}_{}__{}_{}_{}".format(kmax, nside3D, bandwidth, chg2ang(lk), chg2ang(ll), chg2ang(lm))
        myHyper.change_HSH_center(lk, ll, lm, karray, nside3D, loadpriorG=False, doit=True, savePG=True)
        results, newmap, err = myHyper.project4D3d_0(karray)
        np.save("./img1/results_{}_{}_{}.npy".format(kmax, nside3D, bandwidth), results, allow_pickle=True)
    #################################################################
    if Color.FINDBESTFORKRANGE in todo:
        # myHyper.phi, myHyper.theta, myHyper.ksi = hp.pix2vec(nside=nside3D, ipix=np.arange(hp.nside2npix(nside=nside3D)))
        # myHyper.theta, myHyper.phi = hp.pix2ang(nside=nside3D,
        #                                                      ipix=np.arange(hp.nside2npix(nside=nside3D)))
        # myHyper.costheta = np.cos(myHyper.theta)
        # k = 9
        # l = 3
        # for m in np.arange(-l+1,l):
        #     pp = np.array([legendre(l, x) for x in myHyper.costheta])
        #     myHyper.plot_aitoff_df(l,m,myHyper.phi, pp=pp)
        #
        # x0=np.load("./img1/x0_{}_{}.npy".format(kmax, nside3D), allow_pickle=True)
        # x00 = np.load("./img1/x00_{}_{}.npy".format(kmax, nside3D), allow_pickle=True)
        # x0 = np.load("./img1/results_{}_{}.npy".format(kmax, nside3D), allow_pickle=True)
        # x0 = [5.45066325E+00, 1.36266581E+00, -2.65625000E-04]
        (lk, ll, lm) = x0
        myHyper.change_SMICA_resolution(nside3D, myHyper.sigma_smica, bandwidth=bandwidth)
        olderr = 1110.0

        for kk in [x for x in karray if x > 10]:
            print(kk)
            kkarray = [x for x in karray if x <= kk + 1]
            kmax = np.max(kkarray)
            results, newmap, err, packedresults = myHyper.project4D3d(kkarray)
            print(lk, ll, lm, err, "intermediate result")
            if err < olderr:
                olderr = err
                filename = "./img1/Best_{}_{}_{}__{}_{}_{}_{}.png".format(kmax, nside3D, bandwidth, np.round(lk, 1),
                                                                          np.round(ll, 1), np.round(lm, 1),
                                                                          np.round(err * 1, 4))
                title = "Best_{}_{}_{}__{}_{}_{}_{}".format(kmax, nside3D, bandwidth, np.round(lk, 1),
                                                            np.round(ll, 1), np.round(lm, 1),
                                                            np.round(err * 1, 4))
                myHyper.plotNewMap(newmap, err, filename=filename, title=title, plotme=True, save=True)
                print(lk, ll, lm, err, title)
                filenameResults = "./PG_data/Results_{}_{}_{}__{}_{}_{}_{}.npy".format(kmax, nside3D, bandwidth,
                                                                                       np.round(lk, 1),
                                                                                       np.round(ll, 1),
                                                                                       np.round(lm, 1),
                                                                                       np.round(err * 1, 4))
                np.save(filenameResults, packedresults)
                np.save("./PG_data/resultsBest_{}_{}_{}.npy".format(kmax, nside3D, bandwidth), results,
                        allow_pickle=True)
                myHyper.plotHistogram(newmap.squeeze(), nside3D, kmax, plotme=True)
                myHyper.plotNewMap(newmap, err, filename=filename, title=title, plotme=True, save=False)
        # dl_SMICA, dl_SMICA_HU, ell = myHyper.plot_ONLY_CL_From_Image(newmap.squeeze(), bandwidth, xmax=10 * kmax)
    if Color.CREATE_GAUSSIAN_BACKGROUND in todo:
        # Create karray
        karray = list(np.arange(2, 50))
        kmax = 1000
        karray += sorted(list(set([int(k) for k in np.linspace(11, kmax, 30)])))
        print(len(karray))
        kmax = max(karray)
        nside3D = 128  # 128
        bandwidth = 3000
        filename = "./img1/3DGaussianNoise_{}_{}_{}.png".format(kmax, nside3D, bandwidth)
        title = "3DGaussianNoise_{}_{}_{}".format(kmax, nside3D, bandwidth)
        (lambda_k, lambda_l, lambda_m) = x0
        #################################
        myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray,
                        lambda_k, lambda_l, lambda_m, loadpriorG=True, savePG=True)
        #################################################################
        delta = 0.001
        lmax = 1000
        llist = sorted(list(set([int(k) for k in np.geomspace(1, lmax + 1, 100)])))
        wavefunction = myHyper.createGaussianBackground(x0, nside3D, delta, karray=llist)
        wavefunction = myHyper.normalizeFColors(wavefunction, myHyper.sigma_smica).squeeze()
        myHyper.plotNewMap(wavefunction, 0, filename=filename, title=title, plotme=True, save=True)
        myHyper.plot_CL_From_Image(wavefunction, nside3D, planck_theory_cl, xmax=2 * lmax, ymax=0.01)
        myHyper.plot_ONLY_CL_From_Image(wavefunction, nside3D, myHyper.SMICA, nsidesmica=1024, xmax=2 * lmax)
        np.save("./PG_data/3DGaussianNoise_{}_{}.npy".format(kmax, nside3D), wavefunction, allow_pickle=True)
    ###############################################################
    if Color.WORK_86_128 in todo:
        (lk, ll, lm) = x0
        karray = list(np.arange(2, 64))
        kmax = np.max(karray)
        karray = np.array(sorted(list(set(karray)))).astype(int)
        nside3D = 128
        bandwidth = 256
        err = 0
        newmap0 = np.load("./PG_data/86_128_Stuff/newmap86_128.npy")
        mu, sigma = norm.fit(newmap0)
        newmap0 = newmap0 / sigma * sigma_smica
        # df = np.load("./PG_data/86_128_Stuff/df32.npy")
        SMICA_LR = np.load("./PG_data/86_128_Stuff/smica128.npy")
        results = np.load("./PG_data/86_128_Stuff/results86_128.npy")
        if myHyper is None:
            myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray,
                            lk, ll, lm, loadpriorG=True, savePG=True, bandwidth=bandwidth, longG=False)
        #################################################################
        filename = "./img1/SingleBest_{}_{}_{}__{}_{}_{}.png".format(kmax, nside3D, bandwidth, chg2ang(lk),
                                                                     chg2ang(ll), chg2ang(lm))
        title = "Best_{}_{}_{}__{}_{}_{}".format(kmax, nside3D, bandwidth, chg2ang(lk),
                                                 chg2ang(ll), chg2ang(lm))

        myHyper.plotNewMap(newmap0, err, filename=filename, title=title, plotme=True, save=True)
        myHyper.plotHistogram(newmap0.squeeze(), nside3D, kmax, plotme=True)
        myHyper.plotNewMap(newmap0.squeeze() - myHyper.SMICA_LR.squeeze(), err, filename=filename, title="Diff Map",
                           plotme=True, save=False)
        print(lk, ll, lm, err, title)
        #################################################################
        #################################################################
        x01 = np.array([0.00140589, 0.43367364])
        x00 = minimize(newerr, x01, args=(myHyper.SMICA_LR, newmap0),
                       method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
        err = x00.fun
        xx0 = x00.x
        newmap, diffmap, restoredmap = myHyper.optimizeNewMap(newmap0, myHyper.SMICA_LR, xx0=xx0,
                                                              nside3D=nside3D, bandwidth=bandwidth, nosigma=True)
        #################################################################
        #################################################################
        #################################################################
        np.save("./img1/newmap.npy", newmap)
        np.save("./img1/diffmap.npy", diffmap)
        np.save("./img1/restoredmap.npy", restoredmap)
        ###############################################################
        ###############################################################
        #     myHyper.plot_Single_CL_From_Image(diffmap, nside3D, xmax=xmax)
        #     myHyper.plot_Single_CL_From_Image(newmap, nside3D, xmax=xmax)
        ###############################################################
        ###############################################################
    if Color.OPTIMIZE_SPECTRUM in todo:
        df = pd.DataFrame(np.concatenate([results, myHyper.df[:, 0:4]], axis=1),
                          columns=["uncorrected_coeff", "k", "l", "m", "std"])
        filename = "./PG_data/spectrumOptimum_{}_{}_{}__{}_{}_{}".format(kmax, nside3D, bandwidth, chg2ang(lk),
                                                                         chg2ang(ll),
                                                                         chg2ang(lm))
        df.to_pickle(filename)
        #####################################################################
        #####################################################################
        #####################################################################
        #####################################################################
        df["coeff"] = df.uncorrected_coeff / df["std"]
        df["abscoeff"] = df.coeff.abs()
        fcolors = df.coeff.values**2
        (mu, sigma) = norm.fit(fcolors)
        n, bins, patch = plt.hist(fcolors, 600, density=1, facecolor="r", alpha=0.25)
        y = norm.pdf(bins, mu, sigma)
        plt.figure()
        plt.scatter(bins[1:], np.log(n/n[1]))
        # plt.xlim(mu - 4 * sigma, mu + 4 * sigma)
        plt.xlim(0.0, mu + 4 * sigma)
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.ylim(0, None)
        plt.title("Amplitude Histogram \n HU Modeling of Planck SMICA Map")
        plt.savefig("./PG_data/AmplitudeHistogram_{}_{}.png", dpi=300)
        plt.show()

        numberofmodes = df.groupby(['k'])["abscoeff"].count()
        df["meanEnergy"] = df.abscoeff ** 2 * df.k * df.std**2
        # df["meanEnergy"] = df.abscoeff ** 2
        # df["meanEnergy"] = df.abscoeff / df.k
        meanEnergyPerK = df.groupby(['k'])["meanEnergy"].sum()
        plt.plot(numberofmodes, meanEnergyPerK)
        plt.title("Mean Energy per k")
        plt.xlabel("Accessible Modes for given k")
        plt.ylabel("Mean Energy per k \n arbitrary units")
        plt.xlim(0, 250)
        plt.ylim(0, None)
        plt.savefig("./PG_data/MeanEnergy.png", dpi=300)
        plt.show()
