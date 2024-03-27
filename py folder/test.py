import matplotlib
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants, units
import parameters
import healpy as hp
from os import path
from parameters import *
from matplotlib import cm


DefaultSize=[10,6]
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }
axis_font = {'name':'Arial', 'size':'18'}
title_font = {'name':'Arial', 'size':'18'}





def createlines(xran, yran, initialpoint, tang, nlines, ax=None):
    xrange = xran[1] - xran[0] / nlines
    yrange = yran[1] - yran[0] / nlines
    lastx = initialpoint[0]
    lasty = initialpoint[1]
    plt.show()
    fig, ax=plt.subplots()
    for i in range(nlines):
        xpoints = np.linspace(xran[1], xran[1], 20)
        ypoints = lasty + tang * xpoints
        ax.plot(xpoints, ypoints)


def plotCrystalDiffraction(fig, ax, n=10, rays=False, layers=1):
    # Create the circles
    tang = {}
    tang[-1] = tang[0] = 0.03
    for j in range(layers):
        for i in range(20):
            rad = i / 20
            ax.add_artist(plt.Circle(xy=(rad, j * 0.05), radius=0.01, color='r', fill=False))
        tang[j] = tang[0] * tang[j - 1] / (1 + tang[0] * tang[j - 1])

    # create the incoming waves
    if rays:
        for i in range(n):
            r = -0.05 - i / 20
            plt.plot([0.3, 0.6], [r, r])

    return ax


def plot_CL_From_Image( cl_SMICA, planck_theory_cl, ymax, xmax=3000):
    ell = np.arange(len(cl_SMICA))
    themax = np.max(planck_theory_cl[10:3000, 1])
    planck_theory_cl[:, 1] = planck_theory_cl[:, 1] / themax
    pl = hp.sphtfunc.pixwin(nside=1024)
    # Deconvolve the beam and the pixel window function
    dl_SMICA = cl_SMICA / (B_l(10.0, ell) ** 2 * pl[0:2049] ** 2)
    dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi))
    dl_SMICA = dl_SMICA/np.max(dl_SMICA)

    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_subplot(111)
    ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
    ax.plot(planck_theory_cl[:, 0], planck_theory_cl[:, 1], ell * np.pi, dl_SMICA)
    ax.set_ylabel('$\ell$')
    ax.set_title("Angular Power Spectra")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.set_xlim(10, xmax)
    plt.ylim(1E-10, ymax)
    ax.grid()
    plt.show()
    return cl_SMICA, dl_SMICA, ell

def plot_ONLY_CL_From_Image( cl_SMICA, ymax, xmax=3000):
    ell = np.arange(len(cl_SMICA))
    pl = hp.sphtfunc.pixwin(nside=1024)
    # Deconvolve the beam and the pixel window function
    dl_SMICA = cl_SMICA / (B_l(10.0, ell) ** 2 * pl[0:2049] ** 2)
    dl_SMICA = (ell * (ell + 1) * dl_SMICA / (2 * math.pi))
    dl_SMICA = dl_SMICA/np.max(dl_SMICA)

    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_subplot(111)
    ax.set_xlabel("$ell(ell+1)C_ell/2\pi \, \,(\mu K^2)$")
    ax.plot(ell * np.pi, dl_SMICA)
    ax.set_ylabel('$\ell$')
    ax.set_title("Angular Power Spectra")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.set_xlim(10, xmax)
    plt.ylim(1E-10, ymax)
    ax.grid()
    plt.show()
    return cl_SMICA, dl_SMICA, ell

def plot_aitoff(fcolors, kmax=0, k=0, l=0, m=0,title=None):
    hp.mollview(fcolors, min=-0.0003, max=0.0003, title=title, fig=1, unit="K",
                cmap=cm.RdBu_r)
    hp.graticule()
    plt.savefig(imgAddress + title + "_aitoff_{}_{}_{}_{}.png".format(kmax, k, l, m), dpi=300)
    plt.show()

def read_cachedDF(n):
    filename = "/media/home/mp74207/GitHub/CMB_HU/sphericalharmonics/Big_KLM/%06.0f.pkl" % n
    if not path.exists(filename):
        print(n, " does not exist")
        return
    print("read {}".format(n))
    return pd.read_pickle(filename)

def read_all():
    df = read_cachedDF(0)
    for n in np.arange(1,132):
        df = pd.concat([df,read_cachedDF(n)] ,axis=1)
    return df

if __name__=="__main__":
    DefaultSize=[8,8]

    font = {'family' : 'normal',
            'weight' : 'regular',
            'size'   : 18}

    matplotlib.rc('font', **font)
    title_font = {'name':'Arial', 'size':'16'}

    axis_font = {'name':'Arial', 'size':'14'}
    label_font = {'name':'Arial', 'size':'14'}
    fig, ax = plt.subplots()
    fig.set_size_inches( DefaultSize )# resetthe size
    Size = fig.get_size_inches()
    # print ("Size in Inches", Size)
    # # print(radius)
    # plotCrystalDiffraction(fig, ax, n=10, rays=True)
    # ax.set_ylim(-0.5,1.0)
    # ax.set_xlim(0,1.0)
    #
    # yran = xran = [0.5, 0.0]
    # initialpoint = [0.5, 0]
    # tang = 0.05
    # nlines = 10
    #
    # createlines(xran, yran, initialpoint, tang, nlines, ax)
    (mu_smica, sigma_smica) = norm.fit(planck_IQU_SMICA)

    NSIDE = 10
    fig = plt.figure(1, figsize=[12, 12])
    x0 = np.load("./x0.txt.npy")
    bigdf = read_all()
    cl_SMICA = bigdf.dot(x0)
    plot_ONLY_CL_From_Image(cl_SMICA, ymax=0.5, xmax=2000)
    # plot_CL_From_Image(cl_SMICA, planck_theory_cl, xmax=300)
    # This_SMICA = hp.anafast(planck_IQU_SMICA, lmax=2048)
    # This_SMICA = hp.map2alm(planck_IQU_SMICA, lmax=40)
    # fcolors = hp.alm2map(This_SMICA,nside=1024)
    # (mu, sigma) = norm.fit(fcolors)
    # fcolors = sigma_smica / sigma * (fcolors - mu)
    # hp.mollview(fcolors, min=-3*sigma_smica, max=3*sigma_smica, title="Planck Temperature Map", fig=1, unit="K", cmap=cm.RdBu_r)
    # hp.graticule()
    # plt.show()