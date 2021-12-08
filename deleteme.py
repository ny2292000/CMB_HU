import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.fftpack import fft
from parameters import *
from astropy import constants as cc, units as uu
from lib1 import *
from lib2 import *
pd.set_option('display.float_format', lambda x: '%.3e' % x)
from PyAstronomy import *
import itertools
from kneed import KneeLocator
from pandas.plotting._matplotlib.style import get_standard_colors
# Processing Sound Speed on Neutronium.  Data from article was dependent upon energy density MeV/fm3
# as opposed to seconds.
# https://arxiv.org/pdf/1303.4662.pdf

# print("[")
# for xx in vssquared:
#     y0=xx[0]*(uu.MeV/uu.fm**3)/mn/n0
#     t,y,r=whatTimeRadius(y0)
#     print("[",t,",", np.sqrt(xx[1]),"],")
# print("]")

today=4.428e+17
today_y= whatIsY(today*uu.s)
today_y=today_y

defaultsize=[6,4]
colors = get_standard_colors(num_colors=10)

print(alpha, eta, gamma,eta_L,alpha_L, T0.value, n0.value, MN.value, MP.value, ME.value)

myU=Universe(eta, alpha, alpha_L, eta_L, T0, gamma, n0,vssquaredpd)


def getTemperature(obj, gcoef=4 / 5):
    xprior = obj.df.ProtonFraction.iloc[0]
    yprior = obj.df.y.iloc[0]
    Temp_prior = obj.df.Temperature.iloc[0] = 1E-4
    gamma_prior = obj.df.gammaFromPressureY.iloc[0]
    Temp = 0.0
    for i, row in list(obj.df.iterrows())[1:]:
        y = row["y"]
        xnew = row["ProtonFraction"]
        if (xnew > 0.0):
            print(y, xnew)

        if y >= obj.densityAtPreFreezing:
            obj.df.loc[y, "Temperature"] = Temp
            print("START COUNTING")
            continue

        if y <= obj.y_Seq.loc["densityAtTransparency"].y:
            gamma_prior = obj.gammaT
        else:
            gamma_prior = gcoef * gamma_prior
        dx = xnew - xprior
        if xprior < 0.98:
            Temp = Temp_prior * (y / yprior) ** (obj.k0 * gamma_prior - 1) + dx * (
                        y * (MN - MP - ME) * 2 / 3 / cc.k_B).to(uu.K).value
            yprior = y
            Temp_prior = Temp
        else:
            Temp = Temp_prior * (y / yprior) ** (obj.k0 * gamma_prior - 1)
        row["Temperature"] = Temp
        xprior = xnew
        gamma_prior = row["gammaFromPressureY"]
        obj.df.loc[y, "Temperature"] = Temp
        print(row["t"], Temp)
    t_today = obj.df.loc[obj.densityToday, "Temperature"]
    t_transparency = obj.df.loc[obj.densityAtTransparency, "Temperature"]
    return (obj.t_today - t_today) ** 2 * 1E6 + (obj.t_transparency - t_transparency) ** 2


getTemperature(myU, gcoef=4 / 5)