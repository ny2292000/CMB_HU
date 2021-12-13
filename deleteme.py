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


myU.createReport(filename="./AllUniverse.xls")

print(myU.y_Seq)

print(myU.x_Seq)