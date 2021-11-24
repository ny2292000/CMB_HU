import math

import pandas as pd
import scipy
import scipy.integrate as integrate
import scipy.interpolate as sp
from scipy.optimize import minimize
from scipy.special import gamma as gammaF
from astropy import units as uu
from astropy import constants

from parameters import *


def Pressure(xx, frac, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    y = xx
    x = frac
    A = 2 / 5 * (x ** (5 / 3) + (1 - x) ** (5 / 3)) * (2 * y) ** (5 / 3)
    B = -((2 * alpha - 4 * alpha_L) * x * (1 - x) + alpha_L) * y ** 2
    C = gamma * ((2 * eta - 4 * eta_L) * x * (1 - x) + eta_L) * y ** (gamma + 1)
    return ((A + B + C) * n0 * T0).si


def KE(xx, frac, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    y = xx
    x = frac
    MN = (constants.m_n*constants.c**2).to("MeV")
    MP = (constants.m_2*constants.c**2).to("MeV")
    A0 = (1-frac)*MN + frac*MP
    A = 3 / 5 * 2 ** (2 / 3) * (x ** (5 / 3) + (1 - x) ** (5 / 3)) * (y) ** (2 / 3)
    B = -((2 * alpha - 4 * alpha_L) * x * (1 - x) + alpha_L) * y
    C = gamma * ((2 * eta - 4 * eta_L) * x * (1 - x) + eta_L) * y ** gamma
    return ((A0 + A + B + C) * T0).to("MeV").value


def dKE(xx, frac, eta, alpha, alpha_L, eta_L, T0, gamma):
    y = xx
    x = frac
    A = 3 / 5 * 2 / 3 * 2 ** (2 / 3) * (x ** (5 / 3) + (1 - x) ** (5 / 3)) * (y) ** (-1 / 3)
    B = -((2 * alpha - 4 * alpha_L) * x * (1 - x) + alpha_L)
    C = gamma * ((2 * eta - 4 * eta_L) * x * (1 - x) + eta_L) * y ** (gamma - 1) * gamma
    return (A + B + C) * T0


def ddKE(xx, frac, eta, alpha, alpha_L, eta_L, T0, gamma):
    y = xx
    x = frac
    A = 3 / 5 * (x ** (5 / 3) + (1 - x) ** (5 / 3)) * (2 * y) ** (-4 / 3) * 2 / 3 * (-1 / 3)
    C = gamma * ((2 * eta - 4 * eta_L) * x * (1 - x) + eta_L) * y ** (gamma - 2) * gamma * (gamma - 1)
    return (A + C) * eta * T0


def errorf(xx,  eta, alpha, alpha_L, eta_L, T0, gamma):
    frac=0.5
    n0= 0.16E-15*uu.m
    n=n0*xx
    MN = (constants.m_n*constants.c**2).to("MeV")
    MP = (constants.m_2*constants.c**2).to("MeV")
    T0=((3*np.pi**2*n0/2)**(2/3)*constants.hbar**2/2/constants.m_n).to("MeV")
    B=16*uu.MeV - MN
    K=235*uu.MeV
    S=32*uu.MeV
    L=50*uu.MeV
    AA= n * dKE(xx, frac, eta, alpha, alpha_L, eta_L, T0, gamma) - KE(xx, frac, eta, alpha, alpha_L, eta_L, T0, gamma, n0):






