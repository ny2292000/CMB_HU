from lib1 import *
from parameters import *
# # Calculate gammaT for the region Transparency to Today
# gammaT = findGammaT(ionizationfraction=0.5)
# # Calculate Proton Fraction
# protonfraction=findprotonfraction(eta, alpha, alpha_L, eta_L, T0, gamma, n0, mu)
# # Calculate Universe for gammaT0 and t_unfreezing=xout
# gammaT0=1.0
# df0, df1, df2, df3, df4, df5, importantTimes, xout= findGammaT0(gammaT0=gammaT0, protonfraction=protonfraction,gammaT=gammaT)
# print(xout)

def dKEx(y, frac, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    x = frac
    A0= (mn - mp)/T0
    A = (x ** (2/3) - (1 - x) ** (2/3)) * (2*y) ** (2/3)
    B = -(2 * alpha - 4 * alpha_L) * (1 - 2 * x) * y
    C = (2 * eta - 4 * eta_L) * (1 - 2 * x)  * y ** gamma
    D= ( A0 + A + B + C) * T0
    return D

def findy(y,frac,  eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    mu=(constants.hbar*constants.c*(3*np.pi**2*frac*y*n0)**(1/3)).to("MeV")
    return (dKEx(y, frac, eta, alpha, alpha_L, eta_L, T0, gamma, n0) + mu - (mn - mp)).to("MeV").value

frac= 1
x0=10
root = scipy.optimize.root(findy, x0, args=(frac, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
print(root)