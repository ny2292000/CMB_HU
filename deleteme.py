import matplotlib.pyplot as plt
import scipy as sp
from scipy.fftpack import fft
from parameters import *
from astropy import constants as cc, units as uu
from lib1 import *
from lib2 import *
pd.set_option('display.float_format', lambda x: '%.3e' % x)
from PyAstronomy import *
import itertools
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
defaultsize=[6,4]

K=236.0*uu.MeV
B=16.0*uu.MeV
L=50.0*uu.MeV
S=32.0*uu.MeV
MP= (cc.m_p*cc.c**2).to("MeV")
MN= (cc.m_n*cc.c**2).to("MeV")
ME= 0.782*uu.MeV
pi= np.pi
n0=0.16/uu.fm**3
# n0=0.054/uu.fm**3
hydrogenatom=cc.m_p+cc.m_e
T0=((3*np.pi**2*n0/2)**(2/3)*cc.hbar**2/(2*cc.m_n)).to("MeV")
hydrogenatomwavelength = (cc.h/(hydrogenatom*cc.c)).si
hydrogenatomwavelength, T0, K, B, L,S,MP, MN, ME

alpha = -2*(5*B*K - 3*(4*B - K)*T0)/(5*(9*B - K)*T0 + 3*T0**2)
eta = -18/5*(25*B**2 + 10*B*T0 + T0**2)/(5*(9*B - K)*T0 + 3*T0**2)
gamma = 1/9*(5*K + 6*T0)/(5*B + T0)
alpha_L = 1/6*(3*T0*alpha - 3*T0*eta + 6*T0*eta_L - 6*S + 2*T0)/T0
eta_L = 1/18*(9*T0*eta*gamma - 9*T0*alpha + 18*T0*alpha_L + 6*L - 4*T0)/(T0*gamma)
print(alpha, eta, gamma,eta_L,alpha_L, T0, n0)




y0=1
# frac = findprotonfraction(eta, alpha, alpha_L, eta_L, T0, gamma, n0)
df = {}
for x in np.linspace(1,6E-2,1000):
    y = findprotonfraction_y(x, y0, eta, alpha, alpha_L, eta_L, T0, gamma, n0)
    y0=y
    df[y]=x

df = pd.DataFrame.from_dict(df, orient="index")
df["y"] = df.index
df.columns = ["ProtonFraction", "y"]
df.plot(x="y", y="ProtonFraction", logy=True, logx=True)
plt.show()
a=1