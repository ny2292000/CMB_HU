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


def whatIsTime(y):
    dilution = float( y / dbh_y )
    radius = dbh_radius.to("lyr") / dilution ** (1 / 3)
    t = (radius - dbh_radius) / cc.c
    return t.si.value

def whatIsRadius(y):
    dilution = float( y / dbh_y )
    radius = dbh_radius.to("lyr") / dilution ** (1 / 3)
    return radius.to("lyr").value

def whatIsY(t):
    radius = (t * cc.c + dbh_radius).to("lyr")
    dilution = float(  (dbh_radius / radius) ** 3  )
    y = dbh_y * dilution
    return  y

today=4.428e+17
today_y= whatIsY(today*uu.s)
today_y=today_y

defaultsize=[6,4]
colors = get_standard_colors(num_colors=10)

print(alpha, eta, gamma,eta_L,alpha_L, T0.value, n0.value, MN.value, MP.value, ME.value)
#%%
# Characterize Transparency.
ionizationfraction=0.5
gammaT, z_transparency, TransparencyRadius, TransparencyTime, densityAtTransparency, \
    T_at_Transparency = findGammaT(ionizationfraction)
gammaT = gammaT[0]
TransparencyRadius = TransparencyRadius.value
TransparencyTime=TransparencyTime.value
densityAtTransparency=densityAtTransparency.value
T_at_Transparency=T_at_Transparency.value




gammaT, z_transparency, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency
#%% md
# Locating PreFreezing and PostFreezomg times, densities


#%%
myU=Universe(eta, alpha, alpha_L, eta_L, T0, gamma, n0,vssquaredpd)
# vs calculation

#%%
# proton fraction calculation
n=1000
protonfraction = np.linspace(1,0,n)
xout = findprotonfraction(protonfraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)
# xout.plot(x="y", y="ProtonFraction", logx=True)

yy=np.linspace(dbh_y,xout.y.max(),300)
xin= pd.DataFrame(index=np.arange(len(yy)), columns=["y","ProtonFraction",])
xin.y=yy
xin.ProtonFraction=0.0
xin=pd.concat([xout,xin])
xin = xin.sort_values(by="y")
# xin.shape
# xin.plot(x="y", y="ProtonFraction", logx=True)

densityBlackholium = dbh_y
densityNeutronium = dneutron_y
densityPreBigBang = xout.iloc[-1].y
densityPostBigBang = xout.iloc[0].y


# yy =np.concatenate([np.linspace(8,xout.y.max(),300),xout.y.values[1:]])
# xout = findprotonfraction_y(protonfraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)
# xout.plot(x="y", y="ProtonFraction", logx=True)
#%%
# Velocity of Sound
x=xin.y.values
y = myU.vs(x ,xin.ProtonFraction.values)

ind = y>=1/3
y[ind]=1/3
densityAtPreFreezing = x[ind][0]

ind = y<=0.01
densityAtFreezing = x[ind][-1]

# VS and ProtonFraction

df = pd.DataFrame(columns=["t","y","r", "Vs","ProtonFraction","Energy","Temperature","Pressure"])
df.ProtonFraction =xin.ProtonFraction
df.y = xin.y
dff = pd.DataFrame(columns=["t","y","r", "Vs","ProtonFraction","Energy","Temperature","Pressure"])
dff.y= np.concatenate( [np.geomspace(densityPostBigBang, densityAtTransparency, 300),
                        np.geomspace(densityAtTransparency, today_y, 300),
                        [densityBlackholium,densityNeutronium,densityAtPreFreezing,densityAtFreezing,
                         densityPreBigBang,densityPostBigBang,densityAtTransparency,today_y ]])
dff =dff.drop_duplicates(subset=["y"])

dff.ProtonFraction = 1.0
dff.Vs =0.0

df = pd.concat([df,dff])
df = df.drop_duplicates(subset=["y"])
df.Vs=myU.vs(df.y,df.ProtonFraction)
ind = df.Vs>=1/3
df.Vs[ind]=1/3
df.Energy = KE(df.y, df.ProtonFraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)*df.y
df.Pressure= Pressure(df.y, df.ProtonFraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)
df["Density"] = [ (y*n0*cc.m_n).to("kg/m**3").value for y in df.y]
df["t"] = [ whatIsTime(y) for y in df.y]
df["r"] = [ whatIsRadius(y) for y in df.y]
df = df.sort_values(by="y")
df.index = np.arange(len(df))
df =df.drop_duplicates(subset=["y"])
myU.df=df.copy()
myU.getEnergyPressure()
myU.df["TemperatureDensity"]= myU.df.Temperature*myU.df.Density
logP= np.log(myU.df.Pressure)
logy = np.log(myU.df.y)
dlogP = logP[1:].values-logP[0:-1].values
dlogy = logy[1:].values - logy[0:-1].values
myU.df["gammaFromPressureY"]=None
myU.df.gammaFromPressureY.iloc[0:-1]=dlogP/dlogy
myU.df.gammaFromPressureY.ffill()
myU.df = myU.df.sort_values(by="y", ascending = False)
myU.df = myU.df.reindex(np.arange(len(myU.df)))

################################
myU.y_Seq = pd.DataFrame.from_dict({"densityBlackholium": densityBlackholium,
                                     "densityNeutronium": densityNeutronium,
                                     "densityAtPreFreezing": densityAtPreFreezing,
                                     "densityAtFreezing": densityAtFreezing,
                                     "densityPreBigBang": densityPreBigBang,
                                     "densityPostBigBang": densityPostBigBang,
                                     "densityAtTransparency": densityAtTransparency,
                                     "densityToday": today_y}, orient="index", columns=["y"], dtype=float)




myU.y_Seq["Energy"]=np.nan
myU.y_Seq["Pressure"]=np.nan
myU.y_Seq["t"]=np.nan
myU.y_Seq["radius"]=np.nan
myU.y_Seq["Density"]=np.nan

for name,yk in zip(myU.y_Seq.index,myU.y_Seq.y) :
    myU.y_Seq.loc[name]["Energy"]=myU.df[myU.df.y==yk].Energy
    myU.y_Seq.loc[name]["Pressure"]=myU.df[myU.df.y==yk].Pressure
    myU.y_Seq.loc[name]["t"]=myU.df[myU.df.y==yk].t
    myU.y_Seq.loc[name]["radius"]=myU.df[myU.df.y==yk].r
    myU.y_Seq.loc[name]["Density"]=myU.df[myU.df.y==yk].Density



# myU.createReport(cosmologicalangle=2, filename="./ObservableUniverse.xls")
# myU.createReport(cosmologicalangle=2*np.pi, filename="./HypersphericalUniverse.xls")

myU.y_Seq

#%%
myU.getTemperature()
#%%
myU.df.Temperature
#%%
myU.df.plot(x="t", y="Temperature", logx=True)
#%%
myU.df.iloc[0]
#%%
cc.k_B
#%%
myU.df.loc[0:-1:, "gammaFromPressureY"]= dlogP/dlogy
myU.df.gammaFromPressureY=myU.df.gammaFromPressureY.ffill()
#%%
myU.df.loc[0:-1:, "gammaFromPressureY"]= (dlogP/dlogy)[0:]
myU.df.gammaFromPressureY.plot()
#%%
myU.df.loc[0:1876,"gammaFromPressureY"].shape, len(myU.df), len(dlogP/dlogy)
#%%
(dlogP/dlogy)[0:]
#%%
plt.plot((dlogP/dlogy))
#%%
len(dlogP/dlogy)
#%%
len(myU.df)
#%%

#%% md
# PLOTS
#%%
# Plot A

tmin=1E0
tmax=today
dmin = 1E-27
dmax = 1E20

colors = get_standard_colors(num_colors=10)

plt.rcParams['figure.figsize'] = defaultsize
