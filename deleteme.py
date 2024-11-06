import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import integrate
from scipy.optimize import minimize
import astropy.constants as const
import astropy.units as uu
from astropy.cosmology import WMAP9 as cosmo
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
plt.rcParams['font.size'] = 20

saveplot=True

###############################################
# this choice is consistent with the current assumed age of the universe 14.04 GY and short-distance SN1a data
H_0 = 69.69411567633684 * uu.km/uu.s/uu.Mpc
H0Value ="H06969"
fitted=1.6554778950297777
###############################################
R_0 = (const.c/H_0).to("parsec")
sn1a = pd.read_csv("./data/SN1a.txt", names=["name","z","m","error_m","probability"], sep="\t")

z=sn1a.z


# HERE WE ARE MAKING USE OF THE INTERSTELLAR EXTINCTION COEFFICIENT IN THE DATA FILE (A)!!!!!
sn1a["distances_obs"]=10**((sn1a.m)/5+1)*uu.parsec
sn1a["distances_obs_normalized"]=sn1a.distances_obs/R_0
sn1a["distances_obs_comoving_normalized"]=sn1a.distances_obs_normalized/(1+z)
sn1a["modulus_distance"]=sn1a.m

sn1a=sn1a.sort_values(by=["z"])

print( "H_0=",H_0, "The Universe 4D radius is {} billion light-years ".format((R_0.to("lyr").value/1E9).round(2)))
# Two Plots Fitted
sn1a["alpha"]=np.pi/4-np.arcsin(1/np.sqrt(2)/(1+sn1a.z))
sn1a["HU_distances"]=1-(np.cos(sn1a.alpha)-np.sin(sn1a.alpha))
sn1a["HU_4Dradius"]= np.cos(sn1a.alpha)-np.sin(sn1a.alpha)
sn1a["GFactor"] = sn1a.HU_4Dradius**fitted #fitting
sn1a["HU_Fitted"]=sn1a.distances_obs_normalized *sn1a.GFactor
sn1a["GFactor"] = sn1a.HU_4Dradius**1.66  #parameterless prediction
sn1a["HU_Predicted"]=sn1a.distances_obs_normalized *sn1a.GFactor

print("HU Observations Normalized Distances=", sn1a.distances_obs_normalized.max(),"\n"
      "HU Normalized Distances=", sn1a.HU_distances.max())

def errorf2(x):
    fitted=x[0]
    H0=x[1]
    GFactor = sn1ashort.HU_4Dradius**fitted #fitting
    HH_0 = H0 * uu.km/(uu.s*uu.Mpc)
    ###############################################
    R_0 = (const.c/HH_0).to("parsec").value
    sn1ashort.HU_Fitted = (sn1ashort.distances_obs/R_0) * GFactor
    err = sn1ashort.HU_distances- sn1ashort.HU_Fitted
    return np.sum(err*err)

def errorf1(x,H0):
    fitted=x[0]
    GFactor = sn1ashort.HU_4Dradius**fitted #fitting
    HH_0 = H0 * uu.km/(uu.s*uu.Mpc)
    ###############################################
    R_0 = (const.c/HH_0).to("parsec").value
    sn1ashort.HU_Fitted = (sn1ashort.distances_obs/R_0) * GFactor
    print("value ",sn1ashort.HU_Fitted)
    err = (sn1ashort.HU_distances - sn1ashort.HU_Fitted)
    return np.sum(err*err)

def fittedResult(fitted,H0, sn1ashort):
    GFactor = sn1ashort.HU_4Dradius**fitted #fitting
    HH_0 = H0 * uu.km/(uu.s*uu.Mpc)
    ###############################################
    R_0 = (const.c/HH_0).to("parsec").value
    sn1ashort.HU_Fitted = (sn1ashort.distances_obs/R_0) * GFactor
    return sn1ashort

x0=(1.6554778884148815, 69.69411534991869)
sn1ashort = sn1a[sn1a.z< 0.14].sort_values(by="z")
res = minimize(errorf2, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})
H0=res.x[1]
fitted=res.x[0]


x0=[1.66]
sn1ashort = sn1a[sn1a.z> 1.0].sort_values(by="z")

res = minimize(errorf1, x0, method='nelder-mead',args=[H0],
               options={'xatol': 1e-8, 'disp': True})
fitted=res.x[0]
sn1ashort = fittedResult(fitted, H0, sn1ashort)
R0 = (const.c/(H0*(uu.km/(uu.s*uu.Mpc)))).to("Glyr")
fig = plt.figure()
ax = plt.gca()
# ax.scatter(sn1ashort.z, sn1ashort.HU_Fitted, 'o', c='blue', alpha=0.05, markeredgecolor='none')
ax.scatter(sn1ashort.z, sn1ashort.HU_Fitted)
ax.plot(sn1ashort.z, sn1ashort.HU_distances,'b')
ax.set_ylim([1E-2,1])
ax.set_yscale('log')
ax.set_ylabel("Normalized Distance")
ax.set_xlabel("Redshift z")
ax.set_title("Normalized Distance vs Z")
print(res,fitted, H0, R0)
plt.show()

