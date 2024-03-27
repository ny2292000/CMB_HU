import matplotlib.pylab as plt
import scipy.special as sp
from matplotlib import cm
import healpy as hp
import pandas as pd
import numpy as np
import datetime


nside=1024
mm = hp.nside2npix(nside=nside)
theta, phi = hp.pix2ang(nside=nside, ipix=np.arange(mm))
df = pd.DataFrame()
df["theta"]=theta
df["phi"] = phi
n=5
m=5


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


wrapped = wrapper(sp.sph_harm, m,n,theta,phi)

t1=datetime.datetime.now()
for i in np.arange(10):
    wrapped()
t2=datetime.datetime.now()
print((t2-t1).seconds)


df["fcolors"]=np.real(sp.sph_harm(m,n, theta, phi))
hp.mollview(df.fcolors, min=-0.0007, max=0.0007, title="Planck Temperature Map", fig=1, unit="K",cmap=cm.RdBu_r)
hp.graticule()
plt.show()



