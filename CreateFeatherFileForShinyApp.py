import healpy as hp
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import norm

nside = 128
xx, yy, zz = hp.pix2vec(nside=nside, ipix=np.arange(hp.nside2npix(nside)))
ones = np.ones(xx.shape[0])
theta, phi = hp.pix2ang(nside=nside, ipix=np.arange(hp.nside2npix(nside)))
df = pd.DataFrame(np.empty([0, 7]), columns=["ind", "x", "y", "z", "theta", "phi", "density"])
df1 = pd.DataFrame(np.empty([0, 7]), columns=["ind", "x", "y", "z", "theta", "phi", "density"])

for r in np.linspace(0.01, 1499/1500, 10):
    df.ind = r * ones
    df.x = xx * r
    df.y = yy * r
    df.z = zz * r
    df.theta = theta
    df.phi = phi
    df.density = np.random.rand(xx.shape[0])
    title = "Radius={}".format(r)

    mu, sigma = norm.fit(df.density)
    df.density = (df.density - mu) / sigma
    hp.mollview(df.density.squeeze(), title=title, min=-4 * sigma,
                max=4 * sigma, unit="K", cmap=cm.RdBu_r)
    df1 = pd.concat([df1, df])

df1 = df1.reset_index()
df1["index"] = df1.index
df1.to_csv("/home/mp74207/GitHub/R-Projects/UniverseMap/df1.csv")