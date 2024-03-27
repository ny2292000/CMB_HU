import matplotlib.pylab as plt
import scipy.special as sp
from matplotlib import cm
import healpy as hp
import pandas as pd
import numpy as np
import cupy as cu
from numba import cuda
import datetime

# Disable memory pool for device memory (GPU)
cu.cuda.set_allocator(None)

# Disable memory pool for pinned memory (CPU).
cu.cuda.set_pinned_memory_allocator(None)


pool = cu.cuda.MemoryPool(cu.cuda.malloc_managed)
cu.cuda.set_allocator(pool.malloc)


LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

# @cuda.jit
# def fast_factorial(n):
#     if n > 20:
#         raise ValueError
#     return LOOKUP_TABLE[n]


def spherHarmonic(l, m, theta, phi):
    N = np.sqrt((2 * l + 1 / (4 * np.pi) * LOOKUP_TABLE[l - m] / LOOKUP_TABLE[l + m]))
    costheta = cu.asnumpy(np.cos(theta))
    return N * np.exp(1j * phi) * sp.lpmv(m, l, costheta)

mempool = cu.get_default_memory_pool()
mempool.free_all_blocks()
nside=1024
mm = hp.nside2npix(nside=nside)
theta, phi = hp.pix2ang(nside=nside, ipix=np.arange(mm))

df = pd.DataFrame()
df["theta"]=theta
df["phi"] = phi
n=5
m=5

t1=datetime.datetime.now()
for i in np.arange(10):
    sp.sph_harm( m,n,theta,phi)
t2=datetime.datetime.now()
print((t2-t1).seconds)


df["fcolors"]=np.real(sp.sph_harm(m,n, theta, phi))
hp.mollview(df.fcolors, min=-0.0007, max=0.0007, title="Planck Temperature Map", fig=1, unit="K",cmap=cm.RdBu_r)
hp.graticule()
plt.show()

