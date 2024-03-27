import cupy as cu
import scipy  as sp
import numpy as np
from datetime import datetime

# Write our GPU kernel inside of our script

if __name__=="__main__":
    # create a matrix of random numbers and set it to be
    # 32-bit floats
    m=10000
    a = cu.random.rand(m,m)
    a = a.astype(float32)
    b = cu.random.rand(m,1)
    b = b.astype(float32)
    t1 = datetime.now()
    for i in np.arange(1,100000):
        b= a.dot(b)
    print((datetime.now()-t1).microseconds)
    aa=1

