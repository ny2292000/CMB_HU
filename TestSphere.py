from lib1 import *
from lib2 import *
from parameters import *
import matplotlib.pyplot as plt
import itertools
from numba import jit


# myU = Universe(eta, alpha, alpha_L, eta_L, T0, gamma, n0, vs_pd)
# myU.unpickleme()
#
# x0 = myU.x_Seq.loc["Neutronium", "Time (s)"]
# x1 = myU.x_Seq.loc["PreFreezing", "Time (s)"]
# x2 = myU.x_Seq.loc["Freezing", "Time (s)"]
# xSound = myU.xSound
# phase0 = PhaseVS(x0, x2, myU.xSound)
# print(phase0, 2 * np.pi / phase0)
#
# phase0=0.27573703538158384
# # Number of sample points
# N = 1000
# # sample spacing
# T = 2* np.pi/ N
# x = np.linspace(0, N*T, N)
# n=150
# y=PhaseX(phase0,x, n)
# dfX=pd.DataFrame(zip(x,y), columns=["x","DensitySTD"])
# dfX.plot(x="x",y="DensitySTD", logx=False)
# plt.xlabel("x (radians)")
# plt.show()
#
# myU.pickleme()
#
# a1=0
# a2=1000
# a3=2000
#
# dfX["DensityX"] = np.roll(dfX.DensitySTD, a1)
# dfX["DensityY"] = np.roll(dfX.DensitySTD, a2)
# dfX["DensityZ"] = np.roll(dfX.DensitySTD, a3)
#
# dfX.to_pickle(("./dfX"))
dfX=pd.read_pickle("./dfX")
dfY=dfX[dfX.x<=2]
dfY.x=dfY.x-1

def createSphere(dfY):
    dfsphere=pd.DataFrame(columns=["i", "j", "k","x","y","z", "density"])
    for ind,inda in enumerate(itertools.product(dfY.index,dfY.index,dfY.index)):
            if (dfY.iloc[inda[0]].x**2 +dfY.iloc[inda[1]].x**2 +dfY.iloc[inda[2]].x**2 -1)<=0.001:
                dens = dfY.iloc[inda[0]].DensityX +dfY.iloc[inda[1]].DensityY +dfY.iloc[inda[2]].DensityZ
                dfsphere.loc[ind,"i"]=inda[0]
                dfsphere.loc[ind, "j"]=inda[1]
                dfsphere.loc[ind, "k"]=inda[2]
                dfsphere.loc[ind,"x"]=dfY.iloc[inda[0]].x
                dfsphere.loc[ind, "y"]=dfY.iloc[inda[1]].x
                dfsphere.loc[ind, "z"]=dfY.iloc[inda[2]].x
                dfsphere.loc[ind,"density"] = dens
    return dfY, dfsphere

dfY, dfsphere = createSphere(dfY)

dfsphere.to_pickle("./dfsphere")
a1=1