from lib1 import *
from parameters import *
from scipy.integrate import quad

myU = Universe(eta, alpha, alpha_L, eta_L, T0, gamma, n0, vs_pd)
myU.find_k0([2.26414986e+01, 1.32226173e-02, 1.55251641e-03])
myU.getEnergyPressure()
myU.createReport(2)
a = 1


def PhaseVS(x1,x2,xSound):
    def f(s,xSound):
        t0=dbh_t.value
        beta = xSound[0]
        n0 = xSound[1]
        vs0 = 1/np.sqrt(3)
        A = 1/ (1 + np.exp(beta * (s - n0)))
        B = 1/ (s + t0)
        return vs0*A*B
    return quad(f,x1,x2,args=(xSound))[0]

def PhaseN(phase0, n, b=0.01):
    a={}
    for i in range(1,n):
        a[i]=(np.cos(i*phase0)*np.exp(b*i))**2
    df=pd.DataFrame.from_dict(a, orient="index",columns=["Amplitude(n)"])
    df["n"]=df.index
    return df

def PhaseX(phase0, x, n, b=0.01):
    a=0
    for i in range(1,n):
        a += np.cos(i*2.0*np.pi*x)*np.exp(-b*i)
    return a

x0=myU.x_Seq.loc["Neutronium","Time (s)"]
x1=myU.x_Seq.loc["PreFreezing","Time (s)"]
x2=myU.x_Seq.loc["Freezing","Time (s)"]
xSound=myU.xSound
phase0=PhaseVS(x0,x2,myU.xSound)
print(phase0, 2*np.pi/phase0)