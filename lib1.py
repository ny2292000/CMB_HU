import math

import pandas as pd
import scipy
import scipy.integrate as integrate
import scipy.interpolate as sp
from scipy.optimize import minimize
from scipy.special import gamma as gammaF
from parameters import *

def volumeCalc(n, phi, r):
    if phi==2:
        return 4/3*np.pi*r**3
    return np.pi ** ((n - 1) / 2) * r ** 3 / gammaF((n - 1) / 2) * \
           integrate.quad(lambda x: np.sin(x) ** int(n - 2), 0, phi)[0]

def whatIsY(t):
    radius = (t * cc.c + dbh_radius).to("lyr")
    dilution = float(  (dbh_radius / radius) ** 3  )
    y = dbh_y * dilution
    return  y


from parameters import *

H0 = 1
c = 1
R0 = 1
pi4 = math.pi / 4.0
sqrt2 = math.sqrt(2)
RR=14.03E9*uu.lyr.si

today=4.428e+17
today_y= whatIsY(today*uu.s)
today_y=today_y



def findjump(y):
    x0=y[1]
    for x in y[2:]:
        if np.abs(x-x0)>1:
            return [x0,x]
        x0=x
    return []

class Universe():
    def __init__(self, eta, alpha, alpha_L, eta_L, T0, gamma, n0,vssquaredpd):
#          Characterize Transparency.
        ionizationfraction=0.5
        gammaT, z_transparency, TransparencyRadius, TransparencyTime, densityAtTransparency, \
            T_at_Transparency = findGammaT(ionizationfraction)
        gammaT = gammaT[0]
        TransparencyRadius = TransparencyRadius.value
        TransparencyTime=TransparencyTime.value
        densityAtTransparency=densityAtTransparency.value 
        T_at_Transparency=T_at_Transparency.value
#       proton fraction calculation
        n=1000
        protonfraction = np.linspace(1,0,n)
        xout = findprotonfraction(protonfraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)

        yy=np.linspace(dbh_y,xout.y.max(),300)
        xin= pd.DataFrame(index=np.arange(len(yy)), columns=["y","ProtonFraction",])
        xin.y=yy
        xin.ProtonFraction=0.0
        xin=pd.concat([xout,xin])
        xin = xin.sort_values(by="y")

        densityBlackholium = dbh_y
        densityNeutronium = dneutron_y
        densityPreBigBang = xout.iloc[-1].y
        densityPostBigBang = xout.iloc[0].y

        # Velocity of Sound
        x=xin.y.values
        y = self.vs(x ,xin.ProtonFraction.values)

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
        df.Vs=self.vs(df.y,df.ProtonFraction)
        ind = df.Vs>=1/3
        df.Vs.loc[ind]=1/3
        df.Energy = KE(df.y, df.ProtonFraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)*df.y
        df.Pressure= Pressure(df.y, df.ProtonFraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)
        df["Density"] = [ (y*n0*cc.m_n).to("kg/m**3").value for y in df.y]
        df["t"] = [ whatIsTime(y) for y in df.y]
        df["r"] = [ whatIsRadius(y) for y in df.y]
        df = df.sort_values(by="y")
        df.index = np.arange(len(df))
        df =df.drop_duplicates(subset=["y"])
        self.df=df.copy()
        self.getEnergyPressure()
        
        # Derive Gamma from Energy and Pressure curves
        logP= np.log(self.df.Pressure)
        logy = np.log(self.df.y)
        dlogP = logP[1:].values-logP[0:-1].values
        dlogy = logy[1:].values - logy[0:-1].values
        self.df["gammaFromPressureY"]=None
        self.df.gammaFromPressureY.iloc[0:-1]=dlogP/dlogy
        self.df = self.df.sort_values(by="y", ascending = False)
        self.df = self.df.reset_index(drop=True)
        self.df.gammaFromPressureY.ffill(inplace=True)
        self.df.gammaFromPressureY.bfill(inplace=True)
        self.getTemperature()
        self.df["TemperatureDensity"]= self.df.Temperature*self.df.Density
        ################################
        self.y_Seq = pd.DataFrame.from_dict({"densityBlackholium": densityBlackholium,
                                             "densityNeutronium": densityNeutronium,
                                             "densityAtPreFreezing": densityAtPreFreezing,
                                             "densityAtFreezing": densityAtFreezing,
                                             "densityPreBigBang": densityPreBigBang,
                                             "densityPostBigBang": densityPostBigBang,
                                             "densityAtTransparency": densityAtTransparency,
                                             "densityToday": today_y}, orient="index", columns=["y"], dtype=float)




        self.y_Seq["Energy"]=np.nan
        self.y_Seq["Pressure"]=np.nan
        self.y_Seq["t"]=np.nan
        self.y_Seq["radius"]=np.nan
        self.y_Seq["Density"]=np.nan
        self.y_Seq["Temperature"]=np.nan

        for name,yk in zip(self.y_Seq.index,self.y_Seq.y) :
            self.y_Seq.loc[name]["Energy"]=self.df[self.df.y==yk].Energy
            self.y_Seq.loc[name]["Pressure"]=self.df[self.df.y==yk].Pressure
            self.y_Seq.loc[name]["t"]=self.df[self.df.y==yk].t
            self.y_Seq.loc[name]["radius"]=self.df[self.df.y==yk].r
            self.y_Seq.loc[name]["Density"]=self.df[self.df.y==yk].Density
            self.y_Seq.loc[name]["Temperature"]=self.df[self.df.y==yk].Temperature



 
    def vs(self, y,x):
        return 1/3*((15*(2*(eta - 2*eta_L)*(x - 1)*x - eta_L)*T0*(gamma - 1)*gamma*y**(gamma - 2)/n0 + 2*2**(2/3)*(x**(5/3) + (-x + 1)**(5/3))*T0/(n0*y**(4/3)))*n0**2*y**2 + 6*(5*(2*(eta - 2*eta_L)*(x - 1)*x - eta_L)*T0*gamma*y**(gamma - 1)/n0 - 2*2**(2/3)*(x**(5/3) + (-x + 1)**(5/3))*T0/(n0*y**(1/3)) - 5*(2*(alpha - 2*alpha_L)*(x - 1)*x - alpha_L)*T0/n0)*n0**2*y)/((5*(2*(eta - 2*eta_L)*(x - 1)*x - eta_L)*T0*gamma*y**(gamma - 1) - 2*2**(2/3)*(x**(5/3) + (-x + 1)**(5/3))*T0/y**(1/3) - 5*(2*(alpha - 2*alpha_L)*(x - 1)*x - alpha_L)*T0)*n0*y - (3*2**(2/3)*(x**(5/3) + (-x + 1)**(5/3))*T0*y**(2/3) + 5*(2*(alpha - 2*alpha_L)*(x - 1)*x - alpha_L)*T0*y - 5*(2*(eta - 2*eta_L)*(x - 1)*x - eta_L)*T0*y**gamma - 5*MN*(x - 1) + 5*(ME + MP)*x)*n0)


    
    def pickleme(self):
        self.df.to_pickle("./df.pkl")
        self.y_Seq.to_pickle("./y_Seq.pkl")
        self.x_Seq.to_pickle("./x_Seq.pkl")

    def unpickleme(self):
        self.df = pd.read_pickle("./df.pkl")
        self.y_Seq = pd.read_pickle("./y_Seq.pkl")
        self.x_Seq = pd.read_pickle("./x_Seq.pkl")

    def createReport(self, cosmologicalangle=2, filename= "./x_Seq.xls" ):

        volume = volumeCalc(4, cosmologicalangle, self.y_Seq.loc["densityPreBigBang", "radius"] * uu.lyr).to('m**3')
        if cosmologicalangle==2:
            whichUniverse= "Observable "
        else:
            whichUniverse= "Hyperspherical "

        # Unit cell volume with deBroglieLambda side
        cell = (deBroglieLambda) ** 3
        # Number of Neutrons  = 2.5e+79
        CurrentVolume = volumeCalc(4, cosmologicalangle, self.y_Seq.loc["densityToday", "radius"] * uu.lyr).to('m**3')
        MassOfUniverse = (CurrentVolume*rho).si
        NumberOfNeutrons = (MassOfUniverse / cc.m_n).si
        print(CurrentVolume, rho, MassOfUniverse)
        # Energy available
        energyPerNeutron = 0.78254809 * uu.MeV
        Energy = NumberOfNeutrons * energyPerNeutron  # EnergyPerNeutron # 4.5E78MeV = 7.2E65 Joules
        EnergyPerSupernova = 1E51 * uu.erg
        velocityAvg = np.sqrt(2 * Energy.to(uu.joule) / MassOfUniverse)  # 0.04081379 c
        BigBangVolume = volume
        BigBangEnergy = Energy.to('erg')
        NumberOfSupernovae = (BigBangEnergy / EnergyPerSupernova).si

        ls = uu.lyr / 365.25 / 24 / 3600
#         print("Optimized K0 = ", self.k0)
        print(
            "\n",
            "Initial 4D Radius of the Universe = ", (self.y_Seq.loc["densityBlackholium", "radius"] * uu.lyr).to(ls),
            "\n\n"
        )
        print("\n",
              "Initial Volume of the {}Universe".format(whichUniverse), volume, "\n",
              "Number of Neutrons", NumberOfNeutrons, "\n",
              "MassOfUniverse for {} radians =".format(cosmologicalangle), MassOfUniverse, "\n",
              "BigBangEnergy = ", BigBangEnergy, "\n",
              "BigBangEnergyDensity = ", (BigBangEnergy / volume).to("J/m3"), "\n",
              "EnergyPerSupernova = ", EnergyPerSupernova, "\n",
              "Number of Supernovae = ", NumberOfSupernovae, "\n",
              "Cell Length = ", deBroglieLambda,"\n",
              "Current Density = ", rho,"\n",
              )
        self.x_Seq = self.y_Seq.copy()
        self.x_Seq.columns = ["n/n0", "Time (s)", "Radius (lyr)", "Density (Kg/m3)"]
        self.x_Seq["Density (1/fm3)"] = self.x_Seq["n/n0"] * n0
        self.x_Seq["Time (year)"] = self.x_Seq["Time (s)"] / 365.25 / 24 / 3600
        self.x_Seq["Radius (light-seconds)"] = self.x_Seq["Radius (lyr)"] * (365.25 * 24 * 3600)

        for index, row in self.y_Seq.iterrows():
            radius = (row["radius"] * uu.lyr).value
            self.x_Seq.loc[index, "Volume (lyr3)"] = volumeCalc(4, cosmologicalangle, radius)

        pd.set_option('display.float_format', lambda x: '%.3e' % x)
        self.x_Seq.index = [x.replace("density", "").replace("At", "") for x in self.x_Seq.index]
        print(self.x_Seq)

        self.x_Seq.to_excel(filename)

    def getEnergyPressure(self):
        for i, row in self.df.iterrows():
            self.df.loc[self.df.y==row.y, "Energy"] = KE(row["y"], row["ProtonFraction"], eta, alpha, alpha_L, eta_L, T0, gamma, n0)
            self.df.loc[self.df.y==row.y, "Pressure"] = Pressure(row["y"], row["ProtonFraction"], eta, alpha, alpha_L, eta_L, T0, gamma, n0).si
           
    #####################################################
    
    
#     def getTempEverywhere(self, y, x, t, tprior, Tempprior, xprior, massDensity):
#         if self.PreBigBang < y:
#             return 1E-8

#         if (self.PreBigBang - y) * (self.PostBigBang - y) <= 0:
#             dtemp = (2 / 3 * x * (mn - mp - me - m_neutrino) / cc.k_B).si.value
#             return dtemp

#         if self.PostBigBang > y:
#             return Tempprior * (tprior / t) ** (3 * (self.getgamma(massDensity) - 1))
        
    def getTemperature(self):
        self.df = self.df.sort_values(by="y", ascending = False)
        self.df = self.df.reset_index(drop=True)
        xprior = self.df.ProtonFraction.iloc[0]
        yprior = self.df.y.iloc[0]
        Temp_prior = self.df.Temperature.iloc[0]=1E-4 
        gamma_prior = self.df.gammaFromPressureY.iloc[0]
        Temp=0.0
        for i, row in list(self.df.iterrows())[1:]:
            y = row["y"]
            dx = row["ProtonFraction"]-xprior
            if dx <0:
                dx = 0.0
            if xprior<0.98:
                Temp = Temp_prior*(y/yprior)**(4/5*gamma_prior-1) +  dx*( y*(MN-MP-ME)*2/3/cc.k_B ).to(uu.K).value
                yprior = y
                Temp_prior = Temp
            else:
                Temp = Temp_prior*(y/yprior)**(4/5*gamma_prior-1) +  dx*( y*(MN-MP-ME)*2/3/cc.k_B ).to(uu.K).value
            row["Temperature"] = Temp
            xprior = row["ProtonFraction"]
            gamma_prior=row["gammaFromPressureY"]
            self.df.loc[self.df.y == y, "Temperature"]= Temp
        return self.df

#     def find_k0(self, x0):
#         def errf(x):
#             self.k0 = x
#             newTransparencyTemp, newtemp = self.getTemperature()
#             # return  (self.TransparenceTemperature - newTransparencyTemp)**2 + 100*(2.725-newtemp)**2
#             return (self.TransparenceTemperature - newTransparencyTemp) ** 2

#         results = scipy.optimize.minimize(errf, x0, method="Nelder-Mead", options={'xatol': 1e-8, 'disp': True})
#         self.k0 = results.x
#         return self.k0, results, self.df.Temperature.iloc[-1:]

#     def getgamma(self, massDensity):
#         massDensity_limit = self.k0[0]  # 10 kg/m3
#         a = massDensity_limit / massDensity
#         if massDensity < massDensity_limit:
#             return 4 / 3
#         else:
#             return (4 / 3) ** (self.k0[1] * a ** self.k0[2])



#     #####################################################
    









def KE(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    EKy =  3/5*2**(2/3)*(x**(5/3) + (-x + 1)**(5/3))*T0*y**(2/3) + (2*(alpha - 2*alpha_L)*(x - 1)*x - alpha_L)*T0*y - (2*(eta - 2*eta_L)*(x - 1)*x - eta_L)*T0*y**gamma
    return EKy #.to("MeV").value

def Pressure(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    P =  -1/5*(5*(2*(eta - 2*eta_L)*(x - 1)*x - eta_L)*T0*gamma*y**(gamma - 1)/n0 - 2*2**(2/3)*(x**(5/3) + (-x + 1)**(5/3))*T0/(n0*y**(1/3)) - 5*(2*(alpha - 2*alpha_L)*(x - 1)*x - alpha_L)*T0/n0)*n0**2*y**2
    return P #.si

def dKEx(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    # equilibrium equation is d(EK)/dx)= - y*n0*mu
    dEKy_x =  2**(2/3)*T0*y**(2/3)*(x**(2/3) - (-x + 1)**(2/3)) + 2*((alpha - 2*alpha_L)*(x - 1) + (alpha - 2*alpha_L)*x)*T0*y - 2*((eta - 2*eta_L)*(x - 1) + (eta - 2*eta_L)*x)*T0*y**gamma
    return dEKy_x




# def mu(x,y):
#     return (cc.hbar * cc.c * (3 * np.pi ** 2 * (1- x)* y * n0) ** (1 / 3)).to("MeV")

# def findy(y, xx, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
#     # equilibrium equation is d(EK)/dx + mu - (MN-MP)
#     val = (dKEx(y, xx, eta, alpha, alpha_L, eta_L, T0, gamma, n0) + mu (xx,y) - (MN-MP)).to("MeV")
#     return val

def findprotonfraction(xx, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    # This calculated y values for protonfraction inputs - x is the protonfraction array [0,1]
    def mu(x,y):
        return (cc.hbar * cc.c * (3 * np.pi ** 2 * x * y * n0) ** (1 / 3)).to("MeV")
    
    def findy_err(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
        # equilibrium equation is d(EK)/dx + mu - (MN-MP)
        val = (dKEx(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0) + mu (x,y) - (MN-MP)).to("MeV").value
        # print( y, val)
        return val
    df = {}
    y0 = 1.0
    
    for x in xx:
        try:
            # I am solving for y (density) and not x the protonfraction
            # root = scipy.optimize.root(findy, y0, args=(x, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
            root = scipy.optimize.brentq(findy_err, 0, 1, args=(x, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
            df[root] = x
        except Exception:
            pass
#             print("failed  ", x)
    df = pd.DataFrame.from_dict(df, orient="index")
    df.columns=['ProtonFraction']
    df["y"]=df.index
    return df
# protonfraction = np.logspace(0,-3,300) #np.logspace(0,-7,1000) # np.geomspace(1, 1E-5, 1000)
# xout = findprotonfraction(protonfraction, eta, alpha, alpha_L, eta_L, T0, gamma, n0)
# xout.plot(x="y", y="ProtonFraction", logx=True)




def findprotonfraction_y(yy, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    # This calculated protonfraction values for densities y inputs - yy is the density array [0,8]
    def mu(x,y):
        return (cc.hbar * cc.c * (3 * np.pi ** 2 * (1- x)* y * n0) ** (1 / 3)).to("MeV")
    
    def findy_err(x,y, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
        # equilibrium equation is d(EK)/dx + mu - (MN-MP)
        val = (dKEx(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0) + mu (x,y) - (MN-MP)).to("MeV").value
        # print( x, val)
        return val

    df = {}
    for y in yy:
        try:
            # I am solving for y (density) and not x the protonfraction
            root = scipy.optimize.brentq(findy_err, 0, 1, args=(y, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
            df[y]=root
            x0 = root
        except Exception:
            pass
    df = pd.DataFrame.from_dict(df, orient="index")
    df.columns=['ProtonFraction']
    df["y"]=df.index
    return df
# xout = findprotonfraction_y(np.geomspace(1E-9,1e-2,1000), eta, alpha, alpha_L, eta_L, T0, gamma, n0)
# xout.plot(x="y", y="ProtonFraction", logx=True)
    


# Used for getting redshift associated with certain ionization fraction on the plasma (transparency epoch)
def findGammaT(ionizationfraction):
    #here x is the adiabatic cooling gamma and ionizationfraction is the ionizationfraction
    T_today = 2.72548  # today's temperature
    z, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency = \
                                    findTransparencyEpoch(ionizationfraction)
    args = (ionizationfraction,z, T_today, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency),
    
    root = scipy.optimize.root(errfrac, x0=0.1, args=args)
    gammaT = root.x
    return gammaT, z, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency


def errfrac(gammaIn, args):
    ionizationfraction,z, T_today, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency=args
    # https://www.schoolphysics.co.uk/age16-19/Thermal%20physics/Gas%20laws/text/Adiabatic_change_proof/index.html
    # TV**(gamma-1)=C_1
    # T_today*(RR**3)**(gamma-1)= T_at_transparency*(TransparencyRadius**3)**(gamma-1)
    # T_today = T_at_transparency*((TransparencyRadius/RR)**3)**(gamma-1)
    # since gamma = x
    # T_today = T_at_transparency*(TransparencyRadius/RR)**(3*gamma-3)
    errorgamma = T_today - ((TransparencyRadius / RR).si) ** (3 * gammaIn - 3) * T_at_Transparency.value
    return errorgamma


#############################################################
#############################################################  
def findTransparencyEpoch(ionizationfraction):
    x0 = 4000 # temperature
    root = scipy.optimize.root(fracIonization, x0=x0, args=(ionizationfraction))
    z = root.x[0]
    TransparencyRadius = (RR / (1 + z)).to("lyr")
    TransparencyTime = ((TransparencyRadius - dbh_radius) / cc.c).si
    densityAtTransparency = (densityUz(z) / n0).si
    T_at_Transparency = 2.72548 * uu.K * (1 + z)
    return z, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency

#############################################################
#############################################################    
def densityUz(z):
    newradius = (RR / (1 + z)).to("lyr")
    dilution = (dbh_radius / newradius) ** (3)
    density = (dbhMev_fm3 * dilution.si / mn).si
    return density

#############################################################
#############################################################
def fracIonization(z, ionizationfraction):
    # z is redshift and x is the ionization fraction of the plasma
    # start with the getting the density at the given z
    # from that one calculate the temperature for that z
    n = densityUz(z)
    T = 2.72548 * uu.K * (1 + z)
    kb = cc.k_B
    hbar = cc.hbar
    E = 13.6 * uu.eV
    # here we use the Saha equation
    A0 = ((me / cc.c ** 2 *2 * np.pi * kb * T) / hbar ** 2) ** (3 / 2) # equal 1/lambda**3
    A1 = (2 / n * A0 * np.exp(-(E / (kb * T)).si)).si
    # https://www.astro.umd.edu/~miller/teaching/astr422/lecture20.pdf equation 4
    # Saha equation https://en.wikipedia.org/wiki/Saha_ionization_equation
    # this is the number density
    A2 = ionizationfraction * ionizationfraction / (1 - ionizationfraction)
    return A2 - A1

#############################################################
#############################################################


def interpolateProtonFraction(y, lowestPF, densityPreBigBang, densityPostBigBang, f):
    if y >= densityPreBigBang:
        return lowestPF
    if y < densityPostBigBang:
        return 1.0
    return float(f(y))


def alphaZ(x):
    alpha = math.pi / 4 - math.asin(1 / math.sqrt(2) / (1 + x))
    return alpha


def z_Out_Of_Alpha(alpha):
    z = 1.0 / math.sin(pi4 - alpha) / sqrt2 - 1.0
    return z


def alpha_Out_Of_d_HU(d_HU):
    alpha = pi4 - np.asin((1.0 - d_HU) / sqrt2)
    return alpha


def z_Out_Of_d_HU(d_HU):
    alpha = alpha_Out_Of_d_HU(d_HU)
    z = z_Out_Of_Alpha(alpha)
    return z


def d_HU_epoch(R0, z):
    alpha = alphaZ(z)
    d_HU = R0 * (1 - math.cos(alpha) + math.sin(alpha))
    return d_HU

def actualVS(x, x0):
    beta = x0[0]
    n0 = x0[1]
    vs0 = 1 / np.sqrt(3)
    return vs0 / (1 + np.exp(beta * (x - n0)))

def findVSoundCurveParameters(yy):
    def errorf(x):
        error = 0
        for xx in yy.itertuples():
            error += (actualVS(xx.t, x) - xx.cs2) ** 2
        return error

    x0 = [6.48648947e-03, 1.05823880e+03]
    results = minimize(errorf, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

    df = {}
    for x0 in yy.itertuples():
        df[x0.t] =actualVS(x0.t, results.x)

    df1 = pd.DataFrame.from_dict(df, orient="index", columns=["cs2"])
    df1["t"] = df1.index

    return df1,results.x




# def whatTimeRadius(y):
#     dilution = float( y / dbh_y )
#     radius = dbh_radius.to("lyr") / dilution ** (1 / 3)
#     t = (radius - dbh_radius) / cc.c
#     return t, y, radius.to("lyr")


# def whatIsY(t):
#     radius = (t * cc.c + dbh_radius).to("lyr")
#     dilution = float(  (dbh_radius / radius) ** 3  )
#     y = dbh_y * dilution
#     return t.to(uu.s).value, y.si, radius


def densityU(t, u):
    hydrogenatom = 1.66E-24 * uu.g
    newradius = t * cc.c * u + dbh_radius
    dilution = (dbh_radius / newradius) ** (3)
    density = (dbhMev_fm3 * dilution.si)
    return density


def atmU(t, u):  # fraction of standard atmospherica pressure and number of atoms per cubic meter
    hydrogenatom = 1.66E-24 * uu.g
    newradius = t * cc.c * u + dbh_radius
    dilution = (dbh_radius / newradius) ** (3)
    density = (dbh * dilution.si).si
    # fraction of standard atmospherica pressure
    numatm = (density / hydrogenatom / oneATM_atoms).si
    # number of atoms per cubic meter
    numatm_cubic_meter = (density / hydrogenatom).si
    return numatm, numatm_cubic_meter


def whatIsTemp(energy):
    kb = cc.k_B
    return (2 / 3 * energy / kb).si



def whatIsTime(y):
    dilution = float( y / dbh_y )
    radius = dbh_radius.to("lyr") / dilution ** (1 / 3)
    t = (radius - dbh_radius) / cc.c
    return t.si.value

def whatIsRadius(y):
    dilution = float( y / dbh_y )
    radius = dbh_radius.to("lyr") / dilution ** (1 / 3)
    return radius.to("lyr").value


