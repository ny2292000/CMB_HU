import math

import pandas as pd
import scipy
import scipy.integrate as integrate
import scipy.interpolate as sp
from scipy.optimize import minimize
from scipy.special import gamma as gammaF

def volumeCalc(n, phi, r):
    if phi==2:
        return 4/3*np.pi*r**3
    return np.pi ** ((n - 1) / 2) * r ** 3 / gammaF((n - 1) / 2) * \
           integrate.quad(lambda x: np.sin(x) ** int(n - 2), 0, phi)[0]


from parameters import *

H0 = 1
c = 1
R0 = 1
pi4 = math.pi / 4.0
sqrt2 = math.sqrt(2)


class Universe():
    def __init__(self, eta, alpha, alpha_L, eta_L, T0, gamma, n0,vssquaredpd):
        # Calculate gammaT for the region Transparency to Today
        ionizationfraction=0.5
        self.gammaT, self.z_transparency, self.TransparencyRadius, self.TransparencyTime, self.densityAtTransparency, \
            self.T_at_Transparency = findGammaT(ionizationfraction)
        self.k0 = []

        # Calculate Sound Velocitity
        df1,xSound = findVSoundCurveParameters(vssquaredpd)
        self.df1 = df1
        self.df2 = vssquaredpd
        self.xSound=xSound

        # Calculate Proton Fraction
        protonfraction = findprotonfraction(eta, alpha, alpha_L, eta_L, T0, self.gammaT, n0)
        densityBlackholium = dbh_y
        densityNeutronium = dneutron_y
        densityPreBigBang = protonfraction.iloc[-1].y
        densityPostBigBang = protonfraction.iloc[0].y
        self.lowestPF = protonfraction.iloc[-1].x
        densityAtPreFreezing = (300 * (uu.MeV / uu.fm ** 3) / mn / n0).si.value

        self.PreFreezingY = densityAtPreFreezing
        self.PreBigBang = densityPreBigBang
        self.PostBigBang = densityPostBigBang

        densityAtFreezing = (50 * (uu.MeV / uu.fm ** 3) / mn / n0).si.value
        ionizationfraction = 0.5
        z, TransparencyRadius, TransparencyTime, densityAtTransparency, T_at_Transparency = \
                                            findTransparencyEpoch(ionizationfraction)
        densityAtTransparency = densityAtTransparency.value
        densityToday = (rho * cc.c ** 2 / mn / n0).si.value

        self.y_Seq = pd.DataFrame.from_dict({"densityBlackholium": densityBlackholium,
                                             "densityNeutronium": densityNeutronium,
                                             "densityAtPreFreezing": densityAtPreFreezing,
                                             "densityAtFreezing": densityAtFreezing,
                                             "densityPreBigBang": densityPreBigBang,
                                             "densityPostBigBang": densityPostBigBang,
                                             "densityAtTransparency": densityAtTransparency,
                                             "densityToday": densityToday}, orient="index", columns=["y"], dtype=float)

        self.y_Seq["t"] = np.nan
        self.y_Seq["radius"] = np.nan

        for row in self.y_Seq.iterrows():
            t, y, r = whatTimeRadius(row[1]["y"])
            row[1]["t"] = t
            row[1]["radius"] = r

        self.y_Seq = self.y_Seq.sort_values(by="t")

        finalindex = []
        a0 = self.y_Seq.y[0]
        for a in self.y_Seq.y[1:-5]:
            b = np.linspace(start=a0, stop=a, num=101)
            finalindex += list(b[0:-1:])
            a0 = a
        finalindex += [b[-1]]
        for a in self.y_Seq.y[-5::]:
            b = np.geomspace(start=a0, stop=a, num=101)
            finalindex += list(b[0:-1:])
            a0 = a
        finalindex += [b[-1]]
        self.TransparenceIndex = [i for i, x in enumerate(finalindex) if abs(x - densityAtTransparency) / x < 0.01][0]
        self.TransparenceTemperature = T_at_Transparency.value

        self.y_Seq["Density"] = [(x * n0 * mn / cc.c ** 2).si.value for x in self.y_Seq.y]

        finalcolumns = ["t", "y", "radius", "Temperature", "Energy", "Pressure", "Proton_Fraction", "VSound", "Density"]
        self.df = pd.DataFrame(index=finalindex, columns=finalcolumns, dtype=np.float_)
        self.df.y = self.df.index

        protonfractionInterp = sp.interp1d(protonfraction.y, protonfraction.x)
        for i, row in self.df.iterrows():
            y = row["y"]
            t, yy, radius = whatTimeRadius(y)
            x = interpolateProtonFraction(y, self.lowestPF, self.y_Seq.loc["densityPreBigBang", "y"],
                                          self.y_Seq.loc["densityPostBigBang", "y"], protonfractionInterp)
            row["t"] = t
            row["radius"] = radius
            row["Proton_Fraction"] = x
            row["VSound"] = actualVS(t, self.xSound)
            row["Density"] = (row["y"] * n0 * mn / cc.c ** 2).si.value

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
        print("Optimized K0 = ", self.k0)
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
            row["Energy"] = KE(row["y"], row["Proton_Fraction"], eta, alpha, alpha_L, eta_L, T0, gamma, n0)
            row["Pressure"] = Pressure(row["y"], 0, eta, alpha, alpha_L, eta_L, T0, gamma, n0).si.value
        return self.df

    #####################################################
    
    
    def getTempEverywhere(self, y, x, t, tprior, Tempprior, xprior, massDensity):
        if self.PreBigBang < y:
            return 1E-8

        if (self.PreBigBang - y) * (self.PostBigBang - y) <= 0:
            dtemp = (2 / 3 * x * (mn - mp - me - m_neutrino) / cc.k_B).si.value
            return dtemp

        if self.PostBigBang > y:
            return Tempprior * (tprior / t) ** (3 * (self.getgamma(massDensity) - 1))
        
    def getTemperature(self):
        tprior = 0
        xprior = 0
        Tempprior = 1E-4
        for i, row in self.df.iterrows():
            y = row["y"]
            massDensity = row["Density"]
            if (self.PreBigBang - y) * (self.PostBigBang - y) <= 0:
                a = 1

            Temp = self.getTempEverywhere(row["y"], row["Proton_Fraction"], row["t"], tprior, Tempprior, xprior,
                                          massDensity)
            row["Temperature"] = Temp
            tprior = row["t"]
            Tempprior = Temp
            xprior = row["Proton_Fraction"]
        return self.df.Temperature.iloc[self.TransparenceIndex], self.df.Temperature.iloc[-1]

    def find_k0(self, x0):
        def errf(x):
            self.k0 = x
            newTransparencyTemp, newtemp = self.getTemperature()
            # return  (self.TransparenceTemperature - newTransparencyTemp)**2 + 100*(2.725-newtemp)**2
            return (self.TransparenceTemperature - newTransparencyTemp) ** 2

        results = scipy.optimize.minimize(errf, x0, method="Nelder-Mead", options={'xatol': 1e-8, 'disp': True})
        self.k0 = results.x
        return self.k0, results, self.df.Temperature.iloc[-1:]

    def getgamma(self, massDensity):
        massDensity_limit = self.k0[0]  # 10 kg/m3
        a = massDensity_limit / massDensity
        if massDensity < massDensity_limit:
            return 4 / 3
        else:
            return (4 / 3) ** (self.k0[1] * a ** self.k0[2])



    #####################################################
    









def KE(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    EKy =  3/5*2**(2/3)*(x**(5/3) + (-x + 1)**(5/3))*T0*y**(2/3) + (2*(alpha - 2*alpha_L)*(x - 1)*x - alpha_L)*T0*y - (2*(eta - 2*eta_L)*(x - 1)*x - eta_L)*T0*y**gamma
    return EKy.to("MeV").value

def Pressure(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    P =  -1/5*(5*(2*(eta - 2*eta_L)*(x - 1)*x - eta_L)*T0*gamma*y**(gamma - 1)/n0 - 2*2**(2/3)*(x**(5/3) + (-x + 1)**(5/3))*T0/(n0*y**(1/3)) - 5*(2*(alpha - 2*alpha_L)*(x - 1)*x - alpha_L)*T0/n0)*n0**2*y**2
    return P.si

def dKEx(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    # equilibrium equation is d(EK)/dx)= - y*n0*mu
    dEKy_x =  2**(2/3)*T0*y**(2/3)*(x**(2/3) - (-x + 1)**(2/3)) + 2*((alpha - 2*alpha_L)*(x - 1) + (alpha - 2*alpha_L)*x)*T0*y - 2*((eta - 2*eta_L)*(x - 1) + (eta - 2*eta_L)*x)*T0*y**gamma
    return dEKy_x


def findprotonfraction(eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    df = {}
    y0 = 1.0
    for x in np.geomspace(1, 1E-3, 100):
        try:
            # I am solving for y (density) and not x the protonfraction
            # root = scipy.optimize.root(findy, y0, args=(x, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
            root = scipy.optimize.brentq(findy_err, 0, 1, args=(x, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
            df[x] = root
            y0 = root
        except Exception:
            print("failed at {}".format(x))
            df[x] = 0.0
    protonfraction = pd.DataFrame.from_dict(df, orient="index")
    protonfraction["x"] = protonfraction.index
    protonfraction.columns = ["y", "x"]
    protonfraction.index = protonfraction.y
    return protonfraction

def mu(x,y):
    return (cc.hbar * cc.c * (3 * np.pi ** 2 * (1- x)* y * n0) ** (1 / 3)).to("MeV")

def findy(y, xx, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    # equilibrium equation is d(EK)/dx + mu - (MN-MP)
    val = (dKEx(y, xx, eta, alpha, alpha_L, eta_L, T0, gamma, n0) + mu (xx,y) - (MN-MP)).to("MeV")
    return val

def findy_err(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    # equilibrium equation is d(EK)/dx + mu - (MN-MP)
    val = (dKEx(y, x, eta, alpha, alpha_L, eta_L, T0, gamma, n0) + mu (x,y) - (MN-MP)).to("MeV").value
    # print( y, val)
    return val

def findprotonfraction_y(x, y0, eta, alpha, alpha_L, eta_L, T0, gamma, n0):
    try:
        args = (x, eta, alpha, alpha_L, eta_L, T0, gamma, n0)
        bounds = [(0,1.0)]
        root = scipy.optimize.brentq(findy_err, 0,1, args=(x, eta, alpha, alpha_L, eta_L, T0, gamma, n0))
        # root = minimize(findy_err, y0, args=args,method='L-BFGS-B', options={'maxiter': 50000}, bounds=bounds)
        return root
    except Exception:
        print("failed at {}".format(x))
    


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




def whatTimeRadius(y):
    dilution = float(y / 8)
    radius = dbh_radius / dilution ** (1 / 3)
    t = ((radius - dbh_radius) / cc.c).si.value
    return t, y, radius.to("lyr").value


def whatIsY(t):
    radius = (t * cc.c + dbh_radius).to("lyr")
    dilution = (dbh_radius / radius) ** 3
    y = dbh_y * dilution
    return t.to(uu.s).value, y.si, radius.value


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




