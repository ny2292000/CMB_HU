#!/usr/bin/env python
# coding: utf-8

# In[2]:

import matplotlib.pyplot as plt
from lib1 import *
from parameters import *
plt.rcParams['figure.figsize'] = (8,5)




df1,df2, xSound = findVSoundCurveParameters(vssquared)
fig, ax = plt.subplots()
df2.plot(ax=ax)
df0=pd.DataFrame(vssquared, columns=["density","cs2"])
df0.plot(x="density", y="cs2", ax=ax, legend=False)
ax.set_title("Neutronium Normalized Speed of Sound Squared")
ax.set_ylabel("Cs2")
ax.set_xlabel("Density (MeV/fm**3)")
ax.set_ylim(0,0.35)
plt.savefig(imgAddress + "NeutroniumSoundSpeed.png", dpi=300)
plt.show()

# In[2]:



fig, ax = plt.subplots()
for initialT in range(2,10,2):
    t=np.linspace(initialT, initialT+4.5,100)
    GG=initialT/t
    ax.plot(t,GG,label=str(initialT) + "GY")
    ax.legend()
    ax.set_ylabel("G/G0")
    ax.set_xlabel("Age of the Universe")
plt.savefig(imgAddress + "G_Decay.png", dpi=300)
plt.show()

# 
# 
# <h1>The segment below is related to the Baryonic Acoustic Oscillations. I wanted to see how my theory would deal with that. This is work in progress</h1>
# 
# <br>
# <br>

# Current Density is related to the speed of light by Energy Conservation:
# 
# $$ \rho=\frac{c^2}{0.776 *G *2 \pi^2R_0^2} $$

# In[3]:




# # Big Bang Radius and Time

# In[4]:

# ##### constants.c.si, units.lyr.si, np.pi, constants.G.si

# # Calculation of the energy available for the Many-Bangs

# $$ volume(n, \phi) = \frac{ 2*\pi^{(\frac{n-1}{2})}*\int_{0}^{\pi}{sin(x)^{(n - 2)}dx}}{\Gamma(\frac{n-1}{2})}R^3 $$
# 
# #  I will consider the volume of the Visible Universe only (1 radian as opposed to pi radians)

# In[5]:


# I will consider the volume of the WHOLE Universe only (pi radian as opposed to 1 radian)

r=dbh_radius
#Universe Volume



# # So the Big Bang was like 4.2E22 Supernovae explosions of 1E51 Ergs each
# # Total Energy 4.2E66 Joules or 9.54E32 Joules/m^3

# In[6]:


# Mass converted into energy per meter
m0=(9.83E30*units.J/constants.c**2).si
# total mass in a meter
m1=densityBigBang*units.m**3
# Fraction of the total mass into energy
fractionToEnergy=m0/m1
fractionToEnergy, (energyPerNeutron/(constants.m_n*constants.c**2)).si


# $$ \frac{{P\left( {n,{\text{ }}x} \right)}}{{{n_0}{\text{ }}{T_0}{\text{ }}}}{\text{ }} = {\text{ }}\frac{x}{{{T_0}}}\left( {Mp - Mn - Me} \right){\left( {\frac{n}{{{n_0}}}} \right)^2} + \frac{2}{5}[{x^{5/3}} + {\text{ }}{(1 - x)^{5/3}}]{(\frac{{2n}}{{{n_0}}})^{2/3}} - [(2\alpha  - 4{\alpha _L})x(1 - x){\text{ }} + {\alpha _L}]{\left( {\frac{n}{{{n_0}}}} \right)^2} + {\text{ }}\gamma [(2\eta  - 4{\eta _L})x(1 - x){\text{ }} + {\eta _L}]{(\frac{n}{{{n_0}}})^{\gamma  + 1}} $$
# 
# $$ \frac{{\varepsilon (n,x)}}{{{T_0}}} = \frac{x}{{{T_0}}}\left( {Mp - Mn - Me} \right)\frac{n}{{{n_0}}} + \frac{3}{5}[{x^{5/3}} + {\text{ }}{(1 - x)^{5/3}}]{(\frac{{2n}}{{{n_0}}})^{2/3}} - [(2\alpha  - 4{\alpha _L})x(1 - x){\text{ }} + {\alpha _L}]\frac{n}{{{n_0}}} + {\text{ }}[(2\eta  - 4{\eta _L})x(1 - x){\text{ }} + {\eta _L}]{(\frac{n}{{{n_0}}})^\gamma } $$
# 
# 
# 
# $$ {T_0}{\text{ }} = {\text{ }}{\left( {\frac{{3{\pi ^{\text{2}}}{n_0}}}{2}} \right)^{\frac{2}{3}}}\frac{{{\hbar ^2}}}{{2m}} $$

# In[7]:

# Used to find the proton fraction as the neutrons decay between the preBigBang and PostBigBang Epochs









# for y in protonfraction.y:
#     t1, y1, r1 = whatTimeRadius(y)
#     protonfraction["t"] = t1
#     protonfraction["radius"] = r1


# Calculate Universe for gammaT0 and t_unfreezing=xout
calc=True
if calc:
    myU=Universe(eta, alpha, alpha_L, eta_L, T0, gamma, n0)
else:
    myU=Universe(eta, alpha, alpha_L, eta_L, T0, gamma, n0)
    myU.unpickleme()

ax=1



# In[9]:


plt.clf()
ax=myU.ProtonFraction.plot(x="x", y="y", color='r', title="Neutronium Decay", logy=True)
ax.set_xlabel("Proton Fraction")
ax.set_ylabel("n/n0", color='r')
ax.legend("y", loc="lower right")

ax2=ax.twinx()


myU.df.plot(x="ProtonFraction",y="t", color='b', logy=True, ax=ax2, legend=False)
ax2.legend("t", loc="upper right")
ax2.set_ylabel("Time (s)")
plt.savefig(imgAddress + "NeutroniumDecay.png", dpi=300)
plt.show()

# In[10]:


fig = plt.gcf()
plt.clf()

df=myU.df

ax=df.plot(x="t", y="Energy",logx=False, logy=True)
ax.set_xlim(1E2,1E5)
ax.set_ylim(1E-4,1E10)
ax.set_ylabel("Energy (MeV)")
ax1=plt.twinx(ax)
df.plot(x="t", y="Temperature", ax=ax1, color="r",logx=False, logy=True)
ax1.set_xlim(1E2,1E5)
ax1.set_ylim(1E-4,1E10)
for tl in ax1.get_yticklabels():
    tl.set_color('r')
ax1.set_ylabel("Temperature (Kelvin)", color="r")
plt.savefig(imgAddress + "EnergyTemperature.png", dpi=300)
plt.show()

# In[11]:


fig = plt.gcf()
plt.clf()
ax = df.plot (x="t", y="VSound")
ax.set_xlim(1E-4,2E3)
ax.set_ylim(0.0,0.6)
ax.set_title("Neutronium VSound versus Density")
ax.set_xlabel("Time(s)")
ax.set_ylabel("VSound/c")


y_Seq=myU.y_Seq
xcoords = y_Seq.t
colors = ['b','g','r','c','m','y','k','b','g','r']
for xc,c in zip(xcoords,colors):
    ax.axvline(x=xc, label='line at x = {}'.format(xc), c=c)

ax1 = ax.twinx()
ax1.set_ylim(0.0,140)
ax1.plot(df.t, df.y*n0,'r-')
for tl in ax1.get_yticklabels():
    tl.set_color('r')
ax1.set_ylabel("Neutrons/fm3",color="r")
plt.grid(True)
plt.savefig(imgAddress + "NeutroniumSpeedOfSound.png", dpi=300)
plt.show()

# In[12]:


ax=df.plot(x="t", y="Energy", logx=False, logy=True)
ax.set_xlim(1E-4,2E4)
ax.set_ylim(None,1E10)
ax.set_title("Big Bang Energy Profile")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Energy (MeV)")
plt.grid(True)
plt.savefig(imgAddress + "EnergyProfile.png", dpi=300)
plt.show()

# In[13]:


ax=df.plot(x="t", y="Pressure", logx=True, logy=True)
ax.set_xlim(1E-2,1E6)
ax.set_ylim(1E-18,1E41)
ax.set_title("Big Bang Pressure Profile")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Pressure (N/m2)")
ax.legend(loc="center right")
ax1=plt.twinx(ax)

colors = ['b','g','r','c','m','y','k','b','g','r']
for xc,c in zip(xcoords,colors):
    ax.axvline(x=xc, label='line at x = {}'.format(xc), c=c)
    
df.plot(x="t", y="Density", logx=True, logy=True, ax=ax1, color="r")

ax1.set_ylabel("Density (Kg/m3)",color="r")
xcoords = y_Seq.t
# colors for the lines
ax1.set_ylabel("Density (Kg/m3)")
ax1.legend(loc="best")
for tl in ax1.get_yticklabels():
    tl.set_color('r')
plt.grid(axis='y')
plt.savefig(imgAddress + "PressureProfile.png", dpi=300)
plt.show()

# In[14]:



plt.clf() 

ax=df.plot(x="t", y="Temperature", logx=True, logy=True, legend=False)
ax.set_xlim(1E2,1E15)
ax.set_ylim(1E-4,5E11)
ax.set_title("Big Bang Temperature Profile")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Temperature (Kelvin)")
ax1=plt.twinx(ax)
df.plot(x="t", y="Density", logx=True, logy=True, legend=False, ax=ax1)
ax1.set_ylabel("Density Kg/m3",color="r")
# x coordinates for the lines
xcoords = y_Seq.t
# colors for the lines
colors = ['b','g','r','c','m','y','k','b','g','r']
plt.grid(True)
for xc,c in zip(xcoords,colors):
    ax.axvline(x=xc, label='line at x = {}'.format(xc), c=c)
for tl in ax1.get_yticklabels():
    tl.set_color('r')
plt.savefig(imgAddress + "TemperatureProfile.png", dpi=300)


# In[15]:


plt.clf() 


ax=df.astype(float).plot(x="t", y="Energy", logx=True, logy=True)
ax.set_xlim(1E-4,1E15)
ax.set_ylim(None,1E7)
ax.set_title("Big Bang Energy Profile")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Energy (MeV)")
plt.savefig(imgAddress + "UniverseEnergyProfile.png", dpi=300)


# In[16]:


fig = plt.gcf()
plt.clf() 


ax=df.plot(x="t", y= "VSound",logx=True, color="b")

ax.set_xlim(1E2,1E6)
ax.set_ylim(0,1.1)

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Sound_Velocity (Vs/c)")
ax.legend(loc="center")
ax.grid(True)
xcoords = y_Seq.t
colors = ['b','g','r','c','m','y','k','b','g','r']
for xc,c in zip(xcoords,colors):
    ax.axvline(x=xc, label='line at x = {}'.format(xc), c=c)

ax2=plt.twinx(ax)    
df.plot(x="t", y="Proton_Fraction", ax=ax2,logx=True, color="r")
ax2.set_ylim(0,1.1)
ax2.set_ylabel("Proton_Fraction",color="r")
ax2.set_xlim(1E2,1E6)
ax2.legend(loc="upper center")

for tl in ax2.get_yticklabels():
    tl.set_color('r')


plt.savefig(imgAddress + "NeutroniumSpeedOfSoundVersusDecay.png", dpi=100)


# In[17]:


y_Seq


# In[ ]:




