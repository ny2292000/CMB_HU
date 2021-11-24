# Parameters
# Change the address where do you want to save your plots, data
import astropy.constants as cc
import astropy.units as uu
import numpy as np
import pandas as pd
import healpy as hp
from lib3 import B_l
import math
from mpl_toolkits import mplot3d
from scipy.stats import norm

thishome = "/mnt/hd_1/GitHub/AAA_CMB_HU/Data SupernovaLBLgov/"
planck_IQU_SMICA = hp.fitsfunc.read_map(thishome + "COM_CMB_IQU-smica_1024_R2.02_full.fits", dtype=np.float)
(mu_smica, sigma_smica) = norm.fit(planck_IQU_SMICA)
planck_theory_cl = np.loadtxt(thishome +
            "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt", dtype=np.float)
imgAddress ='./img/'
supernovaAddress='./Data SupernovaLBLgov/'
sdssAddress='../sdss/'

time_after_decay=0

glueme=False
saveme = False  # Save my plot or data
loadme = False  # this switch is to be used if you want to read the saved excel file into a dataframe and go withou recalc
NumGalaxies = 10  # number of galaxies to be sampled in the 2-point correlation.  For professional calculation use 500
correctMe=True
d_units=1/uu.fm**3
d_Mev_units=1*uu.MeV/uu.fm**3

mp=(cc.m_p*cc.c**2).to("MeV")
mn=(cc.m_n*cc.c**2).to("MeV")
me=(cc.m_e*cc.c**2).to("MeV")
m_neutrino=0.086E-6*uu.MeV
m_hydrogen=cc.m_p+cc.m_e

K=235.0*uu.MeV
B=16.0*uu.MeV
L=50.0*uu.MeV
S=32.0*uu.MeV
MP= (cc.m_p*cc.c**2).to("MeV")
MN= (cc.m_n*cc.c**2).to("MeV")
ME= (cc.m_e*cc.c**2).to("MeV")
MNE = (cc.m_e*cc.c**2).to("MeV")
pi= np.pi
n0=0.16/uu.fm**3


#Neutronium Stuff
n0=0.16*d_units
gamma= 4/3  # 4/3 is actually the gamma of the gas phase
alpha= 5.87
eta=3.81
alpha_L= 1.2
eta_L= 0.65

# # for n0=0.16
# alpha = 0.518789343227993
# alpha_L = 0.477368057482725
# eta = 4.95844750335946
# eta_L = 2.38254851881421
# gamma = -0.0567134484294691

# # for n0 = 0.054263371
# alpha = 0.380811516179301
# alpha_L = 0.363646873176985
# eta = 94.1933688410687
# eta_L = 47.0311955026687
# gamma = -0.00445029718098298 # this is the gamma of the neutron phase

deBroglieLambda= (cc.h/(m_hydrogen*cc.c)).si
# n0= (1/deBroglieLambda**3/8).to(d_units) #  neutrons per uu.fm**3
# FORCING N0 TO BE THE PAPER'S VALUE

n0_MeV=n0*mn

T0=((3*np.pi**2*n0/2)**(2/3)*cc.hbar**2/(2*cc.m_n)).to("MeV")
n0T0=n0*T0
print(T0)

# Blackholium stuff
# This is the Black Hole density where Fundamental Dilators are deBroglieLambda femtometer apart
# 8 x 1/8 of a FD per cell
# n0_y= (8/deBroglieLambda**3).to(d_units) #  flat hydrogen per uu.fm**3
dbh=m_hydrogen/deBroglieLambda**3
dbhMev_fm3=(dbh*cc.c**2).to(d_Mev_units)
dbh_y=8


# dneutron, dneutronMev_fm3, dilutionNeutron
# This is the Neutron Star density where Fundamental Dilators are 2*deBroglieLambda apart
# 8 * 1/8 a FD per cell
dneutron=cc.m_n/(2*deBroglieLambda)**3
dneutronMev_fm3=(dneutron*cc.c**2).to('MeV/fm**3')
dilutionNeutron=(dbh/dneutron)**(1/3)
neutronenergy=cc.m_n*cc.c**2
dneutron_y=(dneutronMev_fm3/n0_MeV).si.value

#  Number of Atoms per cubic meter at the Current Universe and density.

ccc=cc.c.si
GG=cc.G.si
pi=np.pi
lyr= uu.lyr.si
RR=14.03E9*uu.lyr.si
rho=(ccc**2/(0.776*2*pi**2*GG*RR**2)).si
hydrogenatom=1.66E-24*uu.g
oneATM_atoms=cc.N_A/0.0224*uu.mol/uu.m**3
numberOfAtoms=(rho/hydrogenatom).si
secondsInYear=365.25*24*3600
# rho, numberOfAtoms

tfactor=360

dilutionBlackholium=(rho/dbh)**(1/3)
dilutionBlackholium
dbh_radius=(dilutionBlackholium*RR).to('lyr')
dbh_t=(dbh_radius/cc.c).si
ls=uu.lyr/secondsInYear
BlackholiumRadiusinLightSeconds=dbh_radius.to(ls)
NeutroniumRadius=BlackholiumRadiusinLightSeconds*dilutionNeutron
NeutroniumTime= ((BlackholiumRadiusinLightSeconds*(dilutionNeutron-1))/cc.c).si

vssquared=[[58.86058360352017, 0.01072834645669285],
[74.16859657248726, 0.014862204724409411],
[164.9536822603057, 0.07342519685039367],
[236.94302918017598, 0.140255905511811],
[261.72996757758216, 0.1643700787401574],
[283.0176007410838, 0.19744094488188974],
[305.49791570171374, 0.2353346456692913],
[312.755905511811, 0.2945866141732283],
[318.7401574803149, 0.3249015748031496],
[324.65956461324686, 0.3359251968503937],
[481.1347846225103, 0.33730314960629915],
[797.6053728578045, 0.33730314960629915]
]
vssquarednp=np.array( [[ 1473.7549789124434 , 0.10357773147106887 ],
[ 1355.4456276610904 , 0.12191064237550966 ],
[ 1009.9848159721328 , 0.27097084132871874 ],
[ 881.3141184285442 , 0.374507550673963 ],
[ 848.6010539034853 , 0.4054257993026066 ],
[ 823.6413031407373 , 0.4443432736993886 ],
[ 799.8647019543275 , 0.48511302360304787 ],
[ 792.6814552176963 , 0.5427583386491895 ],
[ 786.9239849535443 , 0.5700013814045977 ],
[ 781.368969728491 , 0.5795905424093751 ],
[ 670.4124972623731 , 0.580778055375975 ],
[ 547.6160008534522 , 0.580778055375975 ]])

vs_pd=pd.DataFrame(vssquarednp,columns=["t","cs2"])



