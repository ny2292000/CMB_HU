import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy import constants as cc, units as uu

from math import pi

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M_sun = 1.989e30  # Mass of the Sun in kg
c = 299792458  # Speed of light in m/s
GM = G * M_sun

# Venus's orbital parameters
a_venus = 108.2e9  # Semi-major axis in meters
e = e_venus = 0.67  # Eccentricity
days_per_year = 365.25
orbital_period_days = orbital_period_venus_days = 224.7  # Orbital period in Earth days
days_in_century = 36525  # Number of days in a century
years_per_century = 100  # Number of years in a century

# Calculate specific angular momentum for Venus
h = np.sqrt(GM * a_venus * (1 - e ** 2))


def r_derivatives(theta, GM, a, x):
    sigma = x[0]
    k=x[1]
    e = x[2]
    h = np.sqrt(GM * a * (1 - k*e ** 2))
    # Expression for r, r_dot, and r_double_dot
    r= h**2/(GM*(e*k*np.cos(sigma*theta) + 1))
    theta_dot= h/r**2
    dr_dtheta = e*h**2*k*sigma*np.sin(sigma*theta)/(GM*(e*k*np.cos(sigma*theta) + 1)**2)
    d2r_dtheta2 = 2*e**2*h**2*k**2*sigma**2*np.sin(sigma*theta)**2/(GM*(e*k*np.cos(sigma*theta) + 1)**3) + e*h**2*k*sigma**2*np.cos(sigma*theta)/(GM*(e*k*np.cos(sigma*theta) + 1)**2)
    domega_dtheta = -2*GM**2*e*k*sigma*(e*k*np.cos(sigma*theta) + 1)*np.sin(sigma*theta)/h**3
    d2r_dt2 = GM**3*e*k*sigma**2*(e*k*np.cos(sigma*theta) + 1)**2*np.cos(sigma*theta)/h**4
    d2theta_dt2 = -2*GM**4*e*k*sigma*(e*k*np.cos(sigma*theta) + 1)**3*np.sin(sigma*theta)/h**6
    dr_dt = GM*e*k*sigma*np.sin(sigma*theta)/h
    return r, dr_dt, d2r_dt2, theta_dot, h




def error_functionHU(x):
    def integrand(theta):
        r, r_dot, r_double_dot, theta_dot, h = r_derivatives(theta, GM, a, x)
        a_theoretical = accel_HU(GM, r, r_dot, theta_dot)
        a_numerical = r_double_dot - r * theta_dot ** 2  # r * theta_dot ** 2  # Assuming r_double_dot is defined to calculate \ddot{r}
        return (a_theoretical - a_numerical) ** 2

    sigma = x[0]
    integral_error, _ = quad(integrand, 0, 2 * np.pi / sigma, epsabs=1e-14, epsrel=1e-14)
    return 100 * integral_error



def accel_HU(GM, r, r_dot, theta_dot):
    v = np.sqrt(r_dot ** 2 + r ** 2 * theta_dot ** 2)
    gamma_v = 1 / np.sqrt(1 - v ** 2 / c ** 2)
    A = (1 + v ** 2 / c ** 2 - 2 * r_dot / c) ** (3 / 2)*gamma_v**3 *( (gamma_v-1)*r_dot/v+1)
    a_theoretical = -GM / r ** 2 / A
    return a_theoretical


def calculatePrecession(errorf, a, e, T):
    h = np.sqrt(GM * a * (1 - e ** 2))
    # Initial guess for sigma
    initial_x = [1,1,0.2]
    result = minimize(errorf, initial_x, method='Nelder-Mead', tol=1E-16)
    optimized_sigma = result.x[0]  # The optimized value for sigma
    print(result.x)
    days_in_century = 36525  # Number of days in a century

    # Calculation of precession per orbit in radians
    delta_phi_per_orbit = 2 * np.pi * (optimized_sigma - 1)
    delta_phi_per_orbit_GR = (6 * pi * GM) / (a * (1 - e ** 2) * c ** 2)

    # Calculate the number of orbits per century
    orbits_per_century = days_in_century / T

    # Precession per century in arcseconds
    delta_phi_per_century = delta_phi_per_orbit * orbits_per_century * (
                180 / pi) * 3600  # Convert radians to arcseconds
    delta_phi_per_century_GR = delta_phi_per_orbit_GR * orbits_per_century * (
                180 / pi) * 3600  # Convert radians to arcseconds

    return delta_phi_per_century, delta_phi_per_century_GR


a_ = [57.91E9, 108.2E9, 149.6E9, 227.9E9, 778.5E9, 1433.5E9, 2872.5E9, 4495.1E9]
e_ = [0.205630, 0.0067, 0.0167, 0.0934, 0.0489, 0.0565, 0.0457, 0.0113]
T_ = [88, 224.7, 365.25, 687, 4333, 10759, 30660, 60182]


planet = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

df = pd.DataFrame(index=planet, columns=['planet', 'delta_HU', "delta_GR"])
df["planet"] = planet
for i, (a, e, T, name) in enumerate(zip(a_, e_, T_, planet)):
    df.loc[name, ['delta_HU', "delta_GR"]] = calculatePrecession(error_functionHU, a, e, T)

ax = df.plot.scatter(x="planet", y='delta_HU')
df.plot(x="planet", y='delta_GR', ax=ax)