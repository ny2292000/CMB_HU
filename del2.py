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


def r_derivatives(theta, h, e, GM, x):
    sigma = x[0]
    # Expression for r, r_dot, and r_double_dot
    r = h ** 2 / (GM * (e * np.cos(sigma * theta) + 1))
    theta_dot = h / r ** 2
    d2r_dt2 = GM ** 3 * e * sigma ** 2 * (e * np.cos(sigma * theta) + 1) ** 2 * np.cos(sigma * theta) / h ** 4
    dr_dt = GM * e * sigma * np.sin(sigma * theta) / h
    return r, dr_dt, d2r_dt2, theta_dot


def error_functionHU(x, h, e, GM):
    def integrand(theta):
        r, r_dot, r_double_dot, theta_dot = r_derivatives(theta, h, e, GM, x[0])
        a_theoretical = accel_HU(GM, r, r_dot, theta_dot)
        a_numerical = r_double_dot - r * theta_dot ** 2
        return (a_theoretical - a_numerical) ** 2

    sigma = x[0]
    integral_error, _ = quad(integrand, 0, 2 * np.pi, epsabs=1e-14, epsrel=1e-14)
    return 100 * integral_error


def accel_HU(GM, r, r_dot, theta_dot):
    v = np.sqrt(r_dot ** 2 + r ** 2 * theta_dot ** 2)
    gamma_v = 1 / np.sqrt(1 - v ** 2 / c ** 2)
    A = gamma_v ** 2
    a_theoretical = -GM / r ** 2 / A
    return a_theoretical


def calculatePrecession(errorf, a, e, T):
    h = np.sqrt(GM * a * (1 - e ** 2))
    # Initial guess for sigma
    initial_x = [0.95]
    result = minimize(errorf, initial_x, args=(h, e, GM), method='Nelder-Mead', tol=1E-16)
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

plt.show()
