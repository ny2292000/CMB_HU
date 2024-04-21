import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
M = 1.989e30  # mass of the Sun in kg
c = 299792458  # speed of light in m/s
GM = 1.32712440041e20  # gravitational parameter for the Sun in m^3/s^2
e = 0.205630  # eccentricity of Mercury
a_mercury = 57.91e9  # semi-major axis of Mercury's orbit in meters

# Calculate specific angular momentum for Mercury
h = np.sqrt(G*M * a_mercury * (1 - e**2))

def r_derivatives(theta, h, e, GM, x):
    sigma = x[0]
    k = x[1]
    # Expression for r, r_dot, and r_double_dot
    r = (h ** 2 / GM) / (1 + k * e * np.cos(theta / sigma))
    r_dot = -GM*e*np.sin(theta/sigma)/(h*sigma)
    r_double_dot =  GM**3*e*(e*np.cos(theta/sigma) + 1)**2*np.cos(theta/sigma)/(h**4*sigma**2)
    return r, r_dot, r_double_dot


def error_function(x):
    def integrand(theta):
        r, r_dot, r_double_dot = r_derivatives(theta, h, e, GM, x)
        theta_dot = h / r**2
        v_squared = r_dot**2 + r**2 * theta_dot**2
        c = 299792458
        vc = v_squared / c**2
        gamma_v = 1/np.sqrt(1 - vc)
        v = np.sqrt(v_squared)
        B =(1 + (gamma_v - 1) * (r_dot / v) ** 2)
        a_theoretical = -GM /gamma_v/ r ** 2/B
        a_numerical = r_double_dot - r * theta_dot ** 2  # Assuming r_double_dot is defined to calculate \ddot{r}
        return (a_theoretical - a_numerical)**2
    sigma = x[0]
    integral_error, _ = quad(integrand, 0, 2*np.pi/sigma, epsabs=1e-14, epsrel=1e-14)
    return 100*integral_error

# Initial guess for sigma
initial_x = [0.9, 1.0]
result = minimize(error_function, initial_x, method='Nelder-Mead', tol=1E-14)
print("Optimized sigma:", result.x)


# Constants
orbital_period_days = 88  # Orbital period of Mercury in days
days_per_year = 365.25  # Average number of days per year, accounting for leap years
years_per_century = 100  # Number of years in a century
optimized_sigma = result.x[0] # The optimized value for sigma
optimized_eccentricity = result.x[1]*e

# Calculate the number of orbits Mercury completes in a century
orbits_per_century = (days_per_year / orbital_period_days) * years_per_century

# Calculate the precession per orbit in radians
precession_per_orbit_radians = 2 * np.pi * (optimized_sigma - 1)

# Convert precession per orbit to arcseconds
precession_per_orbit_arcseconds = np.degrees(precession_per_orbit_radians) * 3600

# Total precession in arcseconds per century
total_precession_arcseconds_per_century = precession_per_orbit_arcseconds * orbits_per_century
print(total_precession_arcseconds_per_century)
