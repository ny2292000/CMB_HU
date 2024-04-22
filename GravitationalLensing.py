import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy import constants as cc, units as uu

# Constants
G = cc.G.si.value  # gravitational constant in m^3 kg^-1 s^-2
M = cc.M_sun.si.value  # mass of the Sun in kg
c = cc.c.si.value  # speed of light in m/s
GM = G*M  # gravitational parameter for the Sun in m^3/s^2

# Constants for Mercury
e=1
# Specific angular momentum (h) for Mercury

# Define the impact parameter and initial velocity
b = 7.0e8  # impact parameter in meters
v_0 = 5.0e4  # initial velocity in m/s (placeholder value)
h = b * v_0

def r_derivatives(theta, h, GM, x):
    sigma = x[0]
    e = x[1]
    # Expression for r, r_dot, and r_double_dot
    r = e / (1 - e * np.sin(theta / sigma))
    r_dot = GM * np.cos(theta/sigma)/(h*sigma)
    r_double_dot = GM*(-e*np.cos(theta/sigma)**2 - e + np.sin(theta/sigma))/(h*sigma**2*(e*np.sin(theta/sigma) - 1))
    theta_dot = h / r**2
    theta_dot_dot = -2*GM**2*(-e*np.sin(theta/sigma) + 1)*np.cos(theta/sigma)/(h**3*e*sigma)
    return r, r_dot, r_double_dot, theta_dot, theta_dot_dot

def new_grav_accel(theta, h, GM, x):
    r, r_dot, r_double_dot, theta_dot, theta_dot_dot = r_derivatives(theta, h, GM, x)
    v_squared = r_dot**2 + r**2 * theta_dot**2
    v = np.sqrt(v_squared)
    gamma_v = 1 / np.sqrt(1 - v**2 / c**2)
    a_theoretical = -GM / gamma_v / r**2 / (1 + (gamma_v - 1) * (r_dot / v)**2)
    return a_theoretical, r, r_double_dot, theta_dot

def error_function(x):
    def integrand(theta):
        a_theoretical, r, r_double_dot, theta_dot = new_grav_accel(theta, h, GM, x)
        # No need to calculate a_theoretical here again, it is returned from the new_grav_accel function
        a_numerical = r_double_dot - r * theta_dot ** 2  # Assuming r_double_dot is defined to calculate \ddot{r}
        return (a_theoretical - a_numerical)**2
    integral_error, _ = quad(integrand, 0, 2 * np.pi, epsabs=1e-6, epsrel=1e-6)
    return 100 * integral_error


# Initial guess for sigma
initial_x = [0.9, 0.33]
result = minimize(error_function, initial_x, method='Nelder-Mead', tol=1E-6)
print("Optimized sigma:", result.x)



# Initial conditions
r0 = 7.0e8  # initial distance from Sun's center in meters (just above the surface)

# Assuming k_optimized is the optimized value of k (eccentricity) obtained from the previous step
e_optimized = result.x[1]  # For example

# Calculate the deflection angle Δθ in radians
delta_theta = 2 * np.arcsin(1 / e_optimized)

# Convert the deflection angle from radians to degrees, if necessary
delta_theta_degrees = np.degrees(delta_theta)

print(f"The deflection angle Δθ is {delta_theta:.6f} radians or {delta_theta_degrees:.6f} degrees.")