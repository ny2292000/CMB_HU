import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
from astropy import constants as cc, units as uu

def calc_ab(v0,GM, e):
    a = GM/(2*v0**2)
    b= a*np.sqrt(e**2-1)
    return a, b





# Constants
G = cc.G  # gravitational constant in m^3 kg^-1 s^-2
M = cc.M_sun  # mass of the Sun in kg
c = cc.c.si.value  # speed of light in m/s
GM = (G*M).si.value  # gravitational parameter for the Sun in m^3/s^2

# Initial conditions
m_sun = cc.M_sun.si.value
r_sun = cc.R_sun.si.value
# Constants for the hyperbola
e=1.2
v0 = 5e5

a, b = calc_ab(v0, GM,e)
print(a/r_sun, b/r_sun)




def r_derivatives(theta,a0, e0, b0, afactor, bfactor, efactor, v0):
    a= a0*afactor
    e=  e0*efactor
    b = b0*bfactor
    # Expression for r, r_dot, and r_double_dot
    r = a*(e**2 - 1)/(e*np.cos(theta) + 1)
    omega= b*v0*(e*np.cos(theta) + 1)**2/(a**2*(e**2 - 1)**2)
    dr_dt= b*e*c*np.sin(theta)/(a*(e**2 - 1))
    d2r_dt2 = b**2*e*c**2*(e*np.cos(theta) + 1)**2*np.cos(theta)/(a**3*(e**2 - 1)**3)
    return r, dr_dt, d2r_dt2, omega

def calc_r(theta, a, e):
    r = a*(e**2 - 1)/(e*np.cos(theta) + 1)
    return r


def newtonian_grav_accel(GM, r):
    a_theoretical = -GM / r**2
    return a_theoretical

def new_grav_accel(GM, r, r_dot):
    alpha = np.sqrt(2*GM/(np.abs(r)*c**2)) # alpha is alreayd v/c
    v = alpha*c
    gamma_v = 1 / np.sqrt(1 -alpha**2)
    B = (1 + (gamma_v - 1) * (r_dot / v)**2)
    a_theoretical = -GM / gamma_v / r**2 / B
    return a_theoretical

def error_functionHU(x):
    afactor, bfactor, efactor = x
    theta_inf = np.arccos(-1 / e)/1.1
    def integrand(theta):
        r, r_dot, r_double_dot, omega = r_derivatives(theta,a, e, b, afactor, bfactor, efactor, v0)
        a_theoretical = new_grav_accel(GM, r, r_dot)
        a_numerical = r_double_dot - r**2*omega # Assuming r_double_dot is defined to calculate \ddot{r}
        return (a_theoretical - a_numerical)**2

    integral_error, _ = quad(integrand, -theta_inf, theta_inf, epsabs=1e-14, epsrel=1e-14)  # Use the correct limits
    return 1E6*integral_error

def error_functionNewton(x):
    afactor, bfactor, efactor = x
    theta_inf = np.arccos(-1 / e)/1.1
    def integrand(theta):
        r, r_dot, r_double_dot, omega = r_derivatives(theta,a, e, b, afactor, bfactor, efactor, v0)
        a_theoretical = newtonian_grav_accel(GM, r)
        a_numerical = r_double_dot - r**2*omega  # Assuming r_double_dot is defined to calculate \ddot{r}
        return (a_theoretical - a_numerical)**2

    integral_error, _ = quad(integrand, -theta_inf, theta_inf, epsabs=1e-14, epsrel=1e-14)  # Use the correct limits
    return 1E6*integral_error

def optimize_trajectory(errf):
    initial_guess = [1, 1, 1]  # Initial guesses for sigma and e
    result = minimize(errf, initial_guess, method='Nelder-Mead', tol=1e-6)
    afactor_optimized, bfactor_optimized, efactor_optimized = result.x
    optimized_a = a*afactor_optimized
    optimized_b = b*bfactor_optimized
    optimized_e = e*efactor_optimized
    print("optimized_a =", optimized_a)
    print("optimized_b =", optimized_b)
    print("optimized_e =", optimized_e)

    # Calculate the deflection angle Δθ in radians
    delta_theta = 2 * np.arctan(np.sqrt(optimized_e**2 - 1) / optimized_e)
    delta_theta_degrees = np.degrees(delta_theta)
    print(f"The deflection angle Δθ is {delta_theta:.6}")
    return delta_theta, afactor_optimized,bfactor_optimized,efactor_optimized


delta_theta, afactor_optimized,bfactor_optimized,efactor_optimized = optimize_trajectory(error_functionNewton)
# optimize_trajectory(error_functionHU)
print(delta_theta, afactor_optimized,bfactor_optimized,efactor_optimized)


# Function to calculate r based on theta

# Generate theta values from -theta_inf to theta_inf
theta_inf = np.arccos(-1 / e)/1.5
# p = 20*a
# theta_inf = np.arccos(1-a/p/(e**2-1))
theta_values = np.linspace(-theta_inf, theta_inf, 1000)

# Plotting the hyperbola
plt.figure(figsize=(8, 6))



optimized_a = 1128828422.832965
optimized_b = 0.0003455479521949524
optimized_e = 1.4486969791557203

r_values = calc_r(theta_values, optimized_a, optimized_e)/r_sun
x_values = r_values * np.cos(theta_values)
y_values = r_values * np.sin(theta_values)

plt.plot(x_values, y_values, label=f'CalcHyperbola: a={optimized_a}, e={optimized_e}')
# Calculate r for each theta
# Convert polar coordinates to Cartesian for plotting
x_values = r_values * np.cos(theta_values)
y_values = r_values * np.sin(theta_values)

plt.plot(x_values, y_values, label=f'Hyperbola: a={a}, e={e}')
plt.scatter([0], [0], color='yellow', label='Central Body')  # Central body (e.g., the Sun)
plt.title('Hyperbolic Trajectory around a Central Body')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.axhline(0, color='grey', linestyle='--')  # Asymptote
plt.axvline(0, color='grey', linestyle='--')  # Asymptote
plt.grid(True)
plt.legend()
plt.axis('equal')  # Ensure equal scaling for x and y axes
plt.show()

