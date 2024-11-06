def r_derivatives(theta, h, e, GM, x):
    sigma = x[0]
    # Expression for r, r_dot, and r_double_dot
    r= h**2/(GM*(e*np.cos(sigma*theta) + 1))
    theta_dot= h/r**2
    dr_dtheta = e*h**2*sigma*np.sin(sigma*theta)/(GM*(e*np.cos(sigma*theta) + 1)**2)
    d2r_dtheta2 = 2*e**2*h**2*sigma**2*np.sin(sigma*theta)**2/(GM*(e*np.cos(sigma*theta) + 1)**3) + e*h**2*sigma**2*np.cos(sigma*theta)/(GM*(e*np.cos(sigma*theta) + 1)**2)
    domega_dtheta = -2*GM**2*e*sigma*(e*np.cos(sigma*theta) + 1)*np.sin(sigma*theta)/h**3
    d2r_dt2 = GM**3*e*sigma**2*(e*np.cos(sigma*theta) + 1)**2*np.cos(sigma*theta)/h**4
    d2theta_dt2 = -2*GM**4*e*sigma*(e*np.cos(sigma*theta) + 1)**3*np.sin(sigma*theta)/h**6
    dr_dt = GM*e*sigma*np.sin(sigma*theta)/h
    return r, dr_dt, d2r_dt2, theta_dot


def error_functionHU(x, h, e, GM):
    sigma = x
    def integrand(theta):
        r, r_dot, r_double_dot,theta_dot = r_derivatives(theta, h, e, GM, x)
        a_theoretical = HU_Accel(GM, r, r_dot, theta_dot)
        a_numerical = r_double_dot - r * theta_dot ** 2  # Assuming r_double_dot is defined to calculate \ddot{r}
        return (a_theoretical - a_numerical)**2

    sigma = x[0]
    integral_error, _ = quad(integrand, 0, 2*np.pi, epsabs=1e-14, epsrel=1e-14)
    return 100*integral_error



def HU_Accel(GM, r, r_dot, theta_dot):
    v1 = np.sqrt(2*GM/r)
    gamma_v1 = 1/np.sqrt(1- v1**2/c**2)
    P2 = r * np.sqrt(1 + v1**2/c**2 - 2*r_dot/c)
    HU_factor = gamma_v1 **(-3) / ( (1-gamma_v1)*(v1/c - r_dot/v1)**2 + 1)
    a_theoretical = -GM / P2**2 *HU_factor
    return a_theoretical

def calculatePrecession(errorf, a, e, T):
    h = np.sqrt(GM * a * (1 - e ** 2))
    # Initial guess for sigma
    initial_x = [0.95]
    result = minimize(errorf, initial_x, method='Nelder-Mead', args=(h,e,GM) ,tol=1E-14)
    optimized_sigma = result.x  # The optimized value for sigma
    if( not result.success):
        print(result)
    days_in_century = 36525  # Number of days in a century

    # Calculation of precession per orbit in radians
    delta_phi_per_orbit = 2 * np.pi * (1/optimized_sigma-1)  #CHANGED
    delta_phi_per_orbit_GR = (6 * np.pi * GM) / (a * (1 - e ** 2) * c ** 2)

    # Calculate the number of orbits per century
    orbits_per_century = days_in_century / T

    # Precession per century in arcseconds
    delta_phi_per_century = delta_phi_per_orbit * orbits_per_century * (
                180 / np.pi) * 3600  # Convert radians to arcseconds
    delta_phi_per_century_GR = delta_phi_per_orbit_GR * orbits_per_century * (
                180 / np.pi) * 3600  # Convert radians to arcseconds

    return delta_phi_per_century[0], delta_phi_per_century_GR



a_ = [57.91E9, 108.2E9, 149.6E9, 227.9E9, 778.5E9, 1433.5E9, 2872.5E9, 4495.1E9]
e_ = [0.205630, 0.0067, 0.0167, 0.0934, 0.0489, 0.0565, 0.0457, 0.0113]
T_ = [88, 224.7, 365.25, 687, 4333, 10759, 30660, 60182]


planet = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

df = pd.DataFrame(index=planet, columns=['planet', 'delta_HU', "delta_GR"])
df["planet"] = planet
for i, (a, e, T, name) in enumerate(zip(a_, e_, T_, planet)):
    df.loc[name, ['delta_HU', "delta_GR"]] = calculatePrecession(error_functionHU, a, e, T)


ax = df.plot.scatter(x="planet", y="delta_GR", label='GR Data', color='red')
df.plot(x="planet", y="delta_HU", label='HU Data', color='blue', ax=ax)

ax.set_ylabel('Planet')
ax.legend()

plt.tight_layout()
plt.show()