import numpy as np

# Constants for each planet (some values will be hypothetical or approximate)
planetary_data = {
    'Mercury': {
        'mass': 3.30e23,  # kg
        'radius': 2.4397e6,  # m
        'theta': 0.03,  # degrees
        'rotation_period': 58.6667 * 24 * 3600,  # seconds
        'rp': 4.6e10,  # m
        'eccentricity': 0.2056
    },
    'Venus': {
        'mass': 4.87e24,
        'radius': 6.0518e6,
        'theta': 177.4,
        'rotation_period': 243 * 24 * 3600,
        'rp': 1.07577e11,
        'eccentricity': 0.0067
    },
    'Earth': {
        'mass': 5.97e24,
        'radius': 6.371e6,
        'theta': 23.4,
        'rotation_period': 23.933 * 3600,
        'rp': 1.471e11,
        'eccentricity': 0.0167
    },
    'Mars': {
        'mass': 6.42e23,
        'radius': 3.3895e6,
        'theta': 25.2,
        'rotation_period': 24.6 * 3600,
        'rp': 2.066e11,
        'eccentricity': 0.0934
    },
    'Jupiter': {
        'mass': 1.90e27,
        'radius': 6.9911e7,
        'theta': 3.1,
        'rotation_period': 9.9167 * 3600,
        'rp': 7.4052e11,
        'eccentricity': 0.0489
    },
    'Saturn': {
        'mass': 5.68e26,
        'radius': 5.8232e7,
        'theta': 26.7,
        'rotation_period': 10.55 * 3600,
        'rp': 1.35255e12,
        'eccentricity': 0.0565
    }
}

# Calculate the spin angular momentum vector
def calculate_spin_angular_momentum(mass, radius, rotation_period, theta):
    I = (2/5) * mass * radius**2  # Moment of inertia for a sphere
    omega = 2 * np.pi / rotation_period  # Angular velocity
    L = I * omega  # Magnitude of angular momentum
    theta_rad = np.radians(theta)
    S = L * np.array([np.sin(theta_rad), 0, np.cos(theta_rad)])
    return S

# Example calculation
for planet, data in planetary_data.items():
    S = calculate_spin_angular_momentum(
        data['mass'], data['radius'], data['rotation_period'], data['theta']
    )
    print(f"Spin angular momentum vector for {planet}: {S}")

# The calculation for orbital angular momentum L would require additional orbital mechanics equations
# and more specific data about each planet's orbit which can be complex.

