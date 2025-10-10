import numpy as np
from matplotlib import pyplot as plt
import math

# %% transform form cartesian to polar based on Noise Mapping convention
def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Cartesian coordinates. Must have the same shape.
    
    Returns
    -------
    r : np.ndarray
        Radius
    theta : np.ndarray
        Polar angle (0 ≤ θ ≤ π)
    phi : np.ndarray
        Azimuthal angle (0 ≤ φ < 2π)
    """
    # Compute spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)              # radius
    theta = np.arctan2(np.sqrt(x**2 + y**2), z) # polar angle from z-axis
    theta = np.pi/2 - theta                      # adjust to go from +pi/2 (top) to -pi/2 (bottom)
    phi = np.arctan2(y, x)                       # azimuthal angle
    phi = np.mod(phi, 2*np.pi)                   # make phi range 0 to 2pi
    
    return r, theta, phi

rad = 1

th = np.array([0.        , 0.26179939, 0.52359878, 0.78539816, 1.04719755,
       1.30899694, 1.57079633, 1.83259571, 2.0943951 , 2.35619449,
       2.61799388, 2.87979327, 3.14159265])

ph = np.array([-1.57079633e+00, -1.30191424e+00, -1.28371502e+00, -1.26295695e+00,
       -1.23908189e+00, -1.21136532e+00, -1.17885393e+00, -1.14027621e+00,
       -1.09391418e+00, -1.03742240e+00, -9.67584849e-01, -8.80030803e-01,
       -7.69041336e-01, -6.27884764e-01, -4.50725363e-01, -2.37454088e-01,
        8.99999998e-05,  2.37284041e-01,  4.50579510e-01,  6.27766868e-01,
        7.68948384e-01,  8.79957729e-01,  9.67526917e-01,  1.03737587e+00,
        1.09387625e+00,  1.14024486e+00,  1.17882767e+00,  1.21134305e+00,
        1.23906280e+00,  1.26294042e+00,  1.28370059e+00,  1.30190154e+00,
        1.57079633e+00])

LEVELS_th_ph = np.array([[55.3, 59. , 62.4, 67.8, 64.1, 65.9, 63.4, 65.7, 59.8, 65.8, 62.5,
        59. , 55.3],
       [55.3, 59. , 62.4, 67.8, 64.1, 65.9, 63.4, 65.7, 59.8, 65.8, 62.5,
        59. , 55.3],
       [47.6, 51.2, 54.7, 64.2, 58.5, 61.2, 54.8, 63.3, 59.5, 59.3, 61.2,
        51.2, 47.6],
       [46.1, 49.8, 53.2, 55.5, 55.2, 52.7, 54.9, 56.2, 53.6, 57. , 63.4,
        49.8, 46.1],
       [55.4, 59. , 62.5, 55.6, 59.5, 61.5, 63.8, 60. , 63.1, 61.3, 62.5,
        59. , 55.4],
       [46.2, 49.8, 53.3, 55.1, 52.9, 58.6, 53.8, 47. , 58.3, 52.5, 57.1,
        49.8, 46.2],
       [54. , 57.6, 61.1, 61. , 52.8, 59.7, 61.2, 54.8, 63.9, 52. , 61.3,
        57.6, 54. ],
       [45.7, 49.4, 52.8, 56.6, 56. , 53.7, 49.6, 54.5, 64.1, 50.3, 54.4,
        49.4, 45.7],
       [42.6, 46.2, 49.7, 53. , 51.9, 59.9, 60.2, 57.7, 50.8, 52.9, 54.4,
        46.2, 42.6],
       [51.6, 55.3, 58.8, 52.1, 58.4, 58.7, 53.9, 51.3, 46.4, 47.5, 56. ,
        55.3, 51.6],
       [48.2, 51.9, 55.4, 47.1, 52.5, 48.9, 52.9, 46.6, 47.3, 47.3, 49.7,
        51.9, 48.2],
       [54.8, 58.5, 61.9, 50.5, 50. , 51.2, 56.1, 52.2, 59.2, 60.1, 65. ,
        58.5, 54.8],
       [44.2, 47.8, 51.3, 46.7, 57.4, 49.6, 55.6, 51.2, 53.6, 54.4, 50.4,
        47.8, 44.2],
       [47. , 50.7, 54.1, 51.7, 54.3, 59.4, 58.6, 52.3, 53.4, 59.7, 59. ,
        50.7, 47. ],
       [51.7, 55.4, 58.9, 47.3, 49.6, 47.2, 54.4, 45.6, 49. , 51.7, 54.4,
        55.4, 51.7],
       [54.1, 57.7, 61.2, 51.4, 47.2, 42.6, 45.8, 45.3, 44.7, 55.4, 54.4,
        57.7, 54.1],
       [39.9, 43.5, 47. , 53.2, 50.6, 49.3, 50.1, 46.2, 47.9, 48.6, 51.3,
        43.5, 39.9],
       [44.1, 47.8, 51.2, 45.7, 43.7, 51.9, 52.1, 51.5, 48.4, 50.1, 61.3,
        47.8, 44.1],
       [46.7, 50.4, 53.9, 47.6, 42.6, 46.7, 40.6, 48. , 52.9, 52.6, 52.9,
        50.4, 46.7],
       [43.7, 47.4, 50.9, 52.4, 49.8, 49.5, 41.7, 46.5, 50.5, 56.4, 57.2,
        47.4, 43.7],
       [53.1, 56.7, 60.2, 52.4, 53.5, 47.5, 50.9, 50.6, 53.7, 48.8, 56.3,
        56.7, 53.1],
       [47.7, 51.4, 54.8, 47.6, 51. , 46.2, 55.9, 56.4, 51.5, 53.4, 53.9,
        51.4, 47.7],
       [48.2, 51.8, 55.3, 55.4, 52.9, 52.1, 50.9, 52.1, 52. , 56.1, 53.4,
        51.8, 48.2],
       [45.3, 49. , 52.5, 56.1, 53. , 51.8, 46.9, 49.9, 63.5, 54.5, 52.4,
        49. , 45.3],
       [52.3, 56. , 59.4, 56.9, 59.7, 59. , 56.9, 59.9, 60.2, 55.8, 59.8,
        56. , 52.3],
       [49.3, 53. , 56.5, 49.9, 50.4, 49.2, 47.9, 56.1, 52.7, 51.1, 57.3,
        53. , 49.3],
       [53.8, 57.4, 60.9, 58.8, 58.4, 53.2, 53.5, 59.7, 58.4, 57. , 61.2,
        57.4, 53.8],
       [47.1, 50.7, 54.2, 55.7, 50.4, 47.5, 52.9, 54.8, 53.9, 57.7, 59.7,
        50.7, 47.1],
       [47.1, 50.7, 54.2, 55.1, 55.1, 60.2, 61.2, 57.8, 58. , 66. , 50.7,
        50.7, 47.1],
       [49.2, 52.8, 56.3, 58.3, 56.3, 55.6, 53.3, 51.4, 57.9, 58.5, 63.7,
        52.8, 49.2],
       [51. , 54.6, 58.1, 59.2, 57.6, 60.1, 60.4, 59.6, 48.5, 62.4, 60. ,
        54.6, 51. ],
       [51.5, 55.1, 58.6, 59.8, 56.7, 53.9, 49.4, 45.2, 59. , 65.6, 56.2,
        55.1, 51.5],
       [51.5, 55.1, 58.6, 59.8, 56.7, 53.9, 49.4, 45.2, 59. , 65.6, 56.2,
        55.1, 51.5]])
# %%Plot My system
##################
th_all, ph_all = np.meshgrid(th,ph)
x = rad* np.cos(th_all)
y = -rad * np.sin(th_all) * np.sin(ph_all)
z = rad - (rad *np.sin(th_all) * np.cos(ph_all))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=LEVELS_th_ph, cmap='viridis', s=50)

# --- Add labels and title ---
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(0,1)

# %% Inverse to new polar coordinates as Noise model convention
# inverse transform
r2, th2, ph2 = cartesian_to_spherical(x, y, z)

x2 = rad* np.cos(th2)
y2 = rad * np.sin(th2) * np.sin(ph2)
z2 = rad - (rad *np.sin(th2) * np.cos(ph2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x2, y2, z2, c=LEVELS_th_ph, cmap='viridis', s=50)

# --- Add labels and title ---
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

# Rotation (clockwise 90 degrees in x-y plane)
theta = np.pi / 2  # 90 degrees

# Apply rotation around Y-axis
x_rot = x2 * np.cos(theta) + z2 * np.sin(theta)
y_rot = y2
z_rot = -x2 * np.sin(theta) + z2 * np.cos(theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x_rot, y_rot, z_rot, c=LEVELS_th_ph, cmap='viridis', s=50)

# --- Add labels and title ---
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.show()