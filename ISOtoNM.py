import numpy as np
def ISOtoNM (rad, th,ph, LEVELS_th_ph):
    """the application of this function is exemplified in Coord_changes_2.py

    Args:
        th (array): _description_
        ph (array): _description_
        LEVELS_th_ph (array): _description_
    """
    # 1 # 
    # Convert ISO Spherical (r, theta, phi) to Cartesian (x, y, z)
    th_all, ph_all = np.meshgrid(th,ph)
    x = rad* np.cos(th_all)
    y = -rad * np.sin(th_all) * np.sin(ph_all)
    z = -rad *np.sin(th_all) * np.cos(ph_all)
    
    # 2 #
    # Convert Cartesian (x, y, z) to Noise Map spherical (r, theta', phi')
    ########################
    # Compute radius (should be 1 if unit sphere)
    r_new = np.sqrt(x**2 + y**2 + z**2)

    # Theta' is polar angle from top (90 deg) to bottom (-90 deg)
    # Elevation = arcsin(z/r)
    theta_prime = np.degrees(np.arcsin(z / r_new))  # in degrees from -90 to 90

    # Phi' is azimuth from 0 to 360
    phi_prime = np.degrees(np.arctan2(y, x))  # arctan2 gives -180 to 180
    phi_prime = (phi_prime + 360) % 360  # convert to 0-360

    theta_rad = np.radians(theta_prime)
    phi_rad = np.radians(phi_prime)
    
    # 3 # Convert back to Cartesian for plotting
    X = r_new * np.cos(theta_rad) * np.cos(phi_rad)
    Y = r_new * np.cos(theta_rad) * np.sin(phi_rad)
    Z = r_new * np.sin(theta_rad)

    return (theta_prime, phi_prime, LEVELS_th_ph)