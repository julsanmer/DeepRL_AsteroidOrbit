import numpy as np


def asteroid_dynamics(t, x, asteroid_dict, c=(1,1,1)):
    """
    Dynamics function for solve_ivp.

    Args:
        t (float): current time
        x (ndarray, shape (6,)): state vector [r(3), v(3)]
        c (tuple): adimensional factors (T, R, V)
        mascon_dict (dict): contains 'muM' (list/array of mascon masses),
                            'xyzM' (array of mascon positions, shape (N,3))
        omega (ndarray, shape (3,)): angular velocity vector

    Returns:
        dxdt (ndarray, shape (6,)): time derivative of state
    """
    # Unfold asteroid parametes
    mascon = asteroid_dict['mascon']
    omega = asteroid_dict['omega']

    # Unpack parameters
    t_ad, r_ad, v_ad = c
    pos = x[0:3]
    vel = x[3:6]

    # Initialize gravitational acceleration
    a_grav = np.zeros(3)

    # Obtain mascon parameters
    muM = mascon['muM']
    xyzM = mascon['xyzM']

    # Loop through masses
    for i in range(len(muM)):
        dpos = pos*r_ad - xyzM[i]
        dr_norm3 = np.linalg.norm(dpos)**3
        a_grav -= muM[i] * dpos / dr_norm3

    # Rotating frame accelerations
    coriolis = -2*np.cross(omega, vel)
    centrifugal = -np.cross(omega, np.cross(omega, pos))

    # Total acceleration
    acc = a_grav * (t_ad/v_ad) + coriolis * t_ad + centrifugal * (r_ad*t_ad/v_ad)

    # State derivative
    dposdt = vel * (v_ad*t_ad/r_ad)
    dveldt = acc
    dxdt = np.hstack((dposdt,
                      dveldt))

    return dxdt