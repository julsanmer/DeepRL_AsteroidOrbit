import casadi as ca
import numpy as np


def integrate(x0, tf, mascon_dict, omega=[0,0,1], c=[1,1,1] , N=100):
    # Obtain integrator
    intg = integrator3D_casadi(mascon_dict, omega, c)

    # Initialize xk and dt_val
    tk = 0
    xk = x0
    dt_val = tf/N

    T = [tk]
    X = [xk]

    # Loop to integrate
    for i in range(N):
        # Integrate and fill trajectory
        Fk = intg(x0=xk,
                  p=np.concatenate([[dt_val]]))
        tk += dt_val
        xk = Fk['xf'].full().flatten()

        T.append(tk)
        X.append(xk)

    # Transform to array
    T = np.array(T)
    X = np.array(X)

    return T, X


def integrator3D_casadi(asteroid_dict, c):
    """
    3D Cartesian integrator with Coriolis & centrifugal effects.

    Args:
        c : tuple or list of constants, e.g. (c1, c2, c3)
            where c3 can be 1/(Isp*g0) for mass depletion
        gravity_model : class or callable with method acc(r) -> (3,)
            returns gravitational acceleration at position r (3D vector)
    """
    # Unfold asteroid parameters
    mascon = asteroid_dict['mascon']
    omega = asteroid_dict['omega']

    # Unfold adimensional parameters
    t_ad, r_ad, v_ad = c

    # Angular velocity
    omega = ca.DM(omega)

    # Unfold mascon distribution
    muM = mascon['muM']
    xyzM = mascon['xyzM']

    # State: position (3), velocity (3)
    x = ca.MX.sym('x', 6)
    pos = x[0:3]
    vel = x[3:6]

    # Timestep and angular velocity
    dt_sym = ca.MX.sym('dt_sym')

    # Initialize total acceleration
    a_grav = ca.MX.zeros(3, 1)

    # Sum contributions from all point masses
    for i in range(len(muM)):
        dpos = pos*r_ad - ca.DM(xyzM[i])
        dr_norm3 = ca.norm_2(dpos)**3
        a_grav -= muM[i] * dpos/dr_norm3

    # Rotating frame terms
    coriolis = -2*ca.cross(omega, vel)
    centrifugal = -ca.cross(omega, ca.cross(omega, pos))

    # Acceleration total
    acc = a_grav*(t_ad/v_ad) + coriolis*t_ad + centrifugal*(r_ad*t_ad/v_ad)

    # RHS
    rhs = ca.vertcat(vel * (v_ad*t_ad/r_ad),
                     acc)

    # Scale by timestep
    rhs_scaled = rhs * dt_sym

    # Set up DAE
    dae = {
        'x': x,
        'p': ca.vertcat(dt_sym),
        'ode': rhs_scaled
    }
    intg = ca.integrator('intg', 'rk', dae)

    return intg
