import casadi as ca
import numpy as np

from src.optimizers import (collision_constraint, escape_constraint,
                            shell_penalty, unpack_wopt)


def optimize_safeorbit(initguess_dict,
                       optim_dict,
                       intg):
    """
    Optimize a trajectory with impulsive dV applied AFTER each integration step,
    and with path constraints:
      - collision avoidance with ellipsoid
      - maximum orbital radius

    Args:
        x0 (array): initial state (6D)
        tf (float): final time
        target_body: object containing target info
        init_guess: dict with 'X_ad', 'dV_ad', 'N'
        adim_factors: scaling factors
        intg: CasADi integrator with params = dt
        ellipsoid_params (tuple): (a,b,c) semi-axes of forbidden ellipsoid (optional)
        r_max (float): maximum allowed orbital radius (optional)
    """
    # Adimensional factors
    t_ad, r_ad, v_ad = optim_dict['c']

    # Retrieve initial guess and adimensionalize
    T0 = initguess_dict['T'] / t_ad
    X0 = np.hstack([initguess_dict['X'][:,0:3]/r_ad,
                    initguess_dict['X'][:,3:6]/v_ad])
    dV0 = initguess_dict['dV'] / v_ad
    N = len(T0)

    # Adimensionalize optimization variables
    tf = optim_dict['tf'] / t_ad
    dvmax = optim_dict['dvmax'] / v_ad
    if 'ellip_axes' in optim_dict:
        ellip_axes = optim_dict['ellip_axes'] / r_ad
    if 'rmax' in optim_dict:
        rmax = optim_dict['rmax'] / r_ad

    # Adimensionalize x0
    x0 = np.hstack((optim_dict['x0'][0:3] / r_ad,
                    optim_dict['x0'][3:6] / v_ad))

    # Unfold dict of orbit corridor
    r_in = optim_dict['shell']['bounds'][0]
    r_out = optim_dict['shell']['bounds'][1]
    gam = optim_dict['shell']['gam']

    # State and delta-v variables
    w_x, w0_x, lbw_x, ubw_x = [], [], [], []
    w_dv, w0_dv, lbw_dv, ubw_dv = [], [], [], []

    # Constraints variables
    g_dyn, lbg_dyn, ubg_dyn = [], [], []
    g1, lbg1, ubg1 = [], [], []
    g2, lbg2, ubg2 = [], [], []

    # Shell penalty
    shell_penalties = []

    # Initial state
    xk = ca.DM(x0)

    # Loop through impulses
    for k in range(N-1):
        # Get times
        dt = T0[k+1] - T0[k]

        # 1. Auxiliary variables for L1 norm
        dvk_plus = ca.MX.sym(f'dv_plus_{k}', 3)
        dvk_minus = ca.MX.sym(f'dv_minus_{k}', 3)

        # Delta-v as difference
        dvk = dvk_plus - dvk_minus

        # Store delta-v decision variables
        w_dv += [dvk_plus,
                 dvk_minus]
        w0_dv += (dV0[k, :].tolist() + dV0[k, :].tolist())
        lbw_dv += [0, 0, 0] * 2
        ubw_dv += dvmax.tolist() * 2

        # Impulse applied
        xk_plus = ca.vertcat(xk[0:3],
                             xk[3:6] + dvk)

        # 2. Propagate unforced dynamics after delta-v
        Fk = intg(x0=xk_plus, p=dt)
        xk_unforced = Fk['xf']

        # Compute penalty term to orbit bandwidth
        rk = ca.norm_2(xk_unforced[0:3])
        penalty_k = shell_penalty(rk, a=r_in, b=r_out)
        shell_penalties.append(penalty_k)

        # 3. Next state decision variable
        xk_next = ca.MX.sym(f'x_{k+1}', 6)
        w_x  += [xk_next]
        w0_x += X0[k+1,:].tolist()
        lbw_x += [-np.inf]*6
        ubw_x += [ np.inf]*6

        # 4. Enforce dynamics
        g_dyn   += [xk_next - xk_unforced]
        lbg_dyn += [0]*6
        ubg_dyn += [0]*6

        # ---------------- Path constraints ---------------- #
        # Collision avoidance: ellipsoid inequality
        if 'ellip_axes' in optim_dict:
            g1_k, lbg1_k, ubg1_k = collision_constraint(xk_unforced,
                                                        ellip_axes)
            g1 += g1_k
            lbg1 += lbg1_k
            ubg1 += ubg1_k

        # Escape avoidance: maximum radius
        if 'rmax' in optim_dict:
            g2_k, lbg2_k, ubg2_k = escape_constraint(xk_unforced,
                                                     rmax)
            g2 += g2_k
            lbg2 += lbg2_k
            ubg2 += ubg2_k
        # --------------------------------------------------- #

        # Update for next iteration
        xk = xk_next

    # Objective: sum of ∆V magnitudes
    J_dv = sum(ca.sum1(dv) for dv in w_dv) / np.max(dvmax)
    J_shell = gam*sum(shell_penalties)
    J = J_dv/N + J_shell/N

    # Place all decision variables in vectors
    w = w_x + w_dv
    w0 = w0_x + w0_dv
    lbw = lbw_x + lbw_dv
    ubw = ubw_x + ubw_dv

    # Place all constraints in vectors
    g = g_dyn + g1 + g2
    lbg = lbg_dyn + lbg1 + lbg2
    ubg = ubg_dyn + ubg1 + ubg2

    # NLP setup
    prob = {'f': J,
            'x': ca.vertcat(*w),
            'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', prob,
                       {'ipopt.print_level': 5,
                        'print_time': True,
                        'ipopt.max_iter': 500})

    # Solve
    sol = solver(x0=ca.vertcat(*w0),
                 lbx=lbw, ubx=ubw,
                 lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    # Unpack solution and save in dict
    T, X, dV = unpack_wopt(w_opt, T0, x0, c=optim_dict['c'])
    status = 'success'
    if solver.stats()['return_status'] != 'Solve_Succeeded':
        status = 'fail'
        dV *= 0
    results_dict = {'T': T,
                    'X': X,
                    'dV': dV,
                    'status': status}

    return results_dict
