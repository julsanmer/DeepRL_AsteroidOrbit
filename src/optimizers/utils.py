import numpy as np


def unpack_wopt(w_opt, T_ad, x0_ad, c=(1,1,1)):
    """
    Unpack CasADi optimization vector w_opt into states and impulses.

    Args:
        w_opt (array): flattened optimizer output
        N (int): number of steps (same as init_guess['N'])

    Returns:
        X (array): shape (N+1, 6), trajectory states
        dV (array): shape (N, 3), impulses
    """
    N = len(T_ad)
    T, R, V = c

    # First 6(N-1) entries = states (excluding x0)
    x_block = w_opt[:6*(N-1)]

    # Remaining entries = dv+ and dv-
    dv_block = w_opt[6*(N-1):]

    # Rebuild trajectory
    X = [x0_ad]
    for k in range(N-1):
        Xk = x_block[6*k:6*(k+1)]
        X.append(Xk)

    # Rebuild impulses
    dV = []
    for k in range(N - 1):
        dv_pos = dv_block[6*k:6*k+3]
        dv_neg = dv_block[6*k+3:6*(k+1)]
        dV.append(dv_pos - dv_neg)

    # Add zero for last impulse
    dV.append(np.zeros(3))

    # Make arrays
    X = np.array(X)
    dV = np.array(dV)
    t = T_ad * T
    X[:,0:3] *= R
    X[:,3:6] *= V
    dV *= V

    return t, X, dV
