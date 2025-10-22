import numpy as np


def cartesian_to_rtn(dv, pos, vel, omega):
    # Compute inertial velocity
    omega_cross_r = np.cross(np.tile(omega,(len(pos), 1)),
                             pos)
    vel_N = vel + omega_cross_r

    # Initialize output
    dv_rtn = np.zeros_like(dv)

    for i in range(len(pos)):
        ri = pos[i]
        vi = vel_N[i]

        # Unit vectors of RTN frame
        eR = ri / np.linalg.norm(ri)
        h = np.cross(ri, vi)
        eN = h / np.linalg.norm(h)
        eT = np.cross(eN, eR)

        # Rotation matrix (columns = RTN axes in inertial frame)
        C_RTN_to_I = np.column_stack((eR, eT, eN))

        # Inverse to go inertial → RTN
        C_I_to_RTN = C_RTN_to_I.T
        dv_rtn[i] = C_I_to_RTN @ dv[i]

    return dv_rtn
