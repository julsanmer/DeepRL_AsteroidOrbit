import casadi as ca


def collision_constraint(x, ellipsoid_params):
    a, b, c = ellipsoid_params
    ellip_fun = ((x[0]/a)**2
                 + (x[1]/b)**2
                 + (x[2]/c)**2)
    g = [ellip_fun]
    lbg = [1.0]
    ubg = [ca.inf]

    return g, lbg, ubg


def escape_constraint(x, rmax):
    r = ca.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    g = [r]
    lbg = [0]
    ubg = [rmax]

    return g, lbg, ubg
