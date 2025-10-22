import casadi as ca


def softplus(z, k=50.0):
    # numerically stable softplus
    return ca.log1p(ca.exp(k*z)) / k
    #return ca.if_else(k * z > 50, z, ca.log1p(ca.exp(k * z)) / k)

def shell_penalty(r, a, b, s=1.0, k=20.0):
    """"
    Smooth function with slope -> flat -> slope.
    If opposite=True, the second slope is negative.
    """
    x = 2*(r-a)/(b-a)-1
    x_a = -1
    x_b = 1

    f_raw = -(s*x - s*softplus(x-x_a, k) - s*softplus(x-x_b, k))

    # Compute flat value: function at the center of the flat region
    flat_val = -(s*(x_a+x_b)/2 - s*softplus((x_a+x_b)/2 - x_a, k)
                 - s*softplus((x_a+x_b)/2 - x_b, k))

    return f_raw - flat_val