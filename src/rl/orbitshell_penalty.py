import numpy as np

def softplus(z, k=50.0):
    """Numerically stable softplus approximation."""
    # Use np.log1p for stability: log(1+exp(x)) instead of log(exp(x)+1)
    return np.log1p(np.exp(k * z)) / k
    # If needed: np.where(k * z > 50, z, np.log1p(np.exp(k * z)) / k)

def shell_penalty(r, a, b, s=1.0, k=20.0):
    """
    Smooth penalty with slope -> flat -> slope.
    Equivalent to the CasADi version, but in NumPy.
    """
    x = 2 * (r - a) / (b - a) - 1
    x_a, x_b = -1, 1

    f_raw = -(s * x
              - s * softplus(x - x_a, k)
              - s * softplus(x - x_b, k))

    # flat value = function at midpoint of flat region
    mid = (x_a + x_b) / 2
    flat_val = -(s * mid
                 - s * softplus(mid - x_a, k)
                 - s * softplus(mid - x_b, k))

    return f_raw - flat_val