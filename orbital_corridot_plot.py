import numpy as np
import matplotlib.pyplot as plt
import casadi as ca


def softplus(z, k=50.0):
    """Smooth approximation of ReLU."""
    return ca.log1p(ca.exp(k*z)) / k


def slope_flat_slope(r, a, b, s=1.0, k=50.0, opposite=False):
    """
    Smooth function with slope -> flat -> slope.
    Always non-negative (minimum = 0).

    x        : CasADi MX or SX variable
    a, b     : start and end of the flat region
    s        : slope magnitude
    k        : sharpness of softplus transitions
    opposite : if True, second slope is negative
    """
    x = 2*(r-a)/(b-a)-1
    x_a = -1
    x_b = 1

    f_raw = -(s*x - s*softplus(x-x_a, k) - s*softplus(x-x_b, k))

    # Compute flat value: function at the center of the flat region
    flat_val = -(s*(x_a+x_b)/2 - s*softplus((x_a+x_b)/2 - x_a, k) - s*softplus((x_a+x_b)/2 - x_b, k))

    return f_raw - flat_val  # shift so minimum = 0

# Kilometers to meters
km2m = 1e3
m2km = 1e-3

# Normalization radius
r_ad = 16 * km2m

# Band
rmin = 22*km2m
rmax = 30*km2m

# Params
a, b, s = rmin/r_ad, rmax/r_ad, 1.0
x_vals = np.linspace(10, 40, 500) * km2m / r_ad

# Smooth curves
ks = [1, 10, 20]
f_vals = [slope_flat_slope(x_vals, a, b, s, k) for k in ks]

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
fontsize = 14
plt.figure(figsize=(6,5))
for f, k in zip(f_vals, ks):
    plt.plot(x_vals*r_ad*m2km, f, label=fr"$\beta = {k}$", linewidth=2)
plt.axvline(x=rmin*m2km, color='k', linestyle='--', linewidth=1.5)
plt.axvline(x=rmax*m2km, color='k', linestyle='--', linewidth=1.5)

plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.xlim(np.min(x_vals)*r_ad*m2km,
         np.max(x_vals)*r_ad*m2km)
plt.xlabel("r [km]", fontsize=fontsize)
plt.ylabel("L [-]", fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.grid(True)
plt.savefig("images/orbitalcorridor.eps", format="eps", dpi=300, bbox_inches="tight")
plt.show()