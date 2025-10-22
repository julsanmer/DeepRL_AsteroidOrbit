import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ---------- Parameters ----------
G = 1.0           # gravitational constant (use 1 for normalized units)
m1 = 1.0          # mass 1
m2 = 0.3          # mass 2
D = 1.0           # separation between the two masses
omega = 1.0       # rotation rate of the rotating frame

# Derived: positions along x with center of mass at origin
x1 = -m2/(m1 + m2) * D
x2 =  m1/(m1 + m2) * D

# ---------- Functions ----------
def Omega(x, y):
    """Effective potential Omega(x,y) = 0.5*omega^2*r^2 + G m1/r1 + G m2/r2"""
    r1 = np.hypot(x - x1, y)
    r2 = np.hypot(x - x2, y)
    # avoid division by zero (clip very small r)
    r1 = np.maximum(r1, 1e-12)
    r2 = np.maximum(r2, 1e-12)
    return 0.5 * omega**2 * (x**2 + y**2) + G*m1 / r1 + G*m2 / r2

def dOmegadx_on_xaxis(x):
    """dOmega/dx evaluated at y=0 (for root-finding)"""
    r1 = np.abs(x - x1)
    r2 = np.abs(x - x2)
    r1 = max(r1, 1e-12)
    r2 = max(r2, 1e-12)
    # derivative: omega^2 * x - G*m1*(x-x1)/r1^3 - G*m2*(x-x2)/r2^3
    return omega**2 * x - G*m1 * (x - x1) / (r1**3) - G*m2 * (x - x2) / (r2**3)

# ---------- find collinear equilibrium points (L1,L2,L3) ----------
# We'll search in three intervals: left of both masses, between them, right of both.
search_left  = (-5*D, min(x1, x2) - 1e-6)
search_mid   = (min(x1, x2) + 1e-6, max(x1, x2) - 1e-6)
search_right = (max(x1, x2) + 1e-6, 5*D)

roots = []
for a, b in (search_left, search_mid, search_right):
    try:
        fa = dOmegadx_on_xaxis(a)
        fb = dOmegadx_on_xaxis(b)
        # only try brentq if sign change — otherwise try small bracket scan
        if fa * fb > 0:
            # scan for any sign change in many subintervals
            xs = np.linspace(a, b, 400)
            fs = [dOmegadx_on_xaxis(xx) for xx in xs]
            sign_changes = []
            for i in range(len(xs)-1):
                if fs[i] * fs[i+1] < 0:
                    sign_changes.append((xs[i], xs[i+1]))
            if sign_changes:
                ra, rb = sign_changes[0]
                r = brentq(dOmegadx_on_xaxis, ra, rb)
                roots.append(r)
        else:
            r = brentq(dOmegadx_on_xaxis, a, b)
            roots.append(r)
    except Exception:
        # ignore intervals where root finding fails
        pass

# Sort and label L3, L1, L2 (convention: L3 leftmost, then L1 between masses, L2 rightmost)
roots = sorted(roots)
# Ensure we have upto three roots; otherwise just use what we found
L_points = roots
# Compute Omega and Jacobi C = 2 * Omega at those points
L_omegas = [Omega(x, 0.0) for x in L_points]
L_Cs = [2.0 * Om for Om in L_omegas]

print("Mass positions: x1 =", x1, "x2 =", x2)
print("Found collinear equilibrium x positions:", L_points)
print("Corresponding Jacobi C values (C = 2 Omega):", L_Cs)

# ---------- grid and Omega ----------
nx, ny = 800, 600
x_min = -2.5*D
x_max =  2.5*D
y_min = -1.5*D
y_max =  1.5*D
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)
Omega_grid = Omega(X, Y)

# Choose contour levels:
# include the Cs at the L-points (if found) and some other chosen values
levels = []
if len(L_Cs) > 0:
    # sort decreasing (higher C = more restricted)
    L_Cs_sorted = sorted(L_Cs, reverse=True)
    levels.extend(L_Cs_sorted)
# add some more levels spanning Omega
Omega_min = np.nanmin(Omega_grid)
Omega_max = np.nanmax(Omega_grid)
extra_levels = np.linspace(2*Omega_min + 0.1*(2*(Omega_max-Omega_min)),
                           2*Omega_max - 0.1*(2*(Omega_max-Omega_min)), 6)
# convert extra levels from "C values" to Omega-levels by dividing by 2 for plotting Omega= C/2:
# but we're plotting contours of Omega directly, so levels should be Omega values:
Omega_levels = list(extra_levels / 2.0)
# combine and deduplicate
for L in Omega_levels:
    if L not in levels:
        levels.append(L)
levels = sorted(levels)

# ---------- plotting ----------
fig, ax = plt.subplots(figsize=(10,6))
# filled contour showing potential background (optional)
cf = ax.contourf(X, Y, Omega_grid, levels=60, cmap='viridis', alpha=0.6)

# plot zero-velocity curves for selected Jacobi constants -> Omega = C/2
# (levels contains Omega values already)
cs = ax.contour(X, Y, Omega_grid, levels=levels, colors='k', linewidths=1.0)
ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

# mark the two masses
ax.plot(x1, 0, 'ro', label=f'm1 (x={x1:.3f})', markersize=8)
ax.plot(x2, 0, 'mo', label=f'm2 (x={x2:.3f})', markersize=8)

# mark L points
for i, xL in enumerate(L_points):
    ax.plot(xL, 0, 'kx')
    ax.text(xL, 0.02*D, f'L{i+1} ({xL:.3f})', ha='center', color='k', fontsize=9)

ax.set_aspect('equal', 'box')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Zero-velocity curves (contours of Ω = C/2). Dark contours = ZVCs.')
ax.legend(loc='upper right')
plt.colorbar(cf, ax=ax, label='Ω (effective potential)')
plt.show()