import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.propagators.events import (make_ellipsoid_event, make_rmax_event)
from src.propagators.integrator_scipy import integrate3D_scipy
from src.utils.ellipsoidShape import Ellipsoid
from src.utils import oe2cartesian

# Conversion factors
km2m = 1e3
m2km = 1e-3

# Asteroid physical parameters
asteroid_dict = {
    'mascon': {
        'muM': np.array([0.6, 0.4]) * 4.46275472004 * 1e5,
        'xyzM': np.array([[(0.4/0.6)*8, 0, 0],
                          [-8, 0, 0]]) * km2m
    },
    'xyzM': np.array([[(0.4/0.6)*8, 0, 0],
                      [-8, 0, 0]]) * km2m,
    'omega': np.array([0, 0, 2*np.pi / (5.27*3600)]),
    'axes': np.array([16, 8, 5]) * km2m
}

# Ellipsoid shape
erosShape =Ellipsoid(asteroid_dict['axes'])

# Adimensional constants
r_ad, v_ad, t_ad = 1, 1, 1
c = (t_ad, r_ad, v_ad)

# Base orbit elements
oe = np.array([np.nan,           # semi-major axis
               0.0,              # eccentricity
               np.nan,           # inclination (to be randomized)
               np.nan,           # RAAN
               np.radians(0),    # argument of periapsis
               np.nan])          # true anomaly (to be randomized)

# Number of samples
N = 10000

# Define ranges (in radians)
a_min, a_max = 18*km2m, 28*km2m
inc_min, inc_max = 0, np.pi      # 0° to 180° inclination
RAAN_min, RAAN_max = 0, 2*np.pi  # 0° to 360° RAAN
nu_min, nu_max = 0, 2*np.pi      # 0° to 360° true anomaly

# Random sampling
np.random.seed(0)
oe_samples = []
for _ in range(N):
    oe_rand = oe.copy()
    oe_rand[0] = np.random.uniform(a_min, a_max)
    oe_rand[2] = np.random.uniform(inc_min, inc_max)
    oe_rand[3] = np.random.uniform(RAAN_min, RAAN_max)
    oe_rand[5] = np.random.uniform(nu_min, nu_max)
    oe_samples.append(oe_rand)
oe_samples = np.array(oe_samples)

# Place event
event_collision = make_ellipsoid_event(erosShape.axes/r_ad)
event_escape = make_rmax_event(r_max=50*km2m/r_ad)
events = [event_collision,
          event_escape]

# Place list to save coordinates
# and events
T_list = []
X_list = []
ellipsoid_flags = []
rmax_flags = []

# Loop through samples
for i in range(N):
    # Obtain sample
    oe = oe_samples[i]

    # Transform to Cartesian in inertial
    pos_N, vel_N = oe2cartesian(oe, np.sum(asteroid_dict['mascon']['muM']))

    # Transform to planetocentric
    pos_P = pos_N
    vel_P = vel_N - np.cross(asteroid_dict['omega'],
                             pos_P)
    x0 = np.hstack((pos_P, vel_P))
    tf = 10*3600

    # Adimensionalize
    x0_ad = np.hstack((x0[0:3]/r_ad,
                       x0[3:6]/v_ad))
    tf_ad = tf / t_ad

    # Integrate
    T, X, event_flags, te, xe = integrate3D_scipy(
        x0_ad.tolist(),
        tf_ad,
        mascon_dict=asteroid_dict['mascon'],
        omega=asteroid_dict['omega'].tolist(),
        c=c,
        N=1000,
        events=events
    )

    # Append to lists
    T_list.append(T)
    X_list.append(X)
    ellipsoid_flags.append(event_flags[0])   # first event (collision)
    rmax_flags.append(event_flags[1])        # second event (r > rmax)
    print(i)

# Dict of results
results_dict = {'asteroid': asteroid_dict,
                'oe0': oe_samples,
                'T': T_list,
                'X': X_list,
                'collision': ellipsoid_flags,
                'escape': rmax_flags}

# Save to pickle file
with open("results/test_data.pkl", "wb") as f:   # "wb" = write binary
    pickle.dump(results_dict, f)

a_list = []
inc_list = []

# Extract orbital elements back from initial samples
for oe in oe_samples:
    a = oe[0]                   # assuming oe = [a, e, i, Ω, ω, M]
    inc = np.degrees(oe[2])     # inclination in deg
    a_list.append(a)
    inc_list.append(inc)

# Count outcomes
n_ellipsoid = sum(ellipsoid_flags)
n_rmax = sum(rmax_flags)
n_stable = len(ellipsoid_flags) - n_ellipsoid - n_rmax

print(f"Ellipsoid collisions: {n_ellipsoid}")
print(f"r > r_max cases: {n_rmax}")
print(f"Stable cases: {n_stable}")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
fontsize=14
plt.figure(figsize=(6,5))
for a, inc, hit_ellipsoid, hit_rmax in zip(a_list, inc_list, ellipsoid_flags, rmax_flags):
    if hit_ellipsoid:
        plt.scatter(a*m2km, inc, color='red', s=14,
                    label="Collision" if "Collision" not in plt.gca().get_legend_handles_labels()[1] else "")
    elif hit_rmax:
        plt.scatter(a*m2km, inc, color='green', s=14,
                    label="$r > r_{\\mathrm{max}}$" if "$r > r_{\\mathrm{max}}$" not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(a*m2km, inc, color='blue', s=14,
                    label="Stable" if "Stable" not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xlabel("Initial semi-major axis [km]", fontsize=fontsize)
plt.ylabel("Initial inclination [$^{\circ}$]", fontsize=fontsize)
#plt.title("10 hours propagation", fontsize=fontsize)
plt.legend(fontsize=fontsize, loc='upper right')
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.grid(True)
plt.savefig("images/orbit_stab1.eps", format="eps", dpi=300, bbox_inches="tight")

# # Plot 2: trajectories (1x2)
# fig, axes = plt.subplots(1, 2, figsize=(12,6), subplot_kw={'projection':'3d'})
#
# # Left: no collision
# for X, collided in zip(X_list, terminal_list):
#     if not collided:
#         axes[0].plot(X[:,0]*r_ad*m2km, X[:,1]*r_ad*m2km, X[:,2]*r_ad*m2km, 'b-', alpha=0.5)
# erosShape.plot3D(axes[0], scale=m2km)
# axes[0].set_title("Natural trajectories (no collision)")
#
# # Right: collision
# for X, collided in zip(X_list, terminal_list):
#     if collided:
#         axes[1].plot(X[:,0]*r_ad*m2km, X[:,1]*r_ad*m2km, X[:,2]*r_ad*m2km, 'r-', alpha=0.5)
# erosShape.plot3D(axes[1], scale=m2km)
# axes[1].set_title("Collided trajectories")
#
# for ax in axes:
#     ax.set_xlabel("x [km]")
#     ax.set_ylabel("y [km]")
#     ax.set_zlabel("z [km]")
#     ax.grid(True)
#
# plt.tight_layout()
plt.show()
