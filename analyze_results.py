import numpy as np
import pickle
import matplotlib.pyplot as plt

from plots import (plot_deltav,
                   plot_radius,
                   plot_reward,
                   plot_statistics,
                   plot_orbits,
                   plot_dvrtn)
from src.utils.ellipsoidShape import Ellipsoid
from plots.dv_rtn import cartesian_to_rtn


# Load results
with open("results/SAC_results.pkl", "rb") as f:  # "rb" = read binary
    SAC_dict = pickle.load(f)

# Load results
with open("results/direct_results.pkl", "rb") as f:  # "rb" = read binary
    OCP_dict = pickle.load(f)

dv_rtn = cartesian_to_rtn(OCP_dict['results_dict'][50]['dV'][0:60],
                          OCP_dict['results_dict'][50]['X'][0:60,0:3],
                          OCP_dict['results_dict'][50]['X'][0:60,3:6],
                          OCP_dict['initial_dict']['asteroid']['omega'])


# Create Eros shape
erosShape = Ellipsoid(axes=OCP_dict['initial_dict']['asteroid']['axes'])

# Plots
plot_orbits(SAC_dict, OCP_dict, erosShape)
plot_statistics(SAC_dict, OCP_dict)
plot_radius(SAC_dict, OCP_dict, 50)
plot_deltav(SAC_dict, OCP_dict, 50)
plot_dvrtn(SAC_dict, OCP_dict)
plot_reward()
plt.show()

# Computation times
twall_OCP = [d['dt_wall'] for d in OCP_dict['results_dict']]
print('SAC wall time: Mean:' + str(np.mean(SAC_dict['optim_dict']['T_wall'])*60)
      + ' Max: ' + str(np.max(SAC_dict['optim_dict']['T_wall'])*60))
print('OCP wall time: Mean:' + str(np.mean(twall_OCP))
      + ' Max: ' + str(np.max(twall_OCP)))
