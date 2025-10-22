import numpy as np
import multiprocessing as mp
import pickle
import time
from tqdm import tqdm

from src.optimizers import (optimize_safeorbit, integrator3D_casadi)
from src.propagators import (integrate3D_dV, make_ellipsoid_event, make_rmax_event)
from src.utils import oe2cartesian


km2m = 1e3
m2km = 1e-3
h2sec = 3600
sec2h = 1 / 3600


# -------------------------------
# Worker function for one episode
# -------------------------------
def run_episode(i, asteroid_dict, data_dict, optim_dict):
    # Retrieve adimensional variables
    t_ad, r_ad, v_ad = optim_dict['c']

    # Get asteroid physical parameters
    mu = np.sum(asteroid_dict['mascon']['muM'])
    omega = asteroid_dict['omega']

    # Get initial orbital element
    oe0 = data_dict['oe0'][i]

    # Transform to cartesian in inertial frame
    pos0_N, vel0_N = oe2cartesian(oe0, mu)

    # Transform to planetocentric frame
    pos0_P = pos0_N
    vel0_P = vel0_N - np.cross(omega, pos0_P)
    x0 = np.hstack((pos0_P, vel0_P))
    optim_dict['x0'] = x0

    # Extract variables
    tf = optim_dict['tf']
    N = optim_dict['N']

    # Delta-v with zeros for natural dynamics
    dV_dict = {
        'T': np.linspace(0, tf, N+1),
        'dV': np.zeros((N+1, 3))
    }

    # Initial guess with natural dynamics (dimensional)
    T0, X0, dV0, _ = integrate3D_dV(x0,
                                    asteroid_dict,
                                    dV_dict)

    # Initial guess for optimizer
    initguess_dict = {
        'T': T0,
        'X': X0,
        'dV': dV0
    }

    # Create integrator inside worker
    intg = integrator3D_casadi(asteroid_dict,
                               c=optim_dict['c'])

    # Optimize safe orbit
    t0_wall = time.perf_counter()
    results_dict = optimize_safeorbit(initguess_dict,
                                      optim_dict,
                                      intg)
    t1_wall = time.perf_counter()
    dV_dict['T'] = results_dict['T']
    dV_dict['dV'] = results_dict['dV']

    # Final integration with dV
    # Place event
    event_collision = make_ellipsoid_event(optim_dict['ellip_axes'])
    event_escape = make_rmax_event(r_max=optim_dict['rmax'])
    events = [event_collision,
              event_escape]
    Tsim, Xsim, _, _ = integrate3D_dV(x0,
                                      asteroid_dict,
                                      dV_dict,
                                      N=60,
                                      events=events)
    results_dict['Tsim'] = Tsim
    results_dict['Xsim'] = Xsim
    results_dict['dt_wall'] = t1_wall - t0_wall

    return results_dict


# -------------------------------
# Parallel execution with progress bar
# -------------------------------
if __name__ == "__main__":
    # Load results
    with open("results/test_data.pkl", "rb") as f:  # "rb" = read binary
        data_dict = pickle.load(f)

    # Number of simulations
    N_episodes = 500

    # Retrieve asteroid parameters
    asteroid_dict = data_dict['asteroid']

    # Obtain mu and rE
    mu = np.sum(asteroid_dict['mascon']['muM'])
    omega = asteroid_dict['omega']
    rE = asteroid_dict['axes'][0]

    # Adimensional constants
    r_ad = rE
    v_ad = np.sqrt(mu/r_ad)
    t_ad = r_ad / v_ad
    c = (t_ad, r_ad, v_ad)

    # Orbital corridor settings
    shell_dict = {'bounds': (22 * km2m/r_ad,
                             30 * km2m/r_ad),
                  'gam': 0.1}
    optim_dict = {'tf': 10 * h2sec,
                  'N': 60,
                  'dvmax': 0.2*np.ones(3),
                  'rmax': 50*km2m,
                  'ellip_axes': asteroid_dict['axes'],
                  'c': c,
                  'shell': shell_dict}

    # Parallelization
    tasks = [(i, asteroid_dict, data_dict, optim_dict)
             for i in range(N_episodes)]
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        results = list(tqdm(
            pool.starmap(run_episode, tasks),
            total=N_episodes,
            desc="Simulating episodes"
        ))

    # Output dictionary to be saved
    output_dict = {'initial_dict': data_dict,
                   'optim_dict': optim_dict,
                   'results_dict': results}

    # Save to pickle file
    with open("results/direct_results.pkl", "wb") as f:
        pickle.dump(output_dict, f)
