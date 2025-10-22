import numpy as np
import pickle
from stable_baselines3 import SAC

from src.propagators import (make_ellipsoid_event, make_rmax_event)
from src.propagators import integrate3D_dV

h2s = 3600
km2m = 1e3


# -------------------------------
# Soft-actor critic
# -------------------------------
if __name__ == "__main__":
    # Load results
    with open("results/test_data.pkl", "rb") as f:  # "rb" = read binary
        data_dict = pickle.load(f)

    # Retrieve asteroid parameters
    asteroid_dict = data_dict['asteroid']

    # Obtain mu and rE
    mu = np.sum(asteroid_dict['mascon']['muM'])
    omega = asteroid_dict['omega']
    rE = asteroid_dict['axes'][0]

    # Final time and number of intervals
    tf = 10 * h2s
    N = 60
    dt = tf / N

    # Maximum impulse and max radius
    dvmax = 0.2
    rmax = 50*km2m

    # Adimensional constants
    r_ad = rE
    v_ad = np.sqrt(mu/r_ad)
    t_ad = r_ad/v_ad
    c = (t_ad, r_ad, v_ad)

    # Reload the trained model
    model = SAC.load("results/sac_model0.zip")

    # Prepare dv dictionary
    t_dv = np.linspace(0, tf, N+1)
    dv_dict = {'T': t_dv,
               'dv_actor': model,
               'dvmax': dvmax}

    # Run a few episodes to see performance
    n_eval = 500

    # Initial conditions
    X0 = data_dict['X'][0:n_eval]

    # Lists
    T_list = []
    X_list = []
    dV_list = []
    status_list = []
    Twall_list = []

    # Place events
    event_collision = make_ellipsoid_event(asteroid_dict['axes'])
    event_escape = make_rmax_event(r_max=rmax)
    events = [event_collision,
              event_escape]

    # Lists
    T_list = []
    X_list = []
    dV_list = []

    # Loop through episodes
    for i in range(n_eval):
        # Obtain x0
        x0 = X0[i][0,:]

        # Final integration with dV
        Tsim, Xsim, dVsim, event, t_wall = integrate3D_dV(x0,
                                                          asteroid_dict,
                                                          dv_dict,
                                                          c=c,
                                                          N=100,
                                                          events=events)
        T_list.append(Tsim)
        X_list.append(Xsim)
        dV_list.append(dVsim)
        Twall_list.append(t_wall)
        if event[0][0] or event[0][1]:
            status_list.append('fail')
        else:
            status_list.append('success')
        print(i)


    # Dict of results
    optim_dict = {'T': T_list,
                  'X': X_list,
                  'dV': dV_list,
                  'status': status_list,
                  'T_wall': t_wall}
    output_dict = {'initial_dict': data_dict,
                   'optim_dict': optim_dict}

    # Save to pickle file
    with open("results/SAC_results.pkl", "wb") as f:
        pickle.dump(output_dict, f)
