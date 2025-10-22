import numpy as np
import time

from scipy.integrate import solve_ivp

from .dynamics import asteroid_dynamics


def integrate3D_dV(x0,
                   asteroid_dict,
                   dv_dict,
                   c=[1,1,1],
                   N=1,
                   events=None):
    t_dV = dv_dict['T']
    if 'dv_actor' in dv_dict:
        dv_actor = dv_dict['dv_actor']
        dvmax = dv_dict['dvmax']
        cum_dv = 0.
        dV = np.zeros((len(t_dV)-1, 3))
    else:
        # Unfold dV
        dV = dv_dict['dV']

    # Output T and X
    t = np.array([t_dV[0]])
    X = x0[np.newaxis,:]
    t_wall = np.full(len(t_dV)-1, np.nan)

    # Loop through
    for i in range(len(t_dV)-1):
        # Interval duration
        t0 = t_dV[i]
        tf = t_dV[i+1]

        # Time grid
        T_eval = np.linspace(t0, tf, N+1)

        # Obtain dv for actor
        if 'dv_actor' in dv_dict:
            # Concatenate into a 7D array
            t0_wall = time.perf_counter()
            obs = np.concatenate([x0[0:3]/c[1],
                                  x0[3:6]/c[2],
                                  [cum_dv]])
            action, _ = dv_actor.predict(obs, deterministic=True)
            dV[i] = dvmax * np.clip(action, -1, 1)
            cum_dv += np.sum(np.abs(np.clip(action, -1, 1))) / (len(t_dV)-1)
            t1_wall = time.perf_counter()
            t_wall[i] = t1_wall - t0_wall

        # Apply dv
        x0[3:6] += dV[i]

        # Call solve_ivp
        sol = solve_ivp(fun=lambda t, x: asteroid_dynamics(t, x, asteroid_dict),
                        t_span=(t0, tf),
                        y0=x0,
                        t_eval=T_eval,
                        method='DOP853',
                        dense_output=True,
                        events=events)

        # Check events
        event_flags = [False] * (len(events) if events is not None else 0)
        event_time = None
        event_state = None
        if sol.t_events is not None:
            for i, t_ev in enumerate(sol.t_events):
                if len(t_ev) > 0:
                    # First trigger of event i
                    event_time = t_ev[0]
                    event_state = sol.y_events[i][0]
                    event_flags[i] = True

        # Generate N+1 evenly spaced points up to event time
        Ti = sol.t.T
        Xi = sol.y.T

        # Update x0
        x0 = Xi[-1,:]

        # Stack it below
        t = np.hstack([t, Ti[1:]])
        X = np.vstack([X, Xi[1:]])

    # Events outputs
    event_out = (event_flags, event_time, event_state)

    return np.squeeze(t), X, dV, event_out, t_wall
