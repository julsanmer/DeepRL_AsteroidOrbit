import numpy as np
from scipy.integrate import solve_ivp

from src.propagators import asteroid_dynamics


def integrate3D_scipy(x0,
                      tf,
                      asteroid_dict,
                      c=[1,1,1],
                      N=100,
                      events=None):
    # Time grid
    T_eval = np.linspace(0, tf, N+1)

    # Call solve_ivp
    sol = solve_ivp(fun=lambda t, x: asteroid_dynamics(t, x, asteroid_dict, c=c),
                    t_span=(0, tf),
                    y0=x0,
                    t_eval=T_eval,
                    method='DOP853',
                    dense_output=True,
                    events=events)

    # Generate N+1 evenly spaced points up to event time
    T = sol.t.T
    X = sol.y.T

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

    return T, X, event_flags, event_time, event_state
