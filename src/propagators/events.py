import numpy as np


def make_ellipsoid_event(axes):
    def ellipsoid_event(t, x):
        return ((x[0]/axes[0])**2 + (x[1]/axes[1])**2
                + (x[2]/axes[2])**2) - 1.0
    # Stop integration / only outside→inside
    ellipsoid_event.terminal = True
    ellipsoid_event.direction = -1

    return ellipsoid_event

def make_rmax_event(r_max):
    def rmax_event(t, x):
        r = np.linalg.norm(x[:3])
        return r - r_max

    # Stop integration / only outside→inside
    rmax_event.terminal = True
    rmax_event.direction = 1

    return rmax_event
