import numpy as np
import gymnasium
from gymnasium import spaces

from src.propagators.integrator_scipy import integrate3D_scipy
from src.rl.orbitshell_penalty import shell_penalty
from src.utils import oe2cartesian


# Conversion factors
km2m = 1e3
m2km = 1e-3


class OrbitEnv(gymnasium.Env):
    def __init__(self, asteroid_dict, optim_dict):
        super().__init__()

        # Asteroid properties
        self.asteroid_dict = asteroid_dict

        # Optimization variables
        self.optim_dict = optim_dict

        # State: [pos, vel, cum_dv]
        obs_high = np.array([10, 10, 10, 10, 10, 10, 100],
                            dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high,
                                            obs_high,
                                            dtype=np.float32)

        # Action: 3D delta-v (clipped)
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(3,),
                                       dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize episode step
        self.k = 0

        # Adimensional factors
        r_ad = self.optim_dict['c'][1]
        v_ad = self.optim_dict['c'][2]

        if seed is not None:
            np.random.seed(seed)

        # Orbital elements
        oe = self._sample_oe()

        # From orbit element to cartesian
        mu = np.sum(self.asteroid_dict['mascon']['muM'])
        pos_N, vel_N = oe2cartesian(oe, mu)

        # Transform to planetocentric
        pos_P = pos_N
        vel_P = vel_N - np.cross(self.asteroid_dict['omega'],
                                 pos_P)
        self.x = np.hstack((pos_P, vel_P))
        self.x = np.append(self.x,
                           0.)

        # Adimensionalize state
        self.x[0:3] /= r_ad
        self.x[3:6] /= v_ad

        # Output observation
        obs = self.x.astype(np.float32)
        info = {}

        return obs, info

    def step(self, action):
        # Initial state for step
        xk = self.x[0:6]

        # Retrieve auxiliary variables
        t_ad, r_ad, v_ad = self.optim_dict['c']
        N = self.optim_dict['N']
        tf_ad = self.optim_dict['tf'] / t_ad
        dvmax_ad = self.optim_dict['dvmax'] / v_ad
        shell = self.optim_dict['shell']

        # Step duration
        dt = tf_ad / N

        # Compute delta-v
        action = np.clip(action, -1, 1)
        dv = dvmax_ad * action
        xk[3:6] += dv

        # Compute dynamics step
        T, X, _, _, _ = integrate3D_scipy(xk,
                                          dt,
                                          self.asteroid_dict,
                                          c=self.optim_dict['c'],
                                          N=1)

        # Advance episode step
        self.k += 1

        # Update state
        self.x[0:6] = X[1]
        self.x[6] += np.sum(np.abs(action)) / self.optim_dict['N']

        # Compute orbital radius
        rk = np.linalg.norm(self.x[:3])

        # Compute orbit shell penalty
        reward = -shell_penalty(rk,
                                shell['bounds'][0]/r_ad,
                                shell['bounds'][1]/r_ad)
        reward *= shell['gam']/N

        # Add impulse penalty
        reward -= np.sum(np.abs(action))/N
        #reward -= (dv[0]**2 + dv[1]**2 + dv[2]**2) / (self.dvmax**2)
        #reward = -np.log(np.sum(np.abs(np.clip(action, -1, 1)))+1e-8)

        # Collision or escape
        has_collided = self._check_collision()
        has_escaped = self._check_escape()

        # Check for termination
        terminated = False
        truncated = False

        # Check collision or escape
        if has_collided or has_escaped:
            terminated = True
            reward = self.optim_dict['H']

        # If max intervals are reached truncate
        if self.k >= self.optim_dict['N']:
            truncated = True

        return self.x.astype(np.float32), reward, terminated, truncated, {}

    def _check_collision(self):
        # Collision flag
        has_collided = False

        # Ellipsoid axes
        r_ad = self.optim_dict['c'][1]
        axes = self.optim_dict['ellip_axes']
        a, b, c = axes / r_ad

        # Get position and check if it is within
        # ellipsoid
        x, y, z = self.x[0:3]
        if (x/a)**2 + (y/b)**2 + (z/c)**2 < 1:
            has_collided = True

        return has_collided

    def _check_escape(self):
        # Escape flag
        has_escaped = False

        # Check if r is beyond max radius
        r_ad = self.optim_dict['c'][1]
        r = np.linalg.norm(self.x[:3])
        if r > self.optim_dict['rmax']/r_ad:
            has_escaped = True

        return has_escaped

    def _sample_oe(self):
        # Base orbit elements
        oe = np.array([np.nan,         # semi-major axis
                       0.0,            # eccentricity
                       np.nan,         # inclination (to be randomized)
                       np.nan,         # RAAN
                       np.radians(0),  # argument of periapsis
                       np.nan])        # true anomaly (to be randomized)

        # Define ranges (in radians)
        a_min, a_max = 18 * km2m, 28 * km2m
        inc_min, inc_max = 0, np.pi        # 0° to 180° inclination
        RAAN_min, RAAN_max = 0, 2*np.pi    # 0° to 360° RAAN
        nu_min, nu_max = 0, 2*np.pi        # 0° to 360° true anomaly

        # Sample orbit element
        oe[0] = np.random.uniform(a_min, a_max)
        oe[2] = np.random.uniform(inc_min, inc_max)
        oe[3] = np.random.uniform(RAAN_min, RAAN_max)
        oe[5] = np.random.uniform(nu_min, nu_max)

        return oe
