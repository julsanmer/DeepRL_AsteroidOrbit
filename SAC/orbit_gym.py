import gymnasium
import numpy as np
from gymnasium import spaces

class OrbitSurvivalEnv(gymnasium.Env,
                       mascon_dict,
                       omega,
                       c):
    def __init__(self):
        super().__init__()
        self.dt = 0.1  # Time step
        self.GM = 1.0  # Gravitational constant * asteroid mass
        self.radius_min = 1.0   # Crash zone
        self.radius_max = 5.0   # Escape zone
        self.thrust_mag = 0.05  # Max thrust magnitude

        # State: [x, y, vx, vy]
        obs_high = np.array([10, 10, 5, 5], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action: 2D thrust vector (clipped)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        if seed is not None:
            np.random.seed(seed)

        self.state = np.array([2.0, 0.0, 0.0, 0.7], dtype=np.float32)

        obs = self.state.astype(np.float32)
        info = {}  # You can add useful stuff here if needed

        return obs, info

    def step(self, action):
        x, y, vx, vy = self.state
        r = np.sqrt(x ** 2 + y ** 2)

        # Gravity
        ax_g = -self.GM * x / r ** 3
        ay_g = -self.GM * y / r ** 3

        # Thrust
        ax_t, ay_t = self.thrust_mag * np.clip(action, -1, 1)

        # Total acceleration
        ax = ax_g + ax_t
        ay = ay_g + ay_t

        # Integrate
        vx += ax * self.dt
        vy += ay * self.dt
        x += vx * self.dt
        y += vy * self.dt

        self.state = np.array([x, y, vx, vy])
        self.t += 1

        # Check for termination
        terminated = False  # agent failed (crashed or escaped)
        truncated = False  # time limit or artificial cutoff

        reward = 1.0  # reward for surviving

        if r < self.radius_min:
            terminated = True
            reward = -100.0  # crash
        elif r > self.radius_max:
            terminated = True
            reward = -50.0  # escape
        elif self.t >= 1000:  # optional episode cap
            truncated = True

        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"t={self.t}, state={self.state}")