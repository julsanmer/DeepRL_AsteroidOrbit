import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.rl import (OrbitEnv, ReplayBufferIgnoreTrunc)


# Conversion factors
km2m = 1e3
m2km = 1e-3
h2s = 3600

# -------------------------------
# Main training
# -------------------------------
if __name__ == "__main__":
    # Load results
    with open("results/test_data.pkl", "rb") as f:
        data_dict = pickle.load(f)

    # Asteroid physical data
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
    shell_dict = {'bounds': (22 * km2m,
                             30 * km2m),
                  'gam': 0.1}
    optim_dict = {'tf': 10 * h2s,
                  'N': 60,
                  'dvmax': 0.2*np.ones(3),
                  'rmax': 50*km2m,
                  'ellip_axes': asteroid_dict['axes'],
                  'c': c,
                  'shell': shell_dict,
                  'H': -5.0}

    # Create environment
    orbitEnv = OrbitEnv(asteroid_dict, optim_dict)
    seed = 9
    log_path = "results/logs/seed" + str(seed)

    # Wrap environment
    env = Monitor(orbitEnv, filename=log_path)
    check_env(env)

    # Train SAC with truncation-aware Q-targets
    max_episodes = 5000
    model = SAC(
        "MlpPolicy",
        env,
        replay_buffer_class=ReplayBufferIgnoreTrunc,
        verbose=1,
        learning_rate=0.0003,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        seed=seed
    )

    # TRAIN
    total_steps = optim_dict['N']*max_episodes
    model.learn(total_timesteps=total_steps)

    # Save model
    model.save("results/sac_model" + str(seed))

    # Load monitor logs
    monitor_df = pd.read_csv(log_path + '.monitor.csv', skiprows=1)

    plt.plot(monitor_df["r"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()