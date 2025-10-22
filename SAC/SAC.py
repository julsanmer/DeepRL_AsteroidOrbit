# import gymnasium as gym
#
# from stable_baselines3 import SAC
#
# env = gym.make("Pendulum-v1", render_mode="human")
#
# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("sac_pendulum")
#
# del model # remove to demonstrate saving and loading
#
# model = SAC.load("sac_pendulum")
#
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import TimeLimit

from orbit_gym import OrbitSurvivalEnv


log_path = "./logs/monitor.csv"

# Wrap in correct order: Monitor first, then TimeLimit
env = Monitor(OrbitSurvivalEnv(), filename=log_path)
env = TimeLimit(env, max_episode_steps=1000)
check_env(env)

# Train the SAC agent
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save the trained model
model.save("sac_orbit_survival")

# # Test the trained agent
# obs = env.reset()
# for _ in range(2):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, _ = env.step(action)
#     env.render()
#     if done:
#         print("Terminated:", reward)
#         break

import matplotlib.pyplot as plt

# Load monitor logs from file (default: "./monitor.csv")
import pandas as pd
monitor_df = pd.read_csv(log_path, skiprows=1)

plt.plot(monitor_df["r"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
