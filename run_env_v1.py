import os

# This is a workaround for the KMP_DUPLICATE_LIB_OK error
# It allows the program to run without crashing due to duplicate libraries.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from stable_baselines3 import PPO

# create the environment
env = gym.make("Acrobot-v1", render_mode="human")

# Load your trained model
model = PPO.load("ppo_acrobot")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()
