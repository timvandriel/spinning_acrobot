import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
from stable_baselines3 import PPO

# 🔁 Import your custom environment
from spinning_acrobot_env import SpinningAcrobotEnv

# 🧠 Register it again
gym.register(
    id="SpinningAcrobot-v0",
    entry_point="spinning_acrobot_env:SpinningAcrobotEnv",
    max_episode_steps=500,
)

# ✅ Now create the environment
env = gym.make("SpinningAcrobot-v0", render_mode="human")

# 🧠 Load your trained model
model = PPO.load("ppo_spinning_acrobot")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()
