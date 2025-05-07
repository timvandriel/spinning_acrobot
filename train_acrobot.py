# train_acrobot.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Register the custom environment if you haven't already
from spinning_acrobot_env import SpinningAcrobotEnv

gym.register(
    id="SpinningAcrobot-v0",
    entry_point="spinning_acrobot_env:SpinningAcrobotEnv",
    max_episode_steps=500,
)

# Create the environment
env = make_vec_env("SpinningAcrobot-v0", n_envs=4)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=200_000)

# Save the model
model.save("ppo_spinning_acrobot")

# Optionally, evaluate the trained agent
env = gym.make("SpinningAcrobot-v0", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()
