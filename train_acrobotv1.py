# train_acrobot.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# Create the environment
env = make_vec_env("Acrobot-v1", n_envs=4)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=400_000)

# Save the model
model.save("ppo_acrobot")

# Optionally, evaluate the trained agent
env = gym.make("Acrobot-v1", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

env.close()
