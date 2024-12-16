from quadcopter.env.quadcopter import DroneControllerEnv
import os
from stable_baselines3 import PPO


env = DroneControllerEnv()

# model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=100000)


num_steps = 1500

obs = env.reset()

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(num_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

env.close()