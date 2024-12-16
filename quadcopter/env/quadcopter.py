from typing import Optional
import gymnasium as gym
import mujoco
import numpy as np

class DroneControllerEnv(gym.Env):

    def __init__(self, size: int = 5):
        self.m = mujoco.MjModel.from_xml_path('skydio_x2/scene.xml')
        self.d = mujoco.MjData(self.m)
        self.target = np.array((0,0,2))
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
    def _get_obs(self):
        return {
            "agent": self.d.qpos,
            "target": self.target,
        }
    
    def _get_info(self):
        return 
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        info = self._get_info()
        observation = self._get_obs()
        return  observation, info
    
    def step(self, action):
        self.d.ctrl[:4] = action
        #reward is the negative distance to the target
        reward = -np.linalg.norm(self.d.qpos[:2] - self.target[:2])
        observation = self._get_obs()
        terminated = False
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info