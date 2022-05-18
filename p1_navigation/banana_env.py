import gym
from gym import spaces
import numpy as np
from unityagents.environment import UnityEnvironment

class BananaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, unity_env_path):
        self._unity_env = UnityEnvironment(file_name=unity_env_path)
        self._brain_name = self._unity_env.brain_names[0]
        self._brain = self._unity_env.brains[self._brain_name]

        self.nA = self._brain.vector_action_space_size
        self.nS = self._brain.vector_observation_space_size

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, (self.nS,))

    def _get_info(self):
        return {}

    def _parse_response(self, env_info):
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            return next_state, reward, done

    def reset(self, seed=None, return_info=False, options={"train_mode": True}):
        # We need the following line to seed self.np_random
        #super().reset()
        env_info = self._unity_env.reset(train_mode=options["train_mode"])[self._brain_name]                
        observation = env_info.vector_observations[0] 
        return (observation, self._get_info()) if return_info else observation

    def step(self, action):
        env_info = self._unity_env.step(action)[self._brain_name] 
        observation, reward, done = self._parse_response(env_info)
        return observation, reward, done, self._get_info()

    def close(self):
        self._unity_env.close()