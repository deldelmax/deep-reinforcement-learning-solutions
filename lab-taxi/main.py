from agent import Agent
from monitor import interact
import gym
import numpy as np
np.random.seed(22)

env = gym.make('Taxi-v3')
agents = {}
agents["agent_sarsa_expected"] = Agent(nA=env.action_space.n, sarsa_type="expected")
agents["agent_sarsa_max"] = Agent(nA=env.action_space.n, sarsa_type="max")
agents["agent_sarsa_zero"] = Agent(nA=env.action_space.n, sarsa_type="zero")
for agent_name, agent in agents.items():
    print(agent_name)
    avg_rewards, best_avg_reward = interact(env, agent)
    print(f"{agent_name=} {best_avg_reward=}")
