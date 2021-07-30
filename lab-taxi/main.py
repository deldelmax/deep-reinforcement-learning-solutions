from agent import Agent
from monitor import interact
import gym
import numpy as np
np.random.seed(22)

env = gym.make('Taxi-v2')

agents = {}
agents["agent_sarsa_expected"] = Agent(nA=env.action_space.n, sarsa_type="expected", eps_constant_val=None, alpha=0.25, gamma=0.99)
agents["agent_sarsa_zero"] = Agent(nA=env.action_space.n, sarsa_type="zero", eps_constant_val=None, alpha=0.25, gamma=0.99)
agents["agent_sarsa_max"] = Agent(nA=env.action_space.n, sarsa_type="max", eps_constant_val=None, alpha=0.25, gamma=0.99)

scores = {}
for agent_name, agent in agents.items():
    print(agent_name)
    bas = []
    for i in range(5):
        avg_rewards, best_avg_reward = interact(env, agent)
        bas.append(best_avg_reward)
    scores[agent_name] = sum(bas) / len(bas)

print(scores)