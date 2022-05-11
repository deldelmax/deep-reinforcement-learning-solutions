import numpy as np
from collections import defaultdict
import math

class Agent:
    def __init__(self, nA, sarsa_type="max", alpha=0.01, gamma=1.):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - sarsa_type: type of TD algorithm to use; one of "max" (Q-learning), "zero", or "expected"
        - eps_constant_val: constant 'eps' value to use; otherwise we use a schedule for eps
        - alpha: learning rate
        - gamma: return discount factor
        """        
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.sarsa_type = sarsa_type
        self.eps = 0.005
        self.alpha = alpha
        self.gamma = gamma
        np.random.seed(22)

    def _get_probs_eps_greedy(self, state):
        actions_probs = np.ones(self.nA) * (self.eps / self.nA)
        greedy_action = self.Q[state].argmax()
        actions_probs[greedy_action] += 1 - self.eps
        return actions_probs

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return int(np.random.choice(np.arange(self.nA),  p=self._get_probs_eps_greedy(state)))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            match self.sarsa_type:
                case "zero":
                    next_estimated_return = self.Q[next_state][self.select_action(next_state)]
                case "max":
                    next_estimated_return = self.Q[next_state].max()
                case "expected":
                    next_estimated_return = np.dot(self.Q[next_state], self._get_probs_eps_greedy(next_state)) 
                case _:
                    raise ValueError
        else:
            next_estimated_return = 0
        target = reward + self.gamma * next_estimated_return 
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
        # self.eps = max(self.eps * self.eps_decay, self.eps_min)


         