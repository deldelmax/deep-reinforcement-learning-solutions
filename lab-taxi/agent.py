import numpy as np
from collections import defaultdict
np.random.seed(22)

class Agent:

    def __init__(self, nA, sarsa_type="max", eps_constant_val=None, alpha=0.01, gamma=1.0):
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
        self._eps_constant_val = eps_constant_val
        
        if eps_constant_val is not None:
            self.eps = eps_constant_val
        else:
            self.eps = 1 # initialize to total random agent, use schedule using

        self.alpha = alpha
        self.gamma = gamma

        if sarsa_type == "expected" and eps_constant_val == None:
            #raise UserWarning("Consider not using 'expected sarsa' with dynamic 'eps'.")
            pass
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        return self._policy_eps_greedy(state)
    
    def step(self, state, action, reward, next_state, done, eps):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if self._eps_constant_val is None:
            self.eps = eps
        else:
            self.eps = self._eps_constant_val
        
        next_estimated_return = self._compute_next_estimated_return(next_state)        
        target = reward + self.gamma * next_estimated_return  
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])  

    def _compute_next_estimated_return(self, next_state):
        if self.sarsa_type == "zero":
            next_action = self._policy_eps_greedy(next_state)
            next_estimated_return = self.Q[next_state][next_action]
        elif self.sarsa_type == "max":
            next_action = self._policy_greedy(next_state)
            next_estimated_return = self.Q[next_state][next_action]
        elif self.sarsa_type == "expected":
            actions_returns = self.Q[next_state]
            actions_probs = np.ones_like(actions_returns) * (self.eps / self.nA)
            greedy_action = self._policy_greedy(next_state)
            actions_probs[greedy_action] += 1 - self.eps
            next_estimated_return = np.dot(actions_probs, actions_returns) 
        else:
            raise NotImplementedError
        return next_estimated_return

    def _policy_eps_greedy(self, state):
        if np.random.rand() > self.eps:
            return self._policy_greedy(state)
        else:
            return self._policy_random_uniform()

    def _policy_greedy(self, state):
        return self.Q[state].argmax()

    def _policy_random_uniform(self):
        return np.random.rand(self.nA).argmax()
    
            


         