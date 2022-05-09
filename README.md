[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# My Solutions to the Deep Reinforcement Learning Nanodegree

![Trained Agents][image1]

This repository contains my solutions to the Labs / Projects of Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program in addition to the default materials provided [here](https://github.com/udacity/deep-reinforcement-learning).


## Table of Contents

### Labs / Projects

My solutions to the labs and projects can be found below.  All of the projects use rich simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents). In the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program, I received reviews of my projects.

- [x] [The Taxi Problem](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/lab-taxi): In this lab, you will train a taxi to pick up and drop off passengers.
- [ ] [Navigation](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/p1_navigation): In the first project, you will train an agent to collect yellow bananas while avoiding blue bananas.
- [ ] [Continuous Control](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/p2_continuous-control): In the second project, you will train an robotic arm to reach target locations.
- [ ] [Collaboration and Competition](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/p3_collab-compet): In the third project, you will train a pair of agents to play tennis! 

### Tutorials

The tutorials led me through implementing various algorithms in reinforcement learning.  All of the code is in PyTorch (v0.4) and Python 3.

- [x] [Dynamic Programming](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/dynamic-programming): Implement Dynamic Programming algorithms such as Policy Evaluation, Policy Improvement, Policy Iteration, and Value Iteration. 
- [x] [Monte Carlo](https://github.com/udacity/deep-reinforcement-learning/tree/master/monte-carlo): Implement Monte Carlo methods for prediction and control. 
- [x] [Temporal-Difference](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/temporal-difference): Implement Temporal-Difference methods such as Sarsa, Q-Learning, and Expected Sarsa. 
- [x] [Discretization](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/discretization): Learn how to discretize continuous state spaces, and solve the Mountain Car environment.
- [x] [Tile Coding](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/tile-coding): Implement a method for discretizing continuous state spaces that enables better generalization.
- [x] [Deep Q-Network](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/dqn): Explore how to use a Deep Q-Network (DQN) to navigate a space vehicle without crashing.
- [ ] [Robotics](https://github.com/dusty-nv/jetson-reinforcement): Use a C++ API to train reinforcement learning agents from virtual robotic simulation in 3D. (_External link_)
- [ ] [Hill Climbing](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/hill-climbing): Use hill climbing with adaptive noise scaling to balance a pole on a moving cart.
- [ ] [Cross-Entropy Method](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/cross-entropy): Use the cross-entropy method to train a car to navigate a steep hill.
- [ ] [REINFORCE](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/reinforce): Learn how to use Monte Carlo Policy Gradients to solve a classic control task.
- [ ] **Proximal Policy Optimization**: Explore how to use Proximal Policy Optimization (PPO) to solve a classic reinforcement learning task. (_Coming soon!_)
- [ ] **Deep Deterministic Policy Gradients**: Explore how to use Deep Deterministic Policy Gradients (DDPG) with OpenAI Gym environments.
  * [Pendulum](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/ddpg-pendulum): Use OpenAI Gym's Pendulum environment.
  * [BipedalWalker](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/ddpg-bipedal): Use OpenAI Gym's BipedalWalker environment.
- [ ] [Finance](https://github.com/maksym-del/deep-reinforcement-learning/tree/master/finance): Train an agent to discover optimal trading strategies.

### Resources

* [RL Cheatsheet](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/cheatsheet): [The PDF file](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet/cheatsheet.pdf) contains key definitions, formulas and pseudocodes. 

## OpenAI Gym Benchmarks

### Classic Control
- `Acrobot-v1` with [Tile Coding](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/tile-coding/Tile_Coding_Solution.ipynb) and Q-Learning  
- `Cartpole-v0` with [Hill Climbing](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/hill-climbing/Hill_Climbing.ipynb) | solved in 13 episodes
- `Cartpole-v0` with [REINFORCE](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/reinforce/REINFORCE.ipynb) | solved in 691 episodes 
- `MountainCarContinuous-v0` with [Cross-Entropy Method](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb) | solved in 47 iterations
- `MountainCar-v0` with [Uniform-Grid Discretization](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/discretization/Discretization_Solution.ipynb) and Q-Learning | solved in <50000 episodes
- `Pendulum-v0` with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb)

### Box2d
- `BipedalWalker-v2` with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/ddpg-bipedal/DDPG.ipynb)
- `CarRacing-v0` with **Deep Q-Networks (DQN)** | _Coming soon!_
- `LunarLander-v2` with [Deep Q-Networks (DQN)](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb) | solved in 1504 episodes

### Toy Text
- `FrozenLake-v0` with [Dynamic Programming](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/dynamic-programming/Dynamic_Programming_Solution.ipynb)
- `Blackjack-v0` with [Monte Carlo Methods](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/monte-carlo/Monte_Carlo_Solution.ipynb)
- `CliffWalking-v0` with [Temporal-Difference Methods](https://github.com/maksym-del/deep-reinforcement-learning/blob/master/temporal-difference/Temporal_Difference_Solution.ipynb)

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).  This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful.  This was confirmed to work in Windows 10 Home.  

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

---
<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
