# Udacity Deep Reinforcement learning Nanodegree
## Problem statement
**Project Navigation**: In this project, we have to train an agent (a double-jointed arm) to keep track of a moving target. The environment is [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment provided by [Unity Machine learning Agents](https://github.com/Unity-Technologies/ml-agents).

![Unity Reacher environment](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

**NOTE:**
1. This project was completed in the Udacity Workspace, but the project can also be completed on a local Machine. Instructions on how to download and setup Unity ML environments can be found in [Unity ML-Agents Github repo](https://github.com/Unity-Technologies/ml-agents). 

## Environment
The **state space** has 33 dimensions each of which is a continuous variable. It includes position, rotation, velocity, and angular velocities of the agent.
The **action space** conmprises of action vectors each havinf 4 dimensions, corresponding to torque applicable to two joints. Every entry in the action vector should be a number in the interval `[-1, 1]`.
A **reward** of `+0.1` is provided for each step that agent's hand is in the goal location. The goal of the agent is to maintain contact with the target location for as many time steps as possible.

### Distributed training
For this project, 2 environments are provided:
- The first version contains a single agent
- The second version contains **20 identical** agents, each with its own copy of the environment. This version is particularly useful for algorithms like **PPO**, **A3C**, and **D4PG** that use multiple (non-interacting parallel) copies of the same agent to distribute the task of gathering experience.

    
### Solving the environment
- For the **first version**: The task is episodic, and in order to solve the environment, the agent must get an average score of `+30` over **100** consecutive episodes.
- For the **second version**: Since there are more than 1 agents, we must achieve an average score of `+30` (over **100** consecutive episodes, and over all agents).

## Getting started

1. Download the environment from one of the links below. You need to only select the environment that matches your operating sytem:
   - Version 1: One (1) Agent 
     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
     - Max OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
     - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
     - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
 
   - Version 2: Twenty (20) Agents
     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
     - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
     - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
     - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
     
 (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
 
 (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), the please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

## Dependencies
1. Python 3.6
2. Pytorch
3. Unity ML-Agents

## Solution
I employed [this](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) DDPG implementation provided by Udacity. Since, the enviroment contains 20 agents working in parallel, I had to make some amendments to this implementation. 
1. As suggested in the **Benchmark implementation (Attempt #4)**, the agents learnt from the experience tuples every **20** timesteps and at very update step, the agents learnt **10** times. 
2. Also, gradient clipping as suggested in **Attempt #3** helped improved the training.
```
self.critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
self.critic_optimizer.step()
```
3. Also, to add a bit of exploration while choosing actions, as suggested in the [DDPG](https://arxiv.org/abs/1509.02971) paper, **Ornstein-Uhlenbeck process** was used to add noise to the chosen actions. 
4. Also, performed manual search for the best values of training and model parameters.


## Running the code
1. After installing all dependcies, clone this repository in your local system.
2. Make sure you have **Jupyter** installed. To install Jupyter:
```
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
```
3. Code structure:
   - `Continuous_Control.ipynb` : Main notebook containing the training function
   - `ddpg.py` : code for **DDPG** agent
   - `model.py` : code for **Actor** and **Critic** networks
   - `workspace_utils.py` : code to keep the Udacity workspace awake during training
   

## Results

![DDPG score](https://github.com/thedatamonk/Unity-Continuous-Control/blob/master/plots/score_ddpg.png)
![DDPG score over 100 episodes](https://github.com/thedatamonk/Unity-Continuous-Control/blob/master/plots/score100_ddpg.png)

The implementation was able to solve the environment in approximately **360** episodes.
