# Hierarchical-DQN-for-SUMO-RL
Repository containing code for running Hierarchical-DQN algorithm for SUMO-RL environment (version 1.3.0). Also supports 2 gym classic control environments:
* CartPole-v1
* MountainCar-v0

Forked from https://github.com/fedingo/Hierarchical-DQN/, which hasn't been updated for a long time and took some time to figure out all the dependencies and fix some bugs.

## Installation and dependencies
Includes the .yml file with the needed conda environment. (tested on linux)

Essencially the most annoying requirements are:
* python==3.9.*
* gym==0.23.1
* tensorflow==1.*
* protobuf<=3.20.*
* sumo-rl==1.3.0

To install, run:
```
conda env create -f conda.yml
conda activate h-dqn
pip install sumo-rl==1.3.0 --no-deps
```

Might run for other requirements as well, but this is what I found that works.

## Logs
For every iteration generates a new folder that has the name of the environment and a current datetime in its name.

This folder stores the weights for the model, the plot of the reward function graph, and the text file with all the rewards that were used to construct the graph.

## Notes on the environments
For CartPole, no subgoals were chosen and the function check_subgoal simply returns False, because I decided not to implement any subgoals in order to not slow down the learning for the algorithm.

I removed the option to run only DQN agent from the implementation because it wasn't working correctly and I didn't need to fix it for my goals. If you want to run DQN, one option is to just run H-DQN with no subgoals, which I did for CartPole.
Or just use DQN implementation from stable-baselines3

## When implementing a new environemnt
1. Register your environment in gym, if needed
1. Define your subgoals in the `SUBGOALS` dictionary in `train_dqn.py` for your env
1. Define `check_subgoals` function in `train_dqn.py` for your env