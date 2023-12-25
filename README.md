# Hierarchical-DQN-CartPole-v1
A repository containing all the code for running Hierarchical-DQN algorithm for a gym classic control CartPole environment.

Forked from https://github.com/fedingo/Hierarchical-DQN/, which hasn't been updated for a long time and took some time to firuge out all the dependencies and fix some bugs.

## Environment
Includes the .yml file with the needed conda environment. (tested on linux)

Essencially the most important requirements are:
* python==3.9.*
* gym==0.23.1
* tensorflow==1.*

Might run for other requirements as well, but this is what I found that works.

## Logs
For every iteration generates a new folder that has the name of the environment and a current datetime in its name.

This folder stores the weights for the model, the plot of the reward function graph, and the text file with all the rewards that were used to construct the graph.
