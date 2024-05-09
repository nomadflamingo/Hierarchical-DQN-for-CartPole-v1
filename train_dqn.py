"""
@author: Saurabh Kumar
"""

import os
import sys
#matplotlib.use('Agg')

#import clustering
import dqn
import gym
import hierarchical_dqn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from tqdm import trange

from datetime import datetime

"""
Parameters for the environment.
Note that some other parameters like 
DISCOUNT, REPLAY_MEMORY_SIZE, BATCH_SIZE etc.
are set in the dqn.py file
"""
ENV_NAME = 'SUMO-RL-1.3-v0'
RENDER = False
LOAD = False
SUBGOALS = {
    'MountainCar-v0': [\
        [- 1, 0],
        [-.7, 0],
        [-.3, 0],
        [  0, 0],
        [ .5, 0]
    ],
    'CartPole-v1': [[]],
    'SUMO-RL-1.3-v0': [[]],
}
AGENT_TYPE = 'h_dqn'
TESTING = False
REWARD_FN = 'average-speed'

NUM_ITERATIONS = 2000
NUM_TRAIN_EPISODES = 100
NUM_EVAL_EPISODES = 100


# Setting up directories and log files (note that logs from prev iterations are not cleared)
now = (datetime.now()).strftime('%Y-%m-%dT%H-%M-%S')
OUTPUT_FOLDER = os.path.join('outputs', f'{now}_{ENV_NAME}_{REWARD_FN} h_dqn')
WEIGHTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'weights')

TRAIN_LOG_FILE = os.path.join(OUTPUT_FOLDER, 'train_returns.log')
EVAL_LOG_FILE = os.path.join(OUTPUT_FOLDER, 'eval_returns.log')
FIG_FILE = os.path.join(OUTPUT_FOLDER, 'figure.png')

gym.register(
    'SUMO-RL-1.3-v0',
    entry_point='sumo_rl:SumoEnvironment',
    kwargs={
        'net_file': '/home/beimukvo/Documents/work/github_repos_NO_JOKES/sumo-rl/nets/2way-single-intersection/single-intersection.net.xml',
        'route_file': '/home/beimukvo/Documents/work/github_repos_NO_JOKES/sumo-rl/nets/2way-single-intersection/single-intersection-gen.rou.xml',
        'use_gui': False,
        'single_agent': True,
        'num_seconds': 900,
        'sumo_warnings': False,
        'reward_fn': REWARD_FN
    },
)

def log(logfile, iteration, rewards):
    """Function that logs the reward statistics obtained by the agent.

    Args:
        logfile: File to log reward statistics.
        iteration: The current iteration.
        rewards: Array of rewards obtained in the current iteration.
    """
    log_string = ''
    for reward in rewards:
        s = '{} {}\n'.format(iteration, reward)
        log_string += s

    with open(logfile, 'a') as f:
        f.write(log_string)


def make_environment(env_name):
    return gym.make(env_name)


def meta_controller_state(state, original_state):


    return np.zeros()


def check_subgoal(state, subgoal_index):

    target = SUBGOALS[ENV_NAME][subgoal_index]

    if ENV_NAME == 'MountainCar-v0':
        return (state[0] - target[0]) < 0.01
    elif ENV_NAME == 'CartPole-v1':
        return False
    elif ENV_NAME == 'SUMO-RL-1.3-v0':
        return False
    else:
        raise Exception(f'check_subgoal function not defined for environment {ENV_NAME}')


def make_agent(agent_type, env, load = True):
    if agent_type == 'h_dqn':
        meta_controller_state_fn, check_subgoal_fn, num_subgoals = None, check_subgoal, 2

        # subgoals = [\
        #     [-.7,-.2],
        #     [-1,0],
        #     [.5,.2],
        #     [ 1,0]
        # ]
        #clustering.get_cluster_fn(n_clusters=num_clusters, extra_bit=use_extra_bit)

        return hierarchical_dqn.HierarchicalDqnAgent(
            state_sizes= env.observation_space.shape,
            subgoals=SUBGOALS[ENV_NAME],
            num_subgoals=num_subgoals,
            num_primitive_actions= env.action_space.n,
            meta_controller_state_fn=meta_controller_state_fn,
            check_subgoal_fn=check_subgoal_fn,
            load = load,
            weights_root=WEIGHTS_FOLDER)
    else:
        raise Exception(f'Agent type {agent_type} is not supported. (DQN was removed due to some bug in the implementation)')
            

def run(env_name=ENV_NAME,
        agent_type=AGENT_TYPE,
        num_iterations=NUM_ITERATIONS,
        num_train_episodes=NUM_TRAIN_EPISODES,
        num_eval_episodes=NUM_EVAL_EPISODES,
        testing=TESTING,
        load_wieghts=LOAD):
    """Function that executes RL training and evaluation.

    Args:
        env_name: Name of the environment that the agent will interact with.
        agent_type: The type RL agent that will be used for training.
        num_iterations: Number of iterations to train for.
        num_train_episodes: Number of training episodes per iteration.
        num_eval_episodes: Number of evaluation episodes per iteration.
        testing: Whether to start training the model or to go straight to evaluation
        load_wieghts: Whether to load the existing model or start with the new one
    """

    print(env_name)
    env = make_environment(env_name)
    env_test = make_environment(env_name)
    # env_test = Monitor(env_test, directory='videos/', video_callable=lambda x: True, resume=True)
    print('Made environment!')
    print(agent_type)
    agent = make_agent(agent_type, env,  load = load_wieghts)
    print('Made agent!')

    if not os.path.exists(WEIGHTS_FOLDER):
        if load_wieghts:
            raise Exception('No weights folder found, cannot load model')
        os.makedirs(os.path.join(WEIGHTS_FOLDER, 'control'))
        os.makedirs(os.path.join(WEIGHTS_FOLDER, 'meta'))

    train_means = []
    eval_means = []

    plt.ion()
    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle(f'{agent_type}_{env_name}')
    ax[0].set_ylabel('Reward function returns')
    ax[1].set_ylabel('Reward function returns')
    ax[1].set_xlabel('Iteration number')
    ax[0].set_title('Train returns')
    line_train, = ax[0].plot(train_means)
    ax[1].set_title('Eval returns')
    line_eval, = ax[1].plot(eval_means)
    plt.show(block=False)

    if testing:
        num_iterations = 1

    for it in range(num_iterations):
        iter_train_returns = []
        iter_eval_returns = []

        if not testing:
            
            # Run train episodes.
            for train_episode in trange(num_train_episodes):
                # Reset the environment.
                state = env.reset()
                #state = np.expand_dims(state, axis=0)
                episode_reward = 0

                # Run the episode.
                terminal = False

                while not terminal:
                    action = agent.sample(state)
                    # Remove the do-nothing action.
                    # if env_name == 'MountainCar-v0':
                    #     if action == 1:
                    #         env_action = 2
                    #     else:
                    #         env_action = action
                    env_action = action

                    next_state, reward, terminal, _ = env.step(env_action)
                    #next_state = np.expand_dims(next_state, axis=0)

                    agent.store(state, action, reward, next_state, terminal)
                    agent.update()

                    episode_reward += reward
                    # Update the state.
                    state = next_state
                
                iter_train_returns.append(episode_reward)

        if not testing:
            agent.save()

        # Run eval episodes.
        for eval_episode in trange(num_eval_episodes):

            # Reset the environment.
            state = env_test.reset()
            #state = np.expand_dims(state, axis=0)

            episode_reward = 0

            # Run the episode.
            terminal = False

            while not terminal:
                info = None
                if agent_type == 'dqn':
                    action = agent.best_action(state)
                else:
                    action = agent.best_action(state)
                if agent_type == 'h_dqn' and info is not None:
                    curr_state = info[0]
                    if not use_memory:
                        curr_state = np.where(np.squeeze(curr_state) == 1)[0][0]
                    else:
                        curr_state = np.squeeze(curr_state)[-1] - 1
                    goal = info[1]
                    heat_map[curr_state][goal] += 1

                # # Remove the do-nothing action.
                env_action = action
                # if action == 1:
                #     env_action = 2
                # else:
                #     env_action = action

                next_state, reward, terminal, _ = env_test.step(env_action)

                if RENDER:
                    env_test.render()

                #next_state = np.expand_dims(next_state, axis=0)
                agent.store(state, action, reward, next_state, terminal, eval=True)
                # if reward > 1:
                #     reward = 1 # For sake of comparison.

                episode_reward += reward

                state = next_state

            iter_eval_returns.append(episode_reward)

        # log all values
        if not TESTING:
            log(TRAIN_LOG_FILE, it, iter_train_returns)
            log(EVAL_LOG_FILE, it, iter_eval_returns)

        # update plots data
        train_means.append(np.mean(iter_train_returns))  # take the last "num_train_episodes" values and get mean for them
        eval_means.append(np.mean(iter_eval_returns))  # take the last "num_eval_episodes" values and get mean for them
        line_train.set_data(range(len(train_means)), train_means)
        line_eval.set_data(range(len(eval_means)), eval_means)

        print("%d# Iteration: Mean Eval Score: %.2f" %(it, eval_means[-1]))
        
        # update plots
        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()
        plt.draw()
        plt.pause(0.1)

        # save the plot
        plt.savefig(FIG_FILE)

run()


