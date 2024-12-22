#!/usr/bin/env python3
'''
    By Xiazhen Xu <xxx212@student.bham.ac.uk>
'''
import sys
import os
sys.path.append("..")

import numpy as np
from env.disassembly_env_v0 import DisassemblyEnv
from algorithm.dqn_pytorch import Agent
from util.Plot_data_diassembly_v0 import plot_learning_curve

import rospy
import gym

if __name__ == '__main__':
    # rospy.init_node('dirty_moon_disassembly', anonymous=True, log_level=rospy.DEBUG)
    models_dir = "models/PPO"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    env = DisassemblyEnv()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=12,
                  eps_end=0.01, input_dims=[8], lr=0.01)

    # figure_file = 'plots/reward.png'
    x = []
    score_ = []
    avg_score = []
    score_history = []
    # episodes = 500
    steps = 6000
    best_score = env.reward_range[0]

    learn_iters = 0
    n_steps = 0
    env.reset()
    env.reset_communication_error()
    for step in range(steps):
        sp = step
        x.append(sp)
        action = env.action_space.sample()
        print(action)
        # print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        # print("obs:",obs)
        score = reward
        score_.append(score)
        score_episode_mean = np.mean(score_[-10:])
        score_history.append(score_episode_mean)
        asc = np.mean(score_history[-10:])
        avg_score.append(asc)
        env.reset_communication_error()
        plot_learning_curve(x, avg_score)
        # p.stepSimulation()
        # env.render()
        if terminated is True:
            print("done")
            env.reset()
            print("reset")
    