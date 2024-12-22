#!/usr/bin/env python3
'''
    By Xiazhen Xu <xxx212@student.bham.ac.uk>
'''
import os
import gymnasium as gym
import time
from env.disassembly_yellow_env_v0 import DisassemblyEnv
import numpy as np
from algorithm.ppo_torch import Agent
from util.Plot_data_diassembly_v0 import *



if __name__ == '__main__':
    env = DisassemblyEnv()
    N = 1500
    agent = Agent(n_actions=12, batch_size=5, gamma=0.8,
                  alpha=0.01, n_epochs=10,
                  input_dims=env.observation_space.shape)

    Episodes = 50

    # figure_file = 'plots/reward.png'

    best_score = env.reward_range[0]
    score_history = []
    avg_score_ = []
    actions = []
    steps = []
    x_ = []

    learn_iters = 0
    avg_score = 0
    # n_steps = 0

    # agent.load_models()
    for i in range(Episodes):
        #if i == 0:
         #   pass
        #else:
        # agent.load_models()
        observation = env.reset()
        terminated = False
        score = 0
        n_steps = 0
        while not terminated:
        # for i in range(5):
            if n_steps % 2 == 0:
                action, prob, val = agent.choose_action_hand(observation)
            else:
                action, prob, val = agent.choose_action(observation)
            #action, prob, val = agent.choose_action_hand(observation)
            # print("step:", sp)
            print("episode:", i)
            print("action:", action)
            print("prob:", prob)
            #print("val:", val)
            observation_, reward, terminated, truncated, info = env.step(action)
            # if terminated:
            #     print("done!!!!!!!!!!!")
            # #     break
            # else:
            # action, prob, val = 5, 10, 10
            print(reward)
            n_steps += 1
            score += reward
            print(score)
            agent.remember(observation_, action, prob, val, reward, terminated)
            # print("steps:", n_steps)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
            actions.append(action)
            steps.append(n_steps)
            # build_action_step(steps, actions)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('ave_score', avg_score)
        print('best_score', best_score)
        avg_score_.append(avg_score)
        x_.append(i)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

    x = [i + 1 for i in range(len(score_history))]

    plot_learning_curve(x, score_history)
    # build_reward_episode(x_, score_history)s



#if __name__ == '__main__': 
 #   env = DisassemblyEnv()
  #  agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=12,
   #               eps_end=0.01, input_dims=[8], lr=0.01)

    # figure_file = 'plots/reward.png'
    #x = []
    #score_ = []
    #avg_score = []
    #score_history = []
    #force_ = []
    #steps = 10
    #best_score = env.reward_range[0]

   # learn_iters = 0
    #n_steps = 0
    #env.reset()
    #env.reset_communication_error()
    #for step in range(steps):
        #sp = step
        #x.append(sp)
        #action = env.action_space.sample()
       # print(action)
        # print(action)
     #   obs, reward, terminated, truncated, info, Force = env.step(action)
      #  print(reward)
        # print("obs:",obs)
        #force = Force[1]
       # score = reward
       # score_.append(score)
        # score_episode_mean = np.mean(score_[-10:])
       # score_history.append(score_)
       # asc = np.mean(score_history[-10:])
      #  avg_score.append(asc)
       # env.reset_communication_error()
      #  force_.append(force)
        # plot_learning_curve(x, avg_score)
       # plot_force_curve(x, force_)

       # if terminated is True:
           # print("done")
          #  env.reset()
           # print("reset")
    
    #plot_learning_curve(x, avg_score)