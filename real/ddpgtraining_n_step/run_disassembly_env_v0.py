import os
import sys
import rospy
import numpy as np
import shlex
import time

import time
import subprocess

import random
from dynamic_reconfigure.client import Client
from absl import app, flags, logging
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import math
import tf
import tf.transformations
import re
import json
import threading
import matplotlib.pyplot as plt

import geometry_msgs.msg as geom_msg
from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveGoal, MoveAction, GraspAction, GraspGoal
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import queue

# from save_result_as_csv import build_force_timestep

from gym import error, spaces, utils
from gym.utils import seeding

from Impedance_controller import *
from constants_disassembly import *
from essential_function import *

from DDPG_n_step.ddpg_torch import Agent
from Plot_data_diassembly_v0 import *

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", None, "IP address of the robot.", required=True)
flags.DEFINE_string("load_gripper", 'false', "Whether or not to load the gripper.")

ImpedencecontrolEnv.force_based_controller

def run_episode(env, agent, max_steps_per_episode, collect_event,data,data_queue):
    state = env.reset_env()
    total_reward = 0
    step_count = 0
    done = False

    while not done and step_count < max_steps_per_episode:
        action = agent.choose_action(np.array(state))
        print("action:",action)
        action_ = env.force_based_controller() + action
        print("mix action", action)
        # main_execution(action)
        collect_event.set()
        next_state, reward, _, done, _ = env.step(np.array(action_))
        # print(next_state)
        print("done?", done)
        collect_event.clear()
        while not data_queue.empty():
            data.append(data_queue.get())
        # Store transition in n-step buffer
        agent.remember(state, action, reward, next_state, done)

        if step_count % agent.n_step == 0:
            agent.learn()

        state = next_state
        total_reward += reward
        step_count += 1

    agent.learn()

    return total_reward

# class execution_module():
#     def __init__(self, action):
#         self.action_input = action

#     def action_execution(self, _):
#         try:
#             roscore = subprocess.Popen('roscore')   
#             time.sleep(1)

#             impedence_controller = subprocess.Popen(['roslaunch', 'franka_real_demo', 'impedance.launch',
#                                                     f'robot_ip:={FLAGS.robot_ip}', f'load_gripper:={FLAGS.load_gripper}'],
#                                                     stdout=subprocess.PIPE)
#             time.sleep(1)
#             rospy.init_node('franka_control_api')
#             # env = DisassemblyEnv()
#             imp_controller = ImpedencecontrolEnv()
#             imp_controller.reset_arm()
#             time.sleep(1)
            
#             imp_controller.client
            
#             imp_controller.set_reference_limitation()

#             for _ in range(10):
#                 obs, reward, info, done, terminated = imp_controller.step(np.array(self.action_input))
#                 print(obs)
                    
#                 if done is True:
#                     imp_controller.reset_arm()

#             # imp_controller.ImpedencePosition(action[0], action[1], action[2], action[3], action[4], action[5]) 
           
#             impedence_controller.terminate()
#             roscore.terminate()
#             sys.exit()


#         except:
#             impedence_controller.terminate()
#             roscore.terminate()
#             sys.exit()

#         # return obs, reward, info, done, terminated
    
# def main_execution(action_choice):
#     app.run(execution_module(action_choice).action_execution)

def main(_):
    try:
        roscore = subprocess.Popen('roscore')   
        time.sleep(1)

        impedence_controller = subprocess.Popen(['roslaunch', 'franka_real_demo', 'impedance.launch',
                                                f'robot_ip:={FLAGS.robot_ip}', f'load_gripper:={FLAGS.load_gripper}'],
                                                stdout=subprocess.PIPE)
        time.sleep(1)
        rospy.init_node('franka_control_api')
        # env = DisassemblyEnv()
    except:
        impedence_controller.terminate()
        roscore.terminate()
        sys.exit()
       
    env = ImpedencecontrolEnv()
    env.client
    env.reset_arm()
    # gripper_control(1)
    time.sleep(1)
    env.set_reference_limitation()

    data_queue = queue.Queue()
    stop_event = threading.Event()
    collect_event = threading.Event()

    collector_thread = threading.Thread(target = env.data_collector, args=(data_queue, stop_event, collect_event))
    collector_thread.start()

    episode_rewards = []
    figure_file = 'plots/reward.png'
    
    episodes = 30
    max_steps_per_episode = 40

    agent = Agent(alpha=0.0001, beta=0.001,
        input_dims=env.observation_space.shape, tau=0.001,
        batch_size=64, fc1_dims=400, fc2_dims=300,
        n_actions=env.action_space.shape[0], n_step=5) 
    
    # for episode in range(episodes):
    #     total_reward = 0
    #     step_count = 0
    #     done = False
    #     state = imp_controller.reset_env()
    #     print("Number",episode)  

    #     while not done and step_count < max_steps_per_episode:
    #         # state = next_state
    #         print(state)
    #         action = agent.choose_action(state)
    #         # print("action", action)
    #         # action = imp_controller.action_space.sample()
    #         obs, reward, info, done, terminated = imp_controller.step(np.array(action))
    #         # next_state, reward, _, done, _ = env.step(np.array(action))
    #         # print(next_state)

    #         # Store transition in n-step buffer
    #         next_state = obs
    #         agent.remember(state, action, reward, next_state, done)

    #         # if step_count % agent.n_step == 0:
    #         #     agent.learn()

    #         state = next_state
    #         total_reward += reward
    #         step_count += 1

    #         agent.learn()
    
    for episode in range(episodes):   
        print(episode)
        data = []
        reward = run_episode(env, agent, max_steps_per_episode,collect_event,data,data_queue)
        episode_rewards.append(reward)
        env.save_data_to_csv(data, episode)
        print(f"Episode {episode + 1}: Total Reward = {reward}")

    plt.plot(episode_rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    # plt.show()
    plt.savefig(figure_file)
    impedence_controller.terminate()
    roscore.terminate()
    sys.exit()




if __name__ == "__main__":
    app.run(main)
    