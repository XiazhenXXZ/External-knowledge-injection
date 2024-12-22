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

# from save_result_as_csv import build_force_timestep

from gym import error, spaces, utils
from gym.utils import seeding

from Impedance_controller import *
from constants_disassembly import *
from essential_function import *
from DDPG_n_step.ddpg_torch import Agent

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", None, "IP address of the robot.", required=True)
flags.DEFINE_string("load_gripper", 'false', "Whether or not to load the gripper.")


# class DisassemblyEnv(gym.Env):
#     def __init__(self):
#         # super(DisassemblyEnv, self).__init__()
#         # self.eepub = rospy.Publisher('/cartesian_impedance_controller/equilibrium_pose', geom_msg.PoseStamped, queue_size=10)
#         # self.client = Client("/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node")

#         self.action_space = spaces.Box(
#             np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]).astype(np.float32),
#             np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).astype(np.float32)
#         )

#         self.low = np.array([min_EEP_x, min_EEP_y, min_EEP_z,
#                              min_Ori_r, min_Ori_p, min_Ori_y,
#                              min_F_x, min_F_y, min_F_z,
#                              min_T_x, min_T_y, min_T_z]).astype(np.float32)
#         self.high = np.array([max_EEP_x, max_EEP_y, max_EEP_z,
#                               max_Ori_r, max_Ori_p, max_Ori_y,
#                               max_F_x, max_F_y, max_F_z,
#                               max_T_x, max_T_y, max_T_z]).astype(np.float32)

#         self.observation_space = spaces.Box(self.low, self.high)
def run_episode(env, agent, max_steps_per_episode):
    # state = env.reset_arm()
    total_reward = 0
    step_count = 0
    done = False

    while not done and step_count < max_steps_per_episode:
        action = agent.choose_action(np.array(state))
        next_state, reward, _, done, _ = env.step(np.array(action))
        print(next_state)

        # Store transition in n-step buffer
        agent.remember(state, action, reward, next_state, done)

        if step_count % agent.n_step == 0:
            agent.learn()

        state = next_state
        total_reward += reward
        step_count += 1

    agent.learn()

    return total_reward

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
       
    imp_controller = ImpedencecontrolEnv()
    imp_controller.client
    imp_controller.reset_arm()
    time.sleep(1)
    imp_controller.set_reference_limitation()
    
    episodes = 2
    max_steps_per_episode = 4

    agent = Agent(alpha=0.0001, beta=0.001,
        input_dims=imp_controller.observation_space.shape, tau=0.001,
        batch_size=64, fc1_dims=400, fc2_dims=300,
        n_actions=imp_controller.action_space.shape[0], n_step=5) 
    print("agent")
    
    for episode in range(episodes):
        total_reward = 0
        step_count = 0
        done = False
        state = imp_controller.reset_env()
        print("Number",episode)  

        while not done and step_count < max_steps_per_episode:
            # state = next_state
            print(state)
            action = agent.choose_action(state)
            # print("action", action)
            # action = imp_controller.action_space.sample()
            obs, reward, info, done, terminated = imp_controller.step(np.array(action))
            # next_state, reward, _, done, _ = env.step(np.array(action))
            # print(next_state)

            # Store transition in n-step buffer
            next_state = obs
            agent.remember(state, action, reward, next_state, done)

            # if step_count % agent.n_step == 0:
            #     agent.learn()

            state = next_state
            total_reward += reward
            step_count += 1

            agent.learn()

    plt.plot(episode_rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    # plt.show()
    plt.savefig(figure_file)
        # for i in range(10):
        #     print(i)
            # action_space = spaces.Box(np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]).astype(np.float32),
                                    # np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).astype(np.float32)
                                    # )
            # action = action_space.sample()
            # reward = run_episode(env, agent, max_steps_per_episode)
            # episode_rewards.append(reward)
            # print(action)
            # obs, reward, info, done, terminated = imp_controller.step(np.array(action))
            # print(obs)
            
            # if done is True:
            #     imp_controller.reset_arm()

            # imp_controller.ImpedencePosition(action[0], action[1], action[2], action[3], action[4], action[5]) 

    impedence_controller.terminate()
    roscore.terminate()
    sys.exit()




if __name__ == "__main__":
    app.run(main)