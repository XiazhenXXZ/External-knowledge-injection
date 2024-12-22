import sys
import rospy
import numpy as np
import shlex
import time
# from psutil import Popen
import geometry_msgs.msg as geom_msg
import time
import subprocess
# from subprocess import PIPE
from dynamic_reconfigure.client import Client
from absl import app, flags, logging
from scipy.spatial.transform import Rotation as R
import os
import gymnasium as gym
import math
import tf
import tf.transformations
import re
import json
import threading
import random
import matplotlib.pyplot as plt
import pandas as pd
import queue

import franka_msgs.msg
import message_filters
import actionlib

from gym import error, spaces, utils
from gym.utils import seeding

from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveGoal, MoveAction, GraspAction, GraspGoal
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from constants_disassembly import *
from essential_function import *

# from save_result_as_csv import build_force_timestep
# FLAGS = flags.FLAGS
# flags.DEFINE_string("robot_ip", None, "IP address of the robot.", required=True)
# flags.DEFINE_string("load_gripper", 'false', "Whether or not to load the gripper.")

class ImpedencecontrolEnv(gym.Env):
    def __init__(self):
        super(ImpedencecontrolEnv, self).__init__()
        self.eepub = rospy.Publisher('/cartesian_impedance_controller/equilibrium_pose', geom_msg.PoseStamped, queue_size=10)
        self.client = Client("/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node")

        # Force
        self.franka_EE_trans = []
        self.franka_EE_quat = []
        self.F_T_EE = np.empty((4,4))
        self.K_F_ext_hat_K = []
        self.ext_force_ee = []
        self.ext_torque_ee = []
        self.quat_fl = []
        self.quat_rr = []
        self.franka_fault = None

        self.force_history = []
        self.stop_flag = 0
        self.time_window = 5
        # self.threshold = 0.1
        self.max_steps = 20

        # sub and pub
        # self.sub = rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.convert_to_geometry_msg, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.franka_callback)

        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.GetEEforce)
        
        # sub.registerCallback(self.gripper_state_callback)

        # ts = message_filters.TimeSynchronizer([franka_sub, eeforce_sub, gripper_sub], queue_size=10)
        # ts.registerCallback(self.GetEEforce)
        self.action_space = spaces.Box(
            np.array([-0.005, -0.005, -0.005, -0.005, -0.005, -0.005]).astype(np.float32),
            np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005]).astype(np.float32)
        )

        # self.low = np.array([min_EEP_x, min_EEP_y, min_EEP_z,
        #                      min_Ori_r, min_Ori_p, min_Ori_y,
        #                      min_F_x, min_F_y, min_F_z,
        #                      min_T_x, min_T_y, min_T_z]).astype(np.float32)
        # self.high = np.array([max_EEP_x, max_EEP_y, max_EEP_z,
        #                       max_Ori_r, max_Ori_p, max_Ori_y,
        #                       max_F_x, max_F_y, max_F_z,
        #                       max_T_x, max_T_y, max_T_z]).astype(np.float32)
        
        self.low = np.array([min_EEP_x, min_EEP_y, min_EEP_z,
                             min_Ori_r, min_Ori_p, min_Ori_y,
                             min_F_x, min_F_y, min_F_z,
                             min_T_x, min_T_y, min_T_z]).astype(np.float32)
        self.high = np.array([max_EEP_x, max_EEP_y, max_EEP_z,
                              max_Ori_r, max_Ori_p, max_Ori_y,
                              max_F_x, max_F_y, max_F_z,
                              max_T_x, max_T_y, max_T_z]).astype(np.float32)

        self.observation_space = spaces.Box(self.low, self.high)
    
    def step(self, action):
        # self.step_count += 1
        self.z_before = self.F_T_EE[2, 3]
        # print("step")
        # exc_action = np.array(action)
        # self.ImpedencePosition(exc_action)    
        self.ImpedencePosition(action[0], action[1], action[2], action[3], action[4], action[5])
        time.sleep(0.1)    

        # End effector position
        EEP_x = self.F_T_EE[0, 3]
        EEP_y = self.F_T_EE[1, 3]
        EEP_z = self.F_T_EE[2, 3]

        self.z = EEP_z
        print("z:", self.z)
        self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
        self.currentOrn = self.quat_rr
        self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
        self.Force = self.K_F_ext_hat_K[0:3]
        self.Torque = self.K_F_ext_hat_K[3:6]

        self.done = False

        self.reward = 0
        self.reward += 100 * (self.z - self.z_before)
        # self.reward += -3 * (EEP_y - 0.02)
        # self.reward += 3 * (EEP_x - 0.5)
        print("rH",self.reward)
        self.reward += 0.01 * (10 - abs(self.resultant_force))
        print("rF",self.reward)
        if self.resultant_force >= 15:
            self.reward -= 0.05
            self.done = False
            print("Failed due to collision")

        


        elif self.z > 0.22:
            self.done = True
            self.reward += 1
            print("Success")

        # elif EEP_x >= 0.6 or EEP_x <= 0.4:
        #     self.done = True

        # elif EEP_y >= 0.07:
        #     self.done = True
        #     self.reward -= 50
        #     print("Fail")

        else:
            self.done = False

        self.terminated = 0
        self.info = {}
        self.obs = [self.Position[0], self.Position[1], self.Position[2],
            self.Orientation[0], self.Position[1], self.Position[2],
            self.Force[0], self.Force[1], self.Force[2],
            self.Torque[0], self.Torque[1], self.Torque[2]]
        # print("finish_step")
        
        return np.array(self.obs).astype(np.float32), self.reward, self.info, self.done, self.terminated
    
    def reset_env(self):
        # End effector position
        # print("reset the environment")
        gripper_control(1)
        # # print("close the gripper")
        # # time.sleep(0.5)
        # open_gripper()
        time.sleep(1)
        # print("open the gripper")
        self.ImpedencePosition(0,0,0.1,0,0,0)
        time.sleep(0.5)
        self.reset_arm()
        time.sleep(0.5)
        self.robot_control_grasptarget([0.49246 + self.generate_random_between(-0.01,0.01),0.03040 + self.generate_random_between(-0.01,0.01),0.14316 + self.generate_random_between(-0.01,0.01),np.pi + self.generate_random_between(-0.01,0.01),0 + self.generate_random_between(-0.01,0.01),np.pi/2 + self.generate_random_between(-0.01,0.01)])
        EEP_x = self.F_T_EE[0, 3]
        EEP_y = self.F_T_EE[1, 3]
        EEP_z = self.F_T_EE[2, 3]
        self.Position = np.array([EEP_x, EEP_y, EEP_z]) #.astype(np.float32)
        self.currentOrn = self.quat_rr
        self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
        self.Force = self.K_F_ext_hat_K[0:3]
        self.Torque = self.K_F_ext_hat_K[3:6]
        observation = [self.Position[0], self.Position[1], self.Position[2],
            self.Orientation[0], self.Position[1], self.Position[2],
            self.Force[0], self.Force[1], self.Force[2],
            self.Torque[0], self.Torque[1], self.Torque[2]]
        # print(observation)
        return observation
    
    def force_based_controller(self):
        force = self.resultant_force
        if force <= 5:
            ass_action = [0,0,0.01,0,0,0]
        else:
            ass_action = [0,0,0,0,0,0]

        return ass_action 
    
    def grasp_fail(self, gripper_width):
        # self.gripper_width = close_gripper()
        if gripper_width < 0.002:
            self.grippstate = 1
        else:
            self.grippstate = 0
        
        return self.grippstate

    def parm_to_selection(self, id):
        if id == 0:
            self.disassembly_state = self.move_up()

        elif id ==1:
            self.disassemblystate = self.move_down()

        elif id ==2:
            self.disassemblystate = self.move_left()

        elif id ==3:
            self.disassemblystate = self.move_right()

        elif id==4:
            self.disassemblystate = self.move_front()

        elif id==5:
            self.disassemblystate = self.move_back()

        return self.disassemblystate

    def move_up(self):
        print("+z")
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            # print(fh[0])
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, 0, 0.05, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([0, 0, 0.05, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all():
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break


        disassemblydiff = abs(EEP_z_-EEP_z)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1

        return self.disassemblystate

    def move_down(self):
        print("-z")
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, 0, -0.05, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([0, 0, -0.05, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.0001).all() or (diff_ <= 0.0001).all():
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        # EEP_z_ = self.F_T_EE[2, 3]
        # fh = []
        # while True:
        #     _ = ImpedencecontrolEnv.monitor_force_change(fh)
        #     stopf = _
        #     print("stop singal", stopf)
        #     if stopf == 0:
        #         self.ImpedencePosition(0, 0, -0.05, 0, 0, 0)
        #     if stopf == 1:
        #         self.ImpedencePosition(0,0,0,0,0,0)
        #         break
        disassemblydiff = abs(EEP_z_-EEP_z)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_right(self):
        print("+y")
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, 0.05, 0, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([0, 0.05, 0, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])

                diff__ = abs (EEP_y - EEP_y_)
                print(diff__)
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all() or (diff__ <= 0.045):
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        # EEP_y_ = self.F_T_EE[1, 3]
        # fh = []
        # while True:
        #     _ = ImpedencecontrolEnv.monitor_force_change(fh)
        #     stopf = _
        #     print("stop singal", stopf)
        #     if stopf == 0:
        #         self.ImpedencePosition(0, 0.05, 0, 0, 0, 0)
        #     if stopf == 1:
        #         self.ImpedencePosition(0,0,0,0,0,0)
        #         break
        disassemblydiff = abs(EEP_y_-EEP_y)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_left(self):
        print("-y")
        stopf = 0
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                target_c = np.array(target)
                # print("Original position:",target_c)
                self.ImpedencePosition(0, -0.05, 0, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                # print("target_ = ", target_)
                target0 = target_c + np.array([0, -0.05, 0, 0, 0, 0])
                # print("target0",target0)
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all():
                    stopf = 1
                    break

                # elif (diff_ <= 0.002).all():
                #     stopf = 1
                #     print("it doesn't move at all")
                #     break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        # while True:
        #     _ = ImpedencecontrolEnv.monitor_force_change(fh)
        #     stopf = _
        #     print("stop singal", stopf)
        #     if stopf == 0:
        #         self.ImpedencePosition(0, -0.05, 0, 0, 0, 0)
        #     if stopf == 1:
        #         self.ImpedencePosition(0,0,0,0,0,0)
        #         break
        disassemblydiff = abs(EEP_y_-EEP_y)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_front(self):
        print("+x")
        stopf = 0
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0.05, 0, 0, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([0.05, 0, 0, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all():
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break

        disassemblydiff = abs(EEP_x_-EEP_x)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1

        stopf = 0
        return self.disassemblystate
        

    def move_back(self):
        print("-x")
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                target_c = np.array(target)
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(-0.05, 0, 0, 0, 0, 0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target_ = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                target0 = target_c + np.array([-0.05, 0, 0, 0, 0, 0])
                diff = abs (target_[:3]  - target0[:3])
                # print("diff::::", diff)
                diff_ = abs (target_[:3] - target_c[:3])
                
                # print("diff#-##::::", diff_)
                if(diff <= 0.003).all() or (diff_ <= 0.003).all():
                    stopf = 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        disassemblydiff = abs(EEP_x_-EEP_x)
        if disassemblydiff >= 0.048:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate

    
    def set_reference_limitation(self):
        time.sleep(1)
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            self.client.update_configuration({"translational_clip_" + direction: 0.01})
            self.client.update_configuration({"rotational_clip_" + direction: 0.04})
        time.sleep(1)
    
    def GetEEforce(self, FandT):
        # self.gripper_width = gripper.position[0]
        self.forceandtorque= np.array(FandT.K_F_ext_hat_K)
        self.Force = self.forceandtorque[0:3]
        self.Torque = self.forceandtorque[3:6]
        # self.Force = self.franka_callself.GetEEforce()back[0:3]
        # self.Torque = self.franka_callback[3:6]
        self.Fx = self.Force[0]
        self.Fy = self.Force[1]
        self.Fz = self.Force[2]
        self.Tx = self.Torque[0]
        self.Ty = self.Torque[1]
        self.Tz = self.Torque[2]
        self.resultant_force = math.sqrt(self.Fx**2 + self.Fy**2 + self.Fz**2)
        self.resultant_torque = math.sqrt(self.Tx**2 + self.Ty**2 + self.Tz**2)
        # print(self.Fz)
        # print(self.Fx, self.Fy, self.Fz, self.resultant_force, self.Tx, self.Ty, self.Tz, self.resultant_torque)
        # build_force_timestep(self.resultant_force)

        # print("force:", Fz,Fy,Fz)
        
        return self.resultant_force, self.Fz
    
    def generate_chart(self, stop_event, update_interval=0.1, save_path = 'final_plot_1.png'):
        plt.ion()
        fig, ax = plt.subplots()
        
        x_data = []
        y_data = []
        start_time = time.time()

        line, = ax.plot([], [], 'b')
        while not stop_event.is_set():
            y = self.Fz
            x_data.append(time.time() - start_time)
            y_data.append(y)

            line.set_xdata(x_data)
            line.set_ydata(y_data)

            ax.relim()
            ax.autoscale_view(True, True, True)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(update_interval)

        plt.ioff()
        plt.savefig(save_path)
        plt.show()
        
    def data_collector(self, data_queue, stop_event, collect_event, update_interval=1):
        start_time = time.time()
        while not stop_event.is_set():
            if collect_event.is_set():
                current_time = time.time() - start_time
                fx = self.Force[0]
                fy = self.Force[1]
                fz = self.Force[2]
                tx = self.Torque[0]
                ty = self.Torque[1]
                tz = self.Torque[2]
            
                data_queue.put((current_time, fx, fy, fz, tx, ty, tz))
                time.sleep(update_interval)

    def save_data_to_csv(self, data, episode_num, base_path='force_torque_data.csv'):
        file_name = f'data/{base_path}_episode_{episode_num}.csv'
        df = pd.DataFrame(data, columns=['Time', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
        df.to_csv(file_name, index=False)
    
    def calculate_average_force(self, force_history):
        
        return np.mean(force_history)
    
    def is_average_force_change_abnormal(self,prev_avg_force, curr_avg_force):
        force_change_rate = np.abs(curr_avg_force - prev_avg_force) / (np.abs(prev_avg_force) + 1e-6)

        return np.any(force_change_rate > 0.5)
    
    def monitor_force_change(self, force_history):
        self.stop_flag = 0
        current_forces = self.resultant_force
        force_history.append(current_forces)

        if len(force_history) > self.time_window:
            # print(len(force_history))
            force_history.pop(0)
        

        if len(force_history) == self.time_window:
            previous_average_force = self.calculate_average_force(force_history[:self.time_window // 2])
            current_average_force = self.calculate_average_force(force_history[self.time_window // 2:])

            if self.is_average_force_change_abnormal(previous_average_force, current_average_force):
                self.stop_flag = 1

        return self.stop_flag
        

    def ImpedencePosition2(self, dx, dy, dz, da, db, dc, speed):
        print("start")
        EEP_x = self.F_T_EE[0, 3]
        EEP_y = self.F_T_EE[1, 3]
        EEP_z = self.F_T_EE[2, 3]
        Euler_angle =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                    Euler_angle[1], 
                                                    Euler_angle[2]
                                                    )
        self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
        # print(self.Position)
        self.currentOrn = self.quat_rr
        self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
        # print(self.Orientation)
            
        target_position = np.array([self.Position[0]+ dx, self.Position[1]+ dy, self.Position[2]+ dz])
        # print(target_position)
        target_orientation = np.array([self.Orientation[0]+ da, self.Orientation[1]+ db, self.Orientation[2]+ dc])

        distance = np.linalg.norm(target_position - self.Position)
        # print("distance",distance)

        total_time = distance / speed
        # print("total_time",total_time)

        time_step = 0.05
        num_steps = int(total_time / time_step)
        # print("number of steps",num_steps)

        positions = []
        orientations = []

        for step in range(1, num_steps + 1):
            alpha = step / num_steps
            intermediate_position = (1 - alpha) * self.Position + alpha * target_position
           
            positions.append(intermediate_position)
            # print("inter position",positions)
            quat_start = self.quat_rr
            quat_target = tf.transformations.quaternion_from_euler(target_orientation[0],
                                                                   target_orientation[1],
                                                                   target_orientation[2])
            
            intermediate_quat = tf.transformations.quaternion_slerp(quat_start, quat_target, alpha)
            # intermediate_quat = R.from_euler('xyz', target_orientation).as_quat()
            orientations.append(intermediate_quat)

        for step in range(num_steps):

            msg = geom_msg.PoseStamped()
            msg.header.frame_id = "0"
            msg.header.stamp = rospy.Time.now()
            msg.pose.position = geom_msg.Point(positions[step][0], positions[step][1], positions[step][2])

            msg.pose.orientation = geom_msg.Quaternion(orientations[step][0], orientations[step][1], orientations[step][2], orientations[step][3])

            self.eepub.publish(msg)
            time.sleep(time_step)
        
        print("finish loop")
        msg.pose.position = geom_msg.Point(target_position[0], target_position[1], target_position[2])
        quat_final = R.from_euler('xyz', target_orientation).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat_final[0], quat_final[1], quat_final[2], quat_final[3])
        self.eepub.publish(msg)
        time.sleep(3)
        




    def PullandTwist(self,targetc,dz,dc):
        time.sleep(1)
        while True:
            if abs(self.F_T_EE[2, 3] - targetc) > 0.01:
                if abs(self.Fz) <= 8:
                    self.ImpedencePosition(0,0,dz,0,0,0)
                elif abs(self.Fz) >8 and abs(self.Fz)<= 10:
                    self.ImpedencePosition(0,0,0.8*dz,0,0,0.5*dc)
                elif abs(self.Fz) >10:
                    self.ImpedencePosition(0,0,0.3*dz,0,0,dc)
                else:
                    self.ImpedencePosition(0,0,0,0,0,0)
            else:
                break

        time.sleep(1)

    def PullandTwist_simple(self, dz, dc):
        self.ImpedencePosition(0,0,0,0,0,dc)
        time.sleep(2.5)
        self.ImpedencePosition(0,0,dz,0,0,0)
        time.sleep(2)
        

    def PullandTwist_fake(self,targetc,dz,dc):
        time.sleep(1)
        while True:
            if abs(self.F_T_EE[2, 3] - targetc) > 0.01:
                if abs(self.Fz) <= 5:
                    self.ImpedencePosition(self.generate_random_between(-0.003,0.003),
                                           self.generate_random_between(-0.003,0.003),
                                           dz,
                                           self.generate_random_between(-0.03,0.03),
                                           self.generate_random_between(-0.03,0.03),
                                           self.generate_random_between(-0.03,0.03))
                elif abs(self.Fz) >5 and abs(self.Fz)<= 10:
                    self.ImpedencePosition(self.generate_random_between(-0.003,0.003),
                                           self.generate_random_between(-0.003,0.003),
                                           dz,
                                           self.generate_random_between(-0.02,0.02),
                                           self.generate_random_between(-0.02,0.02),
                                           dc)
                elif abs(self.Fz) >10:
                    self.ImpedencePosition(self.generate_random_between(-0.003,0.003),
                                           self.generate_random_between(-0.003,0.003),
                                           0.05*dz,
                                           self.generate_random_between(-0.02,0.02),
                                           self.generate_random_between(-0.02,0.02),
                                           dc)
                else:
                    self.ImpedencePosition(0,0,0,0,0,0)
            else:
                break

        time.sleep(1)

    def generate_random_between(self,minv,maxv):
        return minv + (maxv - minv) * random.random()
    
    def selection_skill(self):
        self.ImpedencePosition(-0.002,0,0,0,0,0)
        self.ImpedencePosition(0.002,0,0,0,0,0)
        self.ImpedencePosition(0,-0.002,0,0,0,0)
        self.ImpedencePosition(0,0.002,0,0,0,0)
        self.ImpedencePosition(0,0,-0.002,0,0,0)
        self.ImpedencePosition(0,0,0.002,0,0,0)


    def FakeRLpolicy(self, dx, dy, dz):
        time.sleep(1)
        EEP_x = self.F_T_EE[0, 3]
       
        EEP_y = self.F_T_EE[1, 3]
        EEP_z = self.F_T_EE[2, 3]
        target = np.array([dx+EEP_x, dy+EEP_y, dz+EEP_z])
        i = 0
        while True:
            ddx = 0 if dx == 0 else 1
            ddy = 0 if dy == 0 else 1
            ddz = 0 if dz == 0 else 1
            currentP = np.array([self.F_T_EE[0,3],self.F_T_EE[1,3], self.F_T_EE[2,3]])
            diff = abs(target - currentP)
            # print('target',target)
            # print('currentP', currentP)
            print('diff', diff)
            xxx = (diff > 0.05).any()
            print('condition', xxx)
            if (diff > 0.05).any():
                if (i+1)%3 == 0:
                    print('i=',i)
                    self.ImpedencePosition(self.generate_random_between(-0.002,0.002),
                                           self.generate_random_between(-0.002,0.002),
                                           self.generate_random_between(-0.002,0.002),
                                           self.generate_random_between(-0.01,0.01),
                                           self.generate_random_between(-0.01,0.01),
                                           self.generate_random_between(-0.01,0.01))
                else:
                    self.ImpedencePosition(ddx*0.01,ddy*0.01,ddz*0.01,self.generate_random_between(-0.01,0.01),
                                           self.generate_random_between(-0.01,0.01),
                                           self.generate_random_between(-0.01,0.01))
                    
                    
                    # print('z')

                i += 1
            else:
                break



    def FakeRLpolicy_random(self, dx, dy, dz):
        time.sleep(1)
        EEP_x = self.F_T_EE[0, 3]
       
        EEP_y = self.F_T_EE[1, 3]
        EEP_z = self.F_T_EE[2, 3]
        target = np.array([dx+EEP_x, dy+EEP_y, dz+EEP_z])
        i = 0
        while True:
            currentP = np.array([self.F_T_EE[0,3],self.F_T_EE[1,3], self.F_T_EE[2,3]])
            diff = abs(target - currentP)
            # print('target',target)
            # print('currentP', currentP)
            print('diff', diff)
            xxx = (diff > 0.05).any()
            print('condition', xxx)
            if (diff > 0.05).any():
                for i in range(50):
                    if (i+1)%3 == 0:
                        print('i=',i)
                        self.ImpedencePosition(self.generate_random_between(-0.005,0.005),
                                           self.generate_random_between(-0.005,0.005),
                                           self.generate_random_between(-0.005,0.005),
                                           self.generate_random_between(-0.05,0.05),
                                           self.generate_random_between(-0.05,0.05),
                                           self.generate_random_between(-0.05,0.05))
                    else:
                        self.ImpedencePosition(dx*0.01,dy*0.01,dz*0.01,self.generate_random_between(-0.03,0.03),
                                           self.generate_random_between(-0.03,0.03),
                                           self.generate_random_between(-0.03,0.03))
                    # print('z')

                break
            else:
                break





    def ImpedencePosition(self, dx, dy, dz, da, db, dc):
        # time.sleep(0.2)
        # time.sleep(1)
        # print("move")
        EEP_x = self.F_T_EE[0, 3]
        # print(EEP_x)
        EEP_y = self.F_T_EE[1, 3]
        EEP_z = self.F_T_EE[2, 3]
        Euler_angle =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                    Euler_angle[1], 
                                                    Euler_angle[2]
                                                    )
        self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
        # print(self.Position)
        self.currentOrn = self.quat_rr
        self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
        # print(self.Orientation)
        
        target = [self.Position[0]+dx, self.Position[1]+dy, self.Position[2]+dz, 
                  self.Orientation[0]+da, self.Orientation[1]+db, self.Orientation[2]+dc]
        # print(target)
        

        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"# input("\033[33mPress enter to move the robot down to position. \033[0m")
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(target[0], target[1], target[2])
        quat = R.from_euler('xyz', [target[3], target[4], target[5]]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat[0], quat[1], quat[2], quat[3])
        self.eepub.publish(msg)
        time.sleep(0.3)


    def initialrobot(self):
        fh = self.force_history
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                print("!!!!!!!is 0000000")
                # target0 = np.array([0.61, -0.256, 0.2, np.pi, 0, np.pi/2 + np.pi/4])
                target0 = np.array([0.534,0.002,0.0547, np.pi, 0, 0])
                self.MovetoPoint(target0)
                print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                diff = abs (target  - target0)
                if(diff <= 0.003).all():
                    stopf == 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break

    
    
    def MovetoPoint(self, Target):
        time.sleep(1)
        target = Target
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"# input("\033[33mPress enter to move the robot down to position. \033[0m")
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(target[0], target[1], target[2])
        quat = R.from_euler('xyz', [target[3], target[4], target[5]]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat[0], quat[1], quat[2], quat[3])
        self.eepub.publish(msg)
        time.sleep(3)
        # print("success!!!!!!!!!!!!!!!!!!!!!!")

    def franka_callback(self, data):
        # print(data)
        self.K_F_ext_hat_K = np.array(data.K_F_ext_hat_K)
        # print(self.K_F_ext_hat_K)

        # Tip of finger gripper
        self.O_T_EE = np.array(data.O_T_EE).reshape(4, 4).T
        # print(O_T_EE[:3, :3])
        quat_ee = tf.transformations.quaternion_from_matrix(self.O_T_EE)        

        # Flange of robot
        self.F_T_EE_ = np.array(data.F_T_EE).reshape(4, 4).T
        self.F_T_EE_1 = np.asmatrix(self.O_T_EE) * np.linalg.inv(np.asmatrix(self.F_T_EE_))

        # Hand TCP of robot
        self.hand_TCP = np.array([[0.7071, 0.7071, 0, 0],
                                 [-0.7071, 0.7071, 0, 0],
                                 [0, 0, 1, 0.1034],
                                 [0, 0, 0, 1]])
        self.F_T_EE = np.asmatrix(self.F_T_EE_1) * np.asmatrix(self.hand_TCP)

        self.quat_fl_ = tf.transformations.quaternion_from_matrix(self.F_T_EE_1)
        # print(quat_fl)
        self.Euler_fl_ = tf.transformations.euler_from_quaternion(self.quat_fl_)

        # print("self.F_T_EE:", self.F_T_EE)
        self.quat_fl = tf.transformations.quaternion_from_matrix(self.F_T_EE)
        # print(quat_fl)
        self.Euler_fl = tf.transformations.euler_from_quaternion(self.quat_fl)

        return self.K_F_ext_hat_K, self.F_T_EE
        # print("Force:", self.K_F_ext_hat_K)
        # self.ext_force_ee = self.K_F_ext_hat_K[0:2]
        # self.ext_torque_ee = self.K_F_ext_hat_K[3:5]
    

    
    def franka_state_callback(self, msg):
        self.cart_pose_trans_mat = np.asarray(msg.O_T_EE).reshape(4,4,order='F')
        self.cartesian_pose = {
            'position': self.cart_pose_trans_mat[:3,3],
            'orientation': tf.transformations.quaternion_from_matrix(self.cart_pose_trans_mat[:3,:3]) }
        self.franka_fault = self.franka_fault or msg.last_motion_errors.joint_position_limits_violation or msg.last_motion_errors.cartesian_reflex
 

    def reset_arm(self):
        time.sleep(1)
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(0.39, 0, 0.35)
        quat = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat[0], quat[1], quat[2], quat[3])
        # input("\033[33m\nObserve the surroundings. Press enter to move the robot to the initial position.\033[0m")
        self.eepub.publish(msg)
        time.sleep(1)
        print("reset!!!")

    def robot_control_grasptarget(self,target):
        open_gripper()
        # gripper_control(1)
        # print("Gripper opened.")
        time.sleep(1)

        self.grasptarget = target
        self.pre_grasptarget = target + np.array([0,0,0.2,0,0,0])

        self.MovetoPoint(self.pre_grasptarget)
        time.sleep(3)

        stopf = 0
        # fh = self.force_history
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            # print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                # target0 = np.array([0.61, -0.256, 0.2, np.pi, 0, np.pi/2 + np.pi/4])
                target0 = np.array(target)
                self.MovetoPoint(target0)
                # print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                # print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                diff = abs (target[:3]  - target0[:3])
                if(diff <= 0.003).all():
                    stopf == 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break   

        time.sleep(2)
        stopf = 0 
        gripper_control(0)
        # close_gripper()    
        
        # return self.approachfail

    def _initialrobot(self, target):
        open_gripper()
        print("Gripper opened.")
        time.sleep(1)

        self.grasptarget = target
        self.pre_grasptarget = target + np.array([0,0,0.2,0,0,0])

        self.MovetoPoint(self.pre_grasptarget)
        time.sleep(3)

        fh = self.force_history
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                print("!!!!!!!is 0000000")
                # target0 = np.array([0.61, -0.256, 0.2, np.pi, 0, np.pi/2 + np.pi/4])
                target0 = np.array(target)
                self.MovetoPoint(target0)
                print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                diff = abs (target  - target0)
                if(diff <= 0.003).all():
                    stopf == 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
        

    def robot_control_place(self, tz):
        self.grasptarget = np.array([0.577,0.340,tz,np.pi, 0, 0])
        self.pre_grasptarget = self.grasptarget + np.array([0,0,0.2,0,0,0])

        self.MovetoPoint(self.pre_grasptarget)
        time.sleep(1)
        self.MovetoPoint(self.grasptarget)
        time.sleep(1)

        open_gripper()

    def robot_control_place_blue(self):
        self.grasptarget = np.array([0.577,0.270,0.175,np.pi, 0, 0])
        self.pre_grasptarget = self.grasptarget + np.array([0,0,0.3,0,0,0])

        self.MovetoPoint(self.pre_grasptarget)
        time.sleep(1)
        self.MovetoPoint(self.grasptarget)
        time.sleep(1)

        open_gripper()
    


# def gripper_callback(msg):
#     print(msg)
#     width = msg.position[0]
#     print(width)
#     print("gripper!!!!")

#     return width

# def gripper_listener():
#     # # Initialize the ROS node
#     # rospy.init_node('gripper_width_listener', anonymous=True)

#     # # Subscribe to the gripper state topic
#     # rospy.Subscriber("/franka_gripper/joint_states", JointState, gripper_callback)

#     # # Keep the node running
#     # # rospy.spin()
#     print("listener")
def gripper_control(state):
    # print('1111111')
    gripper_state = str(state)
    Command = ['rosrun', 'franka_real_demo', 'gripper_run', gripper_state]
    node_process = subprocess.Popen(Command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
    # print("121212")
    # print(node_process.communicate())
    stdout, stderr = node_process.communicate()
    # print("333333")
    
    message = stdout
    msg = message.strip()
    # print("msg",msg)
    match = re.search(r'\{.*\}', msg)
    # print("match",match)
    if match:
        # print("match")
        try:
            dict_str = match.group(0)
            data_dict = json.loads(dict_str)  
            # print("3333")
        except json.JSONDecodeError as e:
            print(e)
    else:
        print("none!!!!!!!!!!")

    if stderr:
        print("stderr:", stderr)
    gripper_width = data_dict.get("width", [])
    node_process.wait()  

    print("Gripper finish")
    return gripper_width


def open_gripper():
    # gripper_state = str(state)
    node_process = subprocess.Popen(shlex.split('rosrun franka_real_demo gripper_run 1'), stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
    # print("111111")
    # stdout, stderr = node_process.communicate()
    # message = stdout
    # msg = message.strip()
    # # print(msg)        # target_0 = np.array([0.61, -0.166, 0.15, np.pi, 0, np.pi/2 + np.pi/4])
    #     # imp_controller.MovetoPoint(target_0)
    #     # time.sleep(1)
    # match = re.search(r'\{.*\}', msg)
    # # print("2222222")
    # if match:
    #     # print(match)
    #     try:
    #         dict_str = match.group(0)
    #         # print(dict_str)
    #         data_dict = json.loads(dict_str)
    #         # print(data_dict)
    #     except json.JSONDecodeError as e:
    #         print(e)

    # else:
    #     print("none!!!!!!!!!!")

    # if stderr:
    #     print("stderr:", stderr)
    # # print("msg!!!!!!!!!!!!!!!!!!:", data_dict.get("width", []))
    # gripper_width = data_dict.get("width", [])
    # node_process.wait()  # Wait for the command to complete
    # # ImpedencecontrolEnv.gripper_state_callback()

    # print("Gripper Opened")
    # return gripper_width


def close_gripper():
    node_process = subprocess.Popen(shlex.split('rosrun franka_real_demo gripper_run 0'),stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
    # print("111111")
    stdout, stderr = node_process.communicate()
    message = stdout
    msg = message.strip()
    # print(msg)
    match = re.search(r'\{.*\}', msg)
    # print("2222222")
    if match:
        # print(match)
        try:
            dict_str = match.group(0)
            # print(dict_str)
            data_dict = json.loads(dict_str)
            # print(data_dict)
        except json.JSONDecodeError as e:
            print(e)

    else:
        print("none!!!!!!!!!!")

    if stderr:
        print("stderr:", stderr)
    # print("msg!!!!!!!!!!!!!!!!!!:", data_dict.get("width", []))
    gripper_width = data_dict.get("width", [])
    node_process.wait()  # Wait for the command to complete
    print("Gripper Closed")

    return gripper_width

def activate(FLAGS):
    # roscore = subprocess.Popen('roscore')
    time.sleep(1)

    impedence_controller = subprocess.Popen(['roslaunch', 'franka_real_demo', 'impedance.launch',
                                            f'robot_ip:={FLAGS.robot_ip}', f'load_gripper:={FLAGS.load_gripper}'],
                                            stdout=subprocess.PIPE)

    # impedence_controller = subprocess.Popen(['roslaunch', 'serl_franka_controllers', 'impedance.launch',
    #                                         f'robot_ip:={FLAGS.robot_ip}', f'load_gripper:={FLAGS.load_gripper}'],
    #                                         stdout=subprocess.PIPE)
    time.sleep(1)
    rospy.init_node('franka_control_api')

    return impedence_controller

def terminate(impedence_controller):
    impedence_controller.terminate()
    # roscore.terminate()


# def main(_):
#     try:
#         roscore = subprocess.Popen('roscore')
#         time.sleep(1)

#         impedence_controller = subprocess.Popen(['roslaunch', 'serl_franka_controllers', 'impedance.launch',
#                                                 f'robot_ip:={FLAGS.robot_ip}', f'load_gripper:={FLAGS.load_gripper}'],
#                                                 stdout=subprocess.PIPE)
#         time.sleep(1)
#         rospy.init_node('franka_control_api')
                    
#         imp_controller = ImpedencecontrolEnv()
#         imp_controller.client

    
#         imp_controller.reset_arm()
#         time.sleep(1)
#         imp_controller.set_reference_limitation()
#         # # roscore.terminate()

#         # time.sleep(1)
#         # imp_controller.robot_control_grasptarget([0.496, -0.007, 0.035, np.pi, 0, 0])
#         time.sleep(1)
#         _ = open_gripper()
#         time.sleep(1)

#         target = np.array([0.496, -0.007, 0.235, np.pi, 0, 0])

#         imp_controller.MovetoPoint(target)
#         time.sleep(1)

#         _ = close_gripper()
#         time.sleep(1)

#         # rospy.spin()
        
#         # open_gripper()

#         # close_gripper()
#         # # grasp_client(0.04)
#         # time.sleep(2)
#         # gripper_listener()
#         # gripper_callback()
#         # time.sleep(2)
        
        
#         # force = imp_controller.franka_callback()
#         # print(force)
#         # time.sleep(2)
        
#         # target = np.array([0.534,0.002,0.0547, np.pi, 0, 0])

#         # imp_controller.MovetoPoint(target)
#         # time.sleep(1)

#         # target_ = np.array([0.3, 0, 0.28, np.pi, 0, np.pi/2])

#         # imp_controller.MovetoPoint(target_)
#         # time.sleep(1)

#         # # target_0 = np.array([0.61, -0.166, 0.15, np.pi, 0, np.pi/2 + np.pi/4])
#         # # imp_controller.MovetoPoint(target_0)
#         # # time.sleep(1)

#         # imp_controller.ImpedencePosition(0, 0, 0.05, 0, 0, 0)
#         # time.sleep(1)


#         # imp_controller._initialrobot([0.534,0.002,0.0547, np.pi, 0, 0])
#         # time.sleep(0.5)

#         # imp_controller.ImpedencePosition(0, 0, 0.03, 0, 0, 0)

#         target_ = np.array([0.496, -0.007, 0.135, np.pi, 0, 0])

#         imp_controller.MovetoPoint(target_)
#         time.sleep(1)

        


#         impedence_controller.terminate()
#         roscore.terminate()
#         sys.exit()

#     except:
#         rospy.logerr("Error occured. Terminating the controller.")
#         impedence_controller.terminate()
#         roscore.terminate()
#         sys.exit()


# if __name__ == "__main__":
#     app.run(main)


