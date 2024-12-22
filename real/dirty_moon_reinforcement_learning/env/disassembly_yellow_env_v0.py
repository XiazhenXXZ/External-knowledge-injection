#!/usr/bin/env python3

import os
import sys
import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time

import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import franka_msgs.msg
from franka_gripper.msg import MoveGoal, MoveAction, GraspAction, GraspGoal
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import actionlib
from math import pi
import tf
import tf.transformations

from std_msgs.msg import String
# from moveit_commander.conversions import pose_to_list
# from geometry_msgs.msg import Wrench
from gym import error, spaces, utils
from gym.utils import seeding


def all_close(goal, actual, tolerance):
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(moveit_commander.conversions.pose_to_list(goal), moveit_commander.conversions.pose_to_list(actual), tolerance)

  return True

class DisassemblyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DisassemblyEnv, self).__init__()
        ## Initialize `moveit_commander`_ and a `rospy`_ node:S
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('dirty_moon_disassembly',
                        anonymous=True,log_level=rospy.INFO)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        
        group_name = "panda_arm"
        group = moveit_commander.MoveGroupCommander(group_name)
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)
        planning_frame = group.get_planning_frame()
        eef_link = group.get_end_effector_link()
        group_names = robot.get_group_names()

        # Misc variables
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.group = group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.data = []
        self.joint_effort=[] 

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

        # sub and pub
        self.sub = rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.convert_to_geometry_msg, queue_size=1, tcp_nodelay=True)
        self.pub_eeff        = rospy.Publisher("/franka_state_controller/O_T_EE", geometry_msgs.msg.PoseStamped, queue_size=1)
        self.pub_ee_pose     = rospy.Publisher("/franka_state_controller/ee_pose", geometry_msgs.msg.Pose, queue_size=1)
        self.pub_eeff_flange = rospy.Publisher("/franka_state_controller/O_T_FL", geometry_msgs.msg.PoseStamped, queue_size=1)
        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.franka_callback)
        self.error_recovery_pub = rospy.Publisher('/franka_control/error_recovery/goal', franka_msgs.msg.ErrorRecoveryActionGoal, queue_size=1)
        
        # ====== observation ======#
        min_EEP_x = 0  # EEP = End Effector Position
        min_EEP_y = -math.inf
        min_EEP_z = 0
        max_EEP_x = math.inf
        max_EEP_y = 1
        max_EEP_z = math.inf

        min_Ori_r = 1.3  # Ori = orientation
        min_Ori_p = -1.2
        min_Ori_y = -math.inf
        max_Ori_r = 1.8
        max_Ori_p = 1.2
        max_Ori_y = math.inf

        min_F_x = -math.inf
        min_F_y = -math.inf
        min_F_z = -math.inf
        max_F_x = math.inf
        max_F_y = math.inf
        max_F_z = math.inf

        min_T_x = -math.inf
        min_T_y = -math.inf
        min_T_z = -math.inf
        max_T_x = math.inf
        max_T_y = math.inf
        max_T_z = math.inf

        # Observation_space
        # self.low = np.array([[min_EEP_x, min_EEP_y, min_EEP_z],
        #                      [min_Ori_r, min_Ori_p, min_Ori_y],
        #                      [min_F_x, min_F_y, min_F_z],
        #                      [min_T_x, min_T_y, min_T_z]]).astype(np.float32)
        # self.high = np.array([[max_EEP_x, max_EEP_y, max_EEP_z],
        #                       [max_Ori_r, max_Ori_p, max_Ori_y],
        #                       [max_F_x, max_F_y, max_F_z],
        #                       [max_T_x, max_T_y, max_T_z]]).astype(np.float32)
        self.low = np.array([min_EEP_x, 
                             min_EEP_y, 
                             min_EEP_z,
                             min_Ori_r, 
                             min_Ori_p, 
                             min_Ori_y,
                             min_F_x, 
                             min_F_y, 
                             min_F_z,
                             min_T_x, 
                             min_T_y, 
                             min_T_z]).astype(np.float32)
        self.high = np.array([max_EEP_x, 
                              max_EEP_y, 
                              max_EEP_z,
                              max_Ori_r, 
                              max_Ori_p, 
                              max_Ori_y,
                              max_F_x, 
                              max_F_y, 
                              max_F_z,
                              max_T_x, 
                              max_T_y, 
                              max_T_z]).astype(np.float32)
        
        # Action_space
        self.action_space = gym.spaces.Discrete(12)
        self.observation_space = gym.spaces.Box(self.low, self.high)

        self.step_count = 0
        
        self.z_ = 0
        self.x_ = 0
        self.y_ = 0



    # Initial_position 
    def go_to_joint_state(self):
        robot = self.robot
        group = self.group
        joint_goal = group.get_current_joint_values()
        joint_goal[0] = 1.5590904907476133
        joint_goal[1] = -1.5639568416691758
        joint_goal[2] = -1.857166823143343 
        joint_goal[3] = -1.795508457604208
        joint_goal[4] = -0.0716343073002344
        joint_goal[5] = 1.7507461156647195
        joint_goal[6] = 2.0505296631584193
        group.go(joint_goal, wait=True)
        group.stop()
        current_joints = self.group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    # Grasp
    def grasp_client(self):
        client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)

        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()

        # Creates a goal to send to the action server.
        goal = GraspGoal()
        goal.width = 0.04
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.005
        goal.speed = 0.1
        goal.force = 5

        # Sends the goal to the action server.
        client.send_goal(goal)

        # Waits for the server to finish performing the action.
        client.wait_for_result()
        time.sleep(1)
        # Prints out the result of executing the action
        return client.get_result()  # A GraspResult

    # Release gripper
    def release(self):
        release_action = actionlib.SimpleActionClient(
                "/franka_gripper/gripper_action",
                GripperCommandAction,
            )
        release_action.wait_for_server()
        goal = GripperCommandGoal()
        goal.command.position = 0.035
        goal.command.max_effort = 0.0

        release_action.send_goal(goal)
        release_action.wait_for_result()

    # Initial_robot
    def initial_robot(self):
        self.release()
        self.go_to_joint_state()
        time.sleep(3)
        # for i in range(5):
        #     if i <=5:
        #         self.ptp_control_0()
        #         # tutorial.franka_callback()

        #     else:
        #         self.ee_franka()
        self.ptp_control_0()
        self.grasp_client()
        self.x_ = self.F_T_EE[0, 3]
        self.y_ = self.F_T_EE[1, 3]
        self.z_ = self.F_T_EE[2, 3]

    def convert_to_geometry_msg(self, state_msg):
        # Tip of finger gripper
        self.O_T_EE = np.array(state_msg.O_T_EE).reshape(4, 4).T
        # print(O_T_EE[:3, :3])
        quat_ee = tf.transformations.quaternion_from_matrix(self.O_T_EE)        

        # Flange of robot
        self.F_T_EE_ = np.array(state_msg.F_T_EE).reshape(4, 4).T
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

        return self.F_T_EE
    
    def ptp_control_0(self):
        self.F_T_EE_1[1, 3] += 0.109
        self.pos = self.F_T_EE_1
        Euler_angle =list(self.Euler_fl_)
        Euler_angle[0] += 0
        Euler_angle[1] += 0
        Euler_angle[2] += 0
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                                Euler_angle[1], 
                                                                Euler_angle[2])
        pos = self.pos
        quat_rr = self.quat_rr
        group = self.group
        
        ## We can plan a motion for this group to a desired pose for the end-effector:
        pose_goal_0 = geometry_msgs.msg.Pose()
        pose_goal_0.orientation.x = quat_rr[0]
        pose_goal_0.orientation.y = quat_rr[1]
        pose_goal_0.orientation.z = quat_rr[2]
        pose_goal_0.orientation.w = quat_rr[3]
        pose_goal_0.position.x = pos[0, 3]
        pose_goal_0.position.y = pos[1, 3]
        pose_goal_0.position.z = pos[2, 3]

        group.set_pose_target(pose_goal_0)
        group.go(wait=True)
        
        group.stop()
        group.clear_pose_targets()

        current_pose = self.group.get_current_pose().pose
        time.sleep(1)
        return all_close(pose_goal_0, current_pose, 0.01)
    
    # x negative
    def ptp_control_1(self, scale=1):
        group = self.group
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.position.x -= scale * 0.001 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)
    
    #x positive
    def ptp_control_2(self, scale=1):
        group = self.group
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.position.x += scale * 0.001 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)
    
    # y negative
    def ptp_control_3(self, scale=1):
        group = self.group
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.position.y -= scale * 0.001 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)
    
    # y positive
    def ptp_control_4(self, scale=1):
        group = self.group
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.position.y += scale * 0.001 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)
    
    # z negative
    def ptp_control_5(self, scale=1):
        group = self.group
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.position.z -= scale * 0.001 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)
    
    # z positive
    def ptp_control_6(self, scale=1):
        group = self.group
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.position.z += scale * 0.001 
            
                # wpose.orientation.x = 0.001
        waypoints.append(copy.deepcopy(wpose))
        
        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
                # print("waypoints:", waypoints)
                # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)

    # r negative
    def ptp_control_7(self, scale=1):
        group = self.group
        Euler_angle =list(self.Euler_fl_)
        Euler_angle[0] -= scale * 0.001
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                                Euler_angle[1], 
                                                                Euler_angle[2])
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.orientation.x = self.quat_rr[0]
        wpose.orientation.y = self.quat_rr[1]
        wpose.orientation.z = self.quat_rr[2]
        wpose.orientation.w = self.quat_rr[3] 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)

    # r positive
    def ptp_control_8(self, scale=1):
        group = self.group
        Euler_angle =list(self.Euler_fl_)
        Euler_angle[0] += scale * 0.001
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                                Euler_angle[1], 
                                                                Euler_angle[2])
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.orientation.x = self.quat_rr[0]
        wpose.orientation.y = self.quat_rr[1]
        wpose.orientation.z = self.quat_rr[2]
        wpose.orientation.w = self.quat_rr[3] 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)

    # p negative
    def ptp_control_9(self, scale=1):
        group = self.group
        Euler_angle =list(self.Euler_fl_)
        Euler_angle[1] -= scale * 0.001
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                                Euler_angle[1], 
                                                                Euler_angle[2])
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.orientation.x = self.quat_rr[0]
        wpose.orientation.y = self.quat_rr[1]
        wpose.orientation.z = self.quat_rr[2]
        wpose.orientation.w = self.quat_rr[3] 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)

    # p positive
    def ptp_control_10(self, scale=1):
        group = self.group
        Euler_angle =list(self.Euler_fl_)
        Euler_angle[1] += scale * 0.001
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                                Euler_angle[1], 
                                                                Euler_angle[2])
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.orientation.x = self.quat_rr[0]
        wpose.orientation.y = self.quat_rr[1]
        wpose.orientation.z = self.quat_rr[2]
        wpose.orientation.w = self.quat_rr[3] 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)
    
    # y negative
    def ptp_control_11(self, scale=1):
        group = self.group
        Euler_angle =list(self.Euler_fl_)
        Euler_angle[2] -= scale * 0.001
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                                Euler_angle[1], 
                                                                Euler_angle[2])
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.orientation.x = self.quat_rr[0]
        wpose.orientation.y = self.quat_rr[1]
        wpose.orientation.z = self.quat_rr[2]
        wpose.orientation.w = self.quat_rr[3] 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)
    
    # y positive
    def ptp_control_12(self, scale=1):
        group = self.group
        Euler_angle =list(self.Euler_fl_)
        Euler_angle[2] += scale * 0.001
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                                Euler_angle[1], 
                                                                Euler_angle[2])
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through:
        waypoints = []

        wpose = group.get_current_pose().pose
        wpose.orientation.x = self.quat_rr[0]
        wpose.orientation.y = self.quat_rr[1]
        wpose.orientation.z = self.quat_rr[2]
        wpose.orientation.w = self.quat_rr[3] 
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = group.compute_cartesian_path(
                                        waypoints,   # waypoints to follow
                                        0.001,        # eef_step
                                        0.0)         # jump_threshold
        # print("waypoints:", waypoints)
        # print("wpose:", wpose)
        group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        current_pose = self.group.get_current_pose().pose
        time.sleep(1)

        return all_close(plan, current_pose, 0.01)

    # position_x_negative
    # def ptp_control_1(self):
    #     self.F_T_EE[0, 3] -= 0.001
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[0] += 0
    #     Euler_angle[1] += 0
    #     Euler_angle[2] += 0
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_1 = geometry_msgs.msg.Pose()
    #     pose_goal_1.orientation.x = quat_rr[0]
    #     pose_goal_1.orientation.y = quat_rr[1]
    #     pose_goal_1.orientation.z = quat_rr[2]
    #     pose_goal_1.orientation.w = quat_rr[3]
    #     pose_goal_1.position.x = pos[0, 3]
    #     pose_goal_1.position.y = pos[1, 3]
    #     pose_goal_1.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_1)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_1, current_pose, 0.01)

    # # position_x_positive
    # def ptp_control_2(self):
    #     self.F_T_EE[0, 3] += 0.001
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[0] += 0
    #     Euler_angle[1] += 0
    #     Euler_angle[2] += 0
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_2 = geometry_msgs.msg.Pose()
    #     pose_goal_2.orientation.x = quat_rr[0]
    #     pose_goal_2.orientation.y = quat_rr[1]
    #     pose_goal_2.orientation.z = quat_rr[2]
    #     pose_goal_2.orientation.w = quat_rr[3]
    #     pose_goal_2.position.x = pos[0, 3]
    #     pose_goal_2.position.y = pos[1, 3]
    #     pose_goal_2.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_2)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_2, current_pose, 0.01)

    # # position_y_negative
    # def ptp_control_3(self):
    #     self.F_T_EE[1, 3] -= 0.001
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[0] += 0
    #     Euler_angle[1] += 0
    #     Euler_angle[2] += 0
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_3 = geometry_msgs.msg.Pose()
    #     pose_goal_3.orientation.x = quat_rr[0]
    #     pose_goal_3.orientation.y = quat_rr[1]
    #     pose_goal_3.orientation.z = quat_rr[2]
    #     pose_goal_3.orientation.w = quat_rr[3]
    #     pose_goal_3.position.x = pos[0, 3]
    #     pose_goal_3.position.y = pos[1, 3]
    #     pose_goal_3.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_3)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_3, current_pose, 0.01)

    # # position_y_positive
    # def ptp_control_4(self):
    #     self.F_T_EE[1, 3] += 0.001
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[0] += 0
    #     Euler_angle[1] += 0
    #     Euler_angle[2] += 0
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_4 = geometry_msgs.msg.Pose()
    #     pose_goal_4.orientation.x = quat_rr[0]
    #     pose_goal_4.orientation.y = quat_rr[1]
    #     pose_goal_4.orientation.z = quat_rr[2]
    #     pose_goal_4.orientation.w = quat_rr[3]
    #     pose_goal_4.position.x = pos[0, 3]
    #     pose_goal_4.position.y = pos[1, 3]
    #     pose_goal_4.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_4)415
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_4, current_pose, 0.01)


    # # position_z_negative
    # def ptp_control_5(self):
    #     self.F_T_EE[2, 3] -= 0.001
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[0] += 0
    #     Euler_angle[1] += 0
    #     Euler_angle[2] += 0
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_5 = geometry_msgs.msg.Pose()
    #     pose_goal_5.orientation.x = quat_rr[0]
    #     pose_goal_5.orientation.y = quat_rr[1]
    #     pose_goal_5.orientation.z = quat_rr[2]
    #     pose_goal_5.orientation.w = quat_rr[3]
    #     pose_goal_5.position.x = pos[0, 3]
    #     pose_goal_5.position.y = pos[1, 3]
    #     pose_goal_5.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_5)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_5, current_pose, 0.01)

    # # position_z_positive
    # def ptp_control_6(self):
    #     self.F_T_EE[2, 3] += 0.005
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[0] += 0
    #     Euler_angle[1] += 0
    #     Euler_angle[2] += 0
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_6 = geometry_msgs.msg.Pose()
    #     pose_goal_6.orientation.x = quat_rr[0]
    #     pose_goal_6.orientation.y = quat_rr[1]
    #     pose_goal_6.orientation.z = quat_rr[2]
    #     pose_goal_6.orientation.w = quat_rr[3]
    #     pose_goal_6.position.x = pos[0, 3]
    #     pose_goal_6.position.y = pos[1, 3]
    #     pose_goal_6.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_6)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_6, current_pose, 0.01)
    
    # Euler_angle_r_negative
    # def ptp_control_7(self):
    #     self.F_T_EE[0, 3] += 0
    #     self.F_T_EE[1, 3] += 0
    #     self.F_T_EE[2, 3] += 0
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[0] -= 0.001
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_7 = geometry_msgs.msg.Pose()
    #     pose_goal_7.orientation.x = quat_rr[0]
    #     pose_goal_7.orientation.y = quat_rr[1]
    #     pose_goal_7.orientation.z = quat_rr[2]
    #     pose_goal_7.orientation.w = quat_rr[3]
    #     pose_goal_7.position.x = pos[0, 3]
    #     pose_goal_7.position.y = pos[1, 3]
    #     pose_goal_7.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_7)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_7, current_pose, 0.01)

    # # Euler_angle_r_positive
    # def ptp_control_8(self):
    #     self.F_T_EE[0, 3] += 0
    #     self.F_T_EE[1, 3] += 0
    #     self.F_T_EE[2, 3] += 0
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[0] += 0.001
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_8 = geometry_msgs.msg.Pose()
    #     pose_goal_8.orientation.x = quat_rr[0]
    #     pose_goal_8.orientation.y = quat_rr[1]
    #     pose_goal_8.orientation.z = quat_rr[2]
    #     pose_goal_8.orientation.w = quat_rr[3]
    #     pose_goal_8.position.x = pos[0, 3]
    #     pose_goal_8.position.y = pos[1, 3]
    #     pose_goal_8.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_8)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_8, current_pose, 0.01)

    # # Euler_angle_p_negative
    # def ptp_control_9(self):
    #     self.F_T_EE[0, 3] += 0
    #     self.F_T_EE[1, 3] += 0
    #     self.F_T_EE[2, 3] += 0
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[1] -= 0.001
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_9 = geometry_msgs.msg.Pose()
    #     pose_goal_9.orientation.x = quat_rr[0]
    #     pose_goal_9.orientation.y = quat_rr[1]
    #     pose_goal_9.orientation.z = quat_rr[2]
    #     pose_goal_9.orientation.w = quat_rr[3]
    #     pose_goal_9.position.x = pos[0, 3]
    #     pose_goal_9.position.y = pos[1, 3]
    #     pose_goal_9.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_9)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_9, current_pose, 0.01)

    # # Euler_angle_p_positive
    # def ptp_control_10(self):
    #     self.F_T_EE[0, 3] += 0
    #     self.F_T_EE[1, 3] += 0
    #     self.F_T_EE[2, 3] += 0
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[1] += 0.001
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_10 = geometry_msgs.msg.Pose()
    #     pose_goal_10.orientation.x = quat_rr[0]
    #     pose_goal_10.orientation.y = quat_rr[1]
    #     pose_goal_10.orientation.z = quat_rr[2]
    #     pose_goal_10.orientation.w = quat_rr[3]
    #     pose_goal_10.position.x = pos[0, 3]
    #     pose_goal_10.position.y = pos[1, 3]
    #     pose_goal_10.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_10)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_10, current_pose, 0.01)
    
    # # Euler_angle_y_negative
    # def ptp_control_11(self):
    #     self.F_T_EE[0, 3] += 0
    #     self.F_T_EE[1, 3] += 0
    #     self.F_T_EE[2, 3] += 0
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[2] -= 0.001
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_11 = geometry_msgs.msg.Pose()
    #     pose_goal_11.orientation.x = quat_rr[0]
    #     pose_goal_11.orientation.y = quat_rr[1]
    #     pose_goal_11.orientation.z = quat_rr[2]
    #     pose_goal_11.orientation.w = quat_rr[3]
    #     pose_goal_11.position.x = pos[0, 3]
    #     pose_goal_11.position.y = pos[1, 3]
    #     pose_goal_11.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_11)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_11, current_pose, 0.01)

    # # Euler_angle_y_positive
    # def ptp_control_12(self):
    #     self.F_T_EE[0, 3] += 0
    #     self.F_T_EE[1, 3] += 0
    #     self.F_T_EE[2, 3] += 0
    #     self.pos = self.F_T_EE
    #     Euler_angle =list(self.Euler_fl)
    #     Euler_angle[2] += 0.001
    #     self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
    #                                                             Euler_angle[1], 
    #                                                             Euler_angle[2])
    #     pos = self.pos
    #     quat_rr = self.quat_rr
    #     group = self.group
        
    #     ## We can plan a motion for this group to a desired pose for the end-effector:
    #     pose_goal_12 = geometry_msgs.msg.Pose()
    #     pose_goal_12.orientation.x = quat_rr[0]
    #     pose_goal_12.orientation.y = quat_rr[1]
    #     pose_goal_12.orientation.z = quat_rr[2]
    #     pose_goal_12.orientation.w = quat_rr[3]
    #     pose_goal_12.position.x = pos[0, 3]
    #     pose_goal_12.position.y = pos[1, 3]
    #     pose_goal_12.position.z = pos[2, 3]

    #     group.set_pose_target(pose_goal_12)
    #     group.go(wait=True)
        
    #     group.stop()
    #     group.clear_pose_targets()

    #     current_pose = self.group.get_current_pose().pose
    #     time.sleep(1)
    #     return all_close(pose_goal_12, current_pose, 0.01)

    # def state_machine_callback(self, msg):
    #     self.sm_state = msg.data

    def franka_callback(self, data):
        self.K_F_ext_hat_K = np.array(data.K_F_ext_hat_K)
        # self.ext_force_ee = self.K_F_ext_hat_K[0:2]
        # self.ext_torque_ee = self.K_F_ext_hat_K[3:5]
        return self.K_F_ext_hat_K
    
    def franka_state_callback(self, msg):
        self.cart_pose_trans_mat = np.asarray(msg.O_T_EE).reshape(4,4,order='F')
        self.cartesian_pose = {
            'position': self.cart_pose_trans_mat[:3,3],
            'orientation': tf.transformations.quaternion_from_matrix(self.cart_pose_trans_mat[:3,:3]) }
        self.franka_fault = self.franka_fault or msg.last_motion_errors.joint_position_limits_violation or msg.last_motion_errors.cartesian_reflex

    
    def reset_communication_error(self):
        err_recovery_msg = franka_msgs.msg.ErrorRecoveryActionGoal()
        if self.franka_fault:
            self.error_recovery_pub.publish(err_recovery_msg)
            self.franka_fault = False

    def step(self, action):
        self.step_count += 1
        if action == 0:
            self.ptp_control_1()
        elif action == 1:
            self.ptp_control_2()
        elif action == 2:
            self.ptp_control_3()
        elif action == 3:
            self.ptp_control_4()
        elif action == 4:
            self.ptp_control_5()
        # elif action == 5:
        elif action == 5:
            self.ptp_control_6()
        elif action == 6:
            self.ptp_control_7()
        elif action == 7:
            self.ptp_control_8()
        elif action == 8:
            self.ptp_control_9()
        elif action == 9:
            self.ptp_control_10()
        elif action == 10:
            self.ptp_control_11()
        else:
            self.ptp_control_12()

        # End effector position
        EEP_x = self.F_T_EE[0, 3]
        EEP_y = self.F_T_EE[1, 3]
        EEP_z = self.F_T_EE[2, 3]

        
        # Flange position
        EEP_x_1 = self.F_T_EE_1[0, 3]
        EEP_y_1 = self.F_T_EE_1[1, 3]
        EEP_z_1 = self.F_T_EE_1[2, 3]
        
        self.z = EEP_z
        print("z:", self.z)
        self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
        self.currentOrn = self.quat_rr
        self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
        self.Force = self.K_F_ext_hat_K[0:3]
        print("Force:", self.Force)
        self.Torque = self.K_F_ext_hat_K[3:6]
        self.obs = [EEP_x,
                    EEP_y,
                    EEP_z,
                    self.Orientation[0],
                    self.Orientation[1],
                    self.Orientation[2],
                    self.Force[0],
                    self.Force[1],
                    self.Force[2],
                    self.Torque[0],
                    self.Torque[1],
                    self.Torque[2]]
        
        
        self.reward = 5 * (self.z-self.z_)
        self.reward = -1 * abs(EEP_x - self.x_)
        self.reward = -1 * abs(EEP_y - self.y_)
        # self.reward += -3 * (EEP_y - 0.02)
        # self.reward += 3 * (EEP_x / 0.5)
        # print("Force in training:", self.Force)
        # print("Torque in training:", self.Torque)

        self.z_ = self.z
        self.x_ = EEP_x
        self.y_ = EEP_y
        # self.reward += -3 * (EEP_y - 0.02)
        # self.reward += 3 * (EEP_x / 0.5)
        # print("Force in training:", self.Force)
        # print("Torque in training:", self.Torque)
        if self.z > 0.225:
            self.terminated = True
            self.reward += 10
            print("Success")

        # elif EEP_x >= 0.6 or EEP_x <= 0.4:
        #     self.done = True

        elif self.z <= 0.163:
            self.terminated = True
            self.reward -= 10
            print("Fail")
            # print("Fail_Force", self.Force)
            # print("Fail_Torque", self.Torque)

        # elif EEP_x >= 0.7:
        #     self.terminated = True
        #     self.reward -= 50
        #     print("Fail")
            # print("Fail_Force", self.Force)
            # print("Fail_Torque", self.Torque)

        else:
            self.terminated = False

        self.info = {}
        self.truncated = False

        return np.array(self.obs).astype(np.float32), self.reward, self.terminated, self.truncated, self.info

    def reset(self):
        self.step_count = 0
        # End effector position
        EEP_x = self.F_T_EE[0, 3]
        EEP_y = self.F_T_EE[1, 3]
        EEP_z = self.F_T_EE[2, 3]
        self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
        self.currentOrn = self.quat_rr
        self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
        self.Force = self.K_F_ext_hat_K[0:3]
        self.Torque = self.K_F_ext_hat_K[3:6]
        observation = [EEP_x,
                       EEP_y,
                       EEP_z,
                       self.Orientation[0],
                       self.Orientation[1],
                       self.Orientation[2],
                       self.Force[0],
                       self.Force[1],
                       self.Force[2],
                       self.Torque[0],
                       self.Torque[1],
                       self.Torque[2]]
        self.initial_robot()
        return np.array(observation).astype(np.float32)

if __name__ == "__main__":
    env = DisassemblyEnv()
    env.reset()
    # env.initial_robot()
    for step in range(50):
        action = env.action_space.sample()
        print(action)
        obs, reward, info, done = env.step(action)
        print(reward)
        if done is True:
            env.reset()
