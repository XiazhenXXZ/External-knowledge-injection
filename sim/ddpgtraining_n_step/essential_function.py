from posixpath import join
import pybullet as p
import numpy as np
import pybullet_data
import math
import os
import gym
import random


def vel_constraint(cur, tar, dv):
    res = []
    for i in range(len(tar)):
        diff = tar[i] - cur[i]
        re = 0
        if abs(diff) > dv:
            if diff > 0:
                re = cur[i] + dv
            else:
                re = cur[i] - dv
        else:
            re = cur[i] + diff
        res.append(re)
    return res


def ptp_control_x(pandaUid, pandaEndEffectorIndex, pandaNumDofs, x):
    currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
    currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
    jointPoses = p.calculateInverseKinematics(
        pandaUid, pandaEndEffectorIndex,
        currentPos + np.array([x, 0, 0]),
        currentOrn)
    print("joint_poses:", jointPoses)
    for i in range(pandaNumDofs):
        p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])
        jointPoses = p.calculateInverseKinematics(
            pandaUid, pandaEndEffectorIndex,
            currentPos + np.array([x, 0, 0]),
            currentOrn)
        print("joint_poses:", jointPoses)


def ptp_control_y(pandaUid, pandaEndEffectorIndex, pandaNumDofs, y):
    currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
    currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
    jointPoses = p.calculateInverseKinematics(
        pandaUid, pandaEndEffectorIndex,
        currentPos + np.array([0, y, 0]),
        currentOrn)
    for i in range(pandaNumDofs):
        p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])


def ptp_control_z(pandaUid, pandaEndEffectorIndex, pandaNumDofs, z):
    currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
    currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
    jointPoses = p.calculateInverseKinematics(
        pandaUid, pandaEndEffectorIndex,
        currentPos + np.array([0, 0, z]),
        currentOrn)
    for i in range(pandaNumDofs):
        p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])


def ptp_control_Alpha(pandaUid, pandaEndEffectorIndex, pandaNumDofs, a):
    currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
    currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
    jointPoses = p.calculateInverseKinematics(
        pandaUid, pandaEndEffectorIndex,
        currentPos,
        p.getQuaternionFromEuler([p.getEulerFromQuaternion(currentOrn)[0] + a, p.getEulerFromQuaternion(currentOrn)[1],
                                  p.getEulerFromQuaternion(currentOrn)[2]]))
    for i in range(pandaNumDofs):
        p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])


def ptp_control_Beta(pandaUid, pandaEndEffectorIndex, pandaNumDofs, b):
    currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
    currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
    jointPoses = p.calculateInverseKinematics(
        pandaUid, pandaEndEffectorIndex,
        currentPos,
        p.getQuaternionFromEuler([p.getEulerFromQuaternion(currentOrn)[0], p.getEulerFromQuaternion(currentOrn)[1] + b,
                                  p.getEulerFromQuaternion(currentOrn)[2]]))
    for i in range(pandaNumDofs):
        p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])


def ptp_control_Gamma(pandaUid, pandaEndEffectorIndex, pandaNumDofs, c):
    currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
    currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
    jointPoses = p.calculateInverseKinematics(
        pandaUid, pandaEndEffectorIndex,
        currentPos,
        p.getQuaternionFromEuler([p.getEulerFromQuaternion(currentOrn)[0], p.getEulerFromQuaternion(currentOrn)[1],
                                  p.getEulerFromQuaternion(currentOrn)[2] + c]))
    for i in range(pandaNumDofs):
        p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])

def ptp_controller_translation(pandaUid, pandaEndEffectorIndex, pandaNumDofs, x, y ,z):
    currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
    currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
    jointPoses = p.calculateInverseKinematics(
        pandaUid, pandaEndEffectorIndex,
        currentPos + np.array([x, y, z]),
        currentOrn)
    # print("joint_poses:", jointPoses)
    for i in range(pandaNumDofs):
        p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])
        jointPoses = p.calculateInverseKinematics(
            pandaUid, pandaEndEffectorIndex,
            currentPos + np.array([x, y, z]),
            currentOrn)
        # print("joint_poses:", jointPoses)

def ptp_controller_rotation(pandaUid, pandaEndEffectorIndex, pandaNumDofs, r, p ,y):
    currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
    currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
    jointPoses = p.calculateInverseKinematics(
        pandaUid, pandaEndEffectorIndex,
        currentPos,
        p.getQuaternionFromEuler([p.getEulerFromQuaternion(currentOrn)[0]+r, p.getEulerFromQuaternion(currentOrn)[1] + p,
                                  p.getEulerFromQuaternion(currentOrn)[2]+y]))
    for i in range(pandaNumDofs):
        p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])

