import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import gym
import time

from gym import error, spaces, utils
from gym.utils import seeding
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial import distance
from constants_disassembly import *
from essential_function import *
from position_controller import *


class DisassemblyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        p.connect(p.GUI)
        self.step_counter = 0
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])
        # self.al = np.array([0, 0]).astype(np.float32)
        # self.ah = np.array([np.pi, np.pi]).astype(np.float32)
        # self.low = np.array([[min_EEP_x, min_EEP_y, min_EEP_z],
        #                      [min_Ori_r, min_Ori_p, min_Ori_y],
        #                      [min_F_x, min_F_y, min_F_z],
        #                      [min_T_x, min_T_y, min_T_z]]).astype(np.float32)
        # self.high = np.array([[max_EEP_x, max_EEP_y, max_EEP_z],
        #                       [max_Ori_r, max_Ori_p, max_Ori_y],
        #                       [max_F_x, max_F_y, max_F_z],
        #                       [max_T_x, max_T_y, max_T_z]]).astype(np.float32)
        # self.action_space = spaces.Box(self.al, self.ah)
        # self.observation_space = spaces.Box(self.low, self.high)

        self.action_space = spaces.Box(
            np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]).astype(np.float32),
            np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).astype(np.float32)
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
                             min_x, min_y, min_z,
                             min_r, min_p, min_y_,
                             min_D1, min_O1]).astype(np.float32)
        self.high = np.array([max_EEP_x, max_EEP_y, max_EEP_z,
                              max_Ori_r, max_Ori_p, max_Ori_y,
                              max_x, max_y, max_z,
                              max_r, max_p, max_y_,
                              max_D1, max_O1]).astype(np.float32)

        self.observation_space = spaces.Box(self.low, self.high)
        self.step_count = 0
        self.pandaEndEffectorIndex = 11
        self.pandaNumDofs = 7
        # p.setTimeStep(0.01)
        self.PC = PositionController()

    def grasp_process(self, state, target):
        currentPos = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0]
        target_pos = vel_constraint(currentPos, target, 0.06)
        # targetOrn = targetOrn = p.getEulerFromQuaternion(targetOrn)

        # gripper width init
        if state == 0:
            for i in [9, 10]:
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, 0.02, force=20)

        # gripper moving to grasping-target x,z
        if state == 1:
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex,
                target_pos + np.array([0, 0.2, 0]),
                p.getQuaternionFromEuler([math.pi / 2, 0, 0]))
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5. * 240.)

        # gripper moving to grasping-target y
        if state == 2:
            jointPoses = p.calculateInverseKinematics(
                self.pandaUid, self.pandaEndEffectorIndex,
                target_pos,
                p.getQuaternionFromEuler([math.pi / 2, 0, 0])
            )
            for i in range(self.pandaNumDofs):
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, jointPoses[i], force=5. * 240.)

        if state == 3:
            for i in [9, 10]:
                p.setJointMotorControl2(self.pandaUid, i, p.POSITION_CONTROL, 0.0006, force=20000)

    def update_state(self):
        # self.state_t += self.timeStep
        if self.state_t > self.state_durations[self.state]:
            self.state += 1
            self.state_t = 0
            if self.state >= len(self.state_durations):
                self.state = 0

        if self.state_t < self.state_durations[self.state]:
            self.state_t += self.timeStep

    def initial_robot(self):
        while True:
            self.update_state()

            targetP = [0.5, 0.018, 0.13]
            self.grasp_process(self.state, targetP)
            p.stepSimulation()
            if self.state == 3:
                break

    def step(self, action: np.ndarray):
        print(action)
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        position_, orientation_ = p.getBasePositionAndOrientation(self.barUid)
        # self.z_ = p.getBasePositionAndOrientation(self.barUid)[0][2]
        self.z_ = 0

        EEP_x_ = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0][0]
        EEP_y_ = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0][1]
        EEP_z_ = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0][2]

        self.step_count += 1

        # Actions in delta_pos
        # self.PC.pc_pos(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, action[0], action[1], action[2])
        # # Actions in delta_ori
        # self.PC.pc_ori(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, action[3], action[4], action[5])
        # self.PC.pc(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, 0, 0, 0.02, 0, 0, 0)
        self.PC.pc(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, action[0], action[1], action[2],action[3], action[4], action[5])

        # if action == 0:
        #    essential_function.ptp_control_x(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, 0.02)
        # elif action == 1:
        #    essential_function.ptp_control_y(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, 0.02)
        # elif action == 2:
        #    essential_function.ptp_control_z(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, 0.02)
        # elif action == 3:
        #    essential_function.ptp_control_Alpha(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs,
        #                                         0.02)
        # elif action == 4:
        #    essential_function.ptp_control_Beta(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, 0.02)
        # elif action == 5:
        #    essential_function.ptp_control_Gamma(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs,
        #                                         0.02)
        # elif action == 6:
        #    essential_function.ptp_control_x(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, -0.02)
        # elif action == 7:
        #    essential_function.ptp_control_y(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, -0.02)
        # elif action == 8:
        #    essential_function.ptp_control_z(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, -0.02)
        # elif action == 9:
        #    essential_function.ptp_control_Alpha(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs,
        #                                         -0.02)
        # elif action == 10:
        #    essential_function.ptp_control_Beta(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs, -0.02)
        # else:
        #    essential_function.ptp_control_Gamma(self.pandaUid, self.pandaEndEffectorIndex, self.pandaNumDofs,
        #                                         -0.02)

        p.stepSimulation()
        self.barUid = p.loadURDF("bar/bar.urdf", basePosition=[0.5, 0, 1])
        self.x = p.getBasePositionAndOrientation(self.barUid)[0][0]
        self.y = p.getBasePositionAndOrientation(self.barUid)[0][1]
        self.z = p.getBasePositionAndOrientation(self.barUid)[0][2]
        self.rpy = p.getBasePositionAndOrientation(self.barUid)[1]
        self.rpy_ = np.array(p.getEulerFromQuaternion(self.rpy), dtype=float)
        self.r = self.rpy_[0]
        self.p = self.rpy_[1]
        self.y_ = self.rpy_[2]
        # self.x = p.getBasePositionAndOrientation(self.barUid)[0][0]
        # self.y = p.getBasePositionAndOrientation(self.barUid)[0][1]
        # self.z = p.getBasePositionAndOrientation(self.barUid)[0][2]
        # self.r = p.getBasePositionAndOrientation(self.barUid)[1][0]
        # self.p = p.getBasePositionAndOrientation(self.barUid)[1][1]
        # self.y_ = p.getBasePositionAndOrientation(self.barUid)[1][2]


        print("z:", self.z)
        self.Position = np.array(p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0], dtype=float)
        self.currentOrn = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[1]
        self.Orientation = np.array(p.getEulerFromQuaternion(self.currentOrn), dtype=float)
        p.enableJointForceTorqueSensor(self.pandaUid, 8, 1)
        self.Force = np.array(p.getJointState(self.pandaUid, 8)[2][0:3], dtype=float)

        Fx = self.Force[0]
        Fy = self.Force[1]
        Fz = self.Force[2]
        resultant_force = math.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)
        # print('Force:', resultant_force)
        self.Torque = np.array(p.getJointState(self.pandaUid, 8)[2][3:6], dtype=float)

        self.D1 = distance.euclidean(p.getBasePositionAndOrientation(self.barUid)[0], self.Position)
        self.O1 = distance.euclidean(self.rpy_, self.Orientation)

        # self.obs = [self.Position[0], self.Position[1], self.Position[2],
        #             self.Orientation[0], self.Orientation[1], self.Orientation[2],
        #             self.Force[0], self.Force[1], self.Force[2],
        #             self.Torque[0], self.Torque[1], self.Torque[2]]
        self.obs = [self.Position[0], self.Position[1], self.Position[2],
                    self.Orientation[0], self.Orientation[1], self.Orientation[2],
                    self.x, self.y, self.z, self.r, self.p, self.y_, self.D1, self.O1]

        # End effector position
        EEP_x = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0][0]
        # print(EEP_x)
        EEP_y = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0][1]
        # print(EEP_y)
        EEP_z = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0][2]
        # print(EEP_z)

        position, orientation = p.getBasePositionAndOrientation(self.barUid)

        self.reward = 0
        # print("z:", self.z)
        self.reward += 0.5 * (self.z - 0.15)
        # self.reward += 5 * (EEP_z - EEP_z_)

        self.reward += 0.5 * (EEP_z - 0.15)
        self.reward += -0.2 * abs(EEP_y - 0.02)
        self.reward += -0.2 * abs(EEP_x - 0.5)

        # print("z reward = ", 5 * (EEP_z - EEP_z_))
        # print("x reward = ", -0.5 * abs(EEP_x - EEP_x_))
        # print("y reward = ", -0.5 * abs(EEP_y - EEP_y_))
        # self.reward += 10000 * self.z - 1300
        # self.reward += -0.5 * abs(EEP_y - EEP_y_)
        # self.reward += -0.5 * abs(EEP_x - EEP_x_)

        # if self.z - self.z_ > 0.17:
        #     self.done = True
        #     self.reward += 10
        #     print("Success")
        if self.z - self.z_ > 0.28:
            self.done = True
            self.reward += 1
            print("Success")
        # if self.z > 0.3:
        #     self.done = True
        #     self.reward += 10
        #     print("Success")

        # elif EEP_x >= 0.65 or EEP_x <= 0.35:
        #     self.done = True
        #     # self.reward -= 1
        #     print("x_Fail")
        #
        # elif EEP_y >= 0.06 or EEP_y <= -0.02:
        #     self.done = True
        #     # self.reward -= 1
        #     print("y_Fail")
        #
        # elif EEP_z <= 0.05:
        #     self.done = True
        #     # self.reward -= 1
        #     print("z_Fail")

        else:
            self.done = False

        self.info = {}
        self.terminated = {}
        # time.sleep(0.01)

        return np.array(self.obs).astype(np.float32), self.reward, self.info, self.done, self.terminated

    def reset(self):
        self.step_count = 0
        p.resetSimulation()
        p.setGravity(0, 0, Gravity)
        urdfRootPath = pybullet_data.getDataPath()
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 0)
        self.timeStep = 1 / 240
        self.state = 0
        self.last_state = 0
        self.state_t = 0
        self.state_durations = [0.2, 5, 2, 0.5]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), basePosition=[0, 0, 0],
                                   useFixedBase=True)
        # self.hole = p.loadURDF("Part1/Part1.urdf", basePosition=[0.6, 0, 0], useFixedBase=True)
        # self.peg = p.loadURDF("Part2/Part2.urdf", basePosition=[0.6, 0, 0.03], useFixedBase=False)
        self.shaft1Uid = p.loadURDF("11/11.urdf", basePosition=[0.7, 0, 0.13], useFixedBase=True)
        self.shaft2Uid = p.loadURDF("11/11.urdf", basePosition=[0.3, 0, 0.13], useFixedBase=True)
        self.sheet1Uid = p.loadURDF("sheet2/sheet2.urdf", basePosition=[0.3, 0, 0.08],
                                    useFixedBase=True)
        self.sheet2Uid = p.loadURDF("sheet2/sheet2.urdf", basePosition=[0.7, 0, 0.08],
                                    useFixedBase=True)
        # self.barUid = p.loadURDF("bar/bar.urdf", basePosition=[0.5, 0, 0.75])
        self.barUid = p.loadURDF("bar/bar.urdf", basePosition=[0.5, 0, 1])


        self.Position = np.array(p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[0], dtype=float)
        self.currentOrn = p.getLinkState(self.pandaUid, self.pandaEndEffectorIndex)[1]
        self.Orientation = np.array(p.getEulerFromQuaternion(self.currentOrn), dtype=float)
        p.enableJointForceTorqueSensor(self.pandaUid, 8, 1)
        self.Force = np.array(p.getJointState(self.pandaUid, 8)[2][0:3], dtype=float)
        self.Torque = np.array(p.getJointState(self.pandaUid, 8)[2][3:6], dtype=float)

        self.x_init = p.getBasePositionAndOrientation(self.barUid)[0][0]
        self.y_init = p.getBasePositionAndOrientation(self.barUid)[0][1]
        self.z_init = p.getBasePositionAndOrientation(self.barUid)[0][2]
        print("z_init:", p.getBasePositionAndOrientation(self.barUid)[0])
        self.rpy_init = p.getBasePositionAndOrientation(self.barUid)[1]
        self.rpy_init_ = np.array(p.getEulerFromQuaternion(self.rpy_init), dtype=float)
        self.r_init = self.rpy_init_[0]
        self.p_init = self.rpy_init_[1]
        self.y_init_ = self.rpy_init_[2]
        self.D1_init = distance.euclidean(p.getBasePositionAndOrientation(self.barUid)[0], self.Position)
        self.O1_init = distance.euclidean(self.rpy_init_, self.Orientation)

        # observation = [self.Position[0], self.Position[1], self.Position[2],
        #                self.Orientation[0], self.Orientation[1], self.Orientation[2],
        #                self.Force[0], self.Force[1], self.Force[2],
        #                self.Torque[0], self.Torque[1], self.Torque[2]]
        observation = [self.Position[0], self.Position[1], self.Position[2],
                    self.Orientation[0], self.Orientation[1], self.Orientation[2],
                    self.x_init, self.y_init, self.z_init, self.r_init, self.p_init, self.y_init_, self.D1_init, self.O1_init]

        self.initial_robot()
        return np.array(observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=-70,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = DisassemblyEnv()
    env.reset()
    # env.initial_robot()
    for step in range(50000):
        action = env.action_space.sample()
        print(action)
        obs, reward, info, done, terminated = env.step(action)
        print(obs)
        p.stepSimulation()
        if done is True:
            env.reset()
