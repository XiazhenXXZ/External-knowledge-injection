import pybullet as p
import numpy as np


class PositionController:
    def pc_pos(self, pandaUid, pandaEndEffectorIndex, pandaNumDofs, delta_x, delta_y, delta_z):
        currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
        currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
        jointPoses = p.calculateInverseKinematics(
            pandaUid, pandaEndEffectorIndex,
            currentPos + np.array([delta_x, delta_y, delta_z]),
            currentOrn)
        for i in range(pandaNumDofs):
            p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])

    def pc_ori(self, pandaUid, pandaEndEffectorIndex, pandaNumDofs, delta_Alpha, delta_Beta, delta_Gamma):
        currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
        currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
        jointPoses = p.calculateInverseKinematics(
            pandaUid, pandaEndEffectorIndex,
            currentPos,
            p.getQuaternionFromEuler([p.getEulerFromQuaternion(currentOrn)[0] + delta_Alpha,
                                      p.getEulerFromQuaternion(currentOrn)[1] + delta_Beta,
                                      p.getEulerFromQuaternion(currentOrn)[2] + delta_Gamma]))
        for i in range(pandaNumDofs):
            p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])

    def pc(self, pandaUid, pandaEndEffectorIndex, pandaNumDofs, delta_x, delta_y, delta_z, delta_Alpha, delta_Beta, delta_Gamma):
        currentPos = p.getLinkState(pandaUid, pandaEndEffectorIndex)[0]
        currentOrn = p.getLinkState(pandaUid, pandaEndEffectorIndex)[1]
        jointPoses = p.calculateInverseKinematics(
            pandaUid, pandaEndEffectorIndex,
            currentPos + np.array([delta_x, delta_y, delta_z]),
            p.getQuaternionFromEuler([p.getEulerFromQuaternion(currentOrn)[0] + delta_Alpha,
                                      p.getEulerFromQuaternion(currentOrn)[1] + delta_Beta,
                                      p.getEulerFromQuaternion(currentOrn)[2] + delta_Gamma]))
        for i in range(pandaNumDofs):
            p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointPoses[i])

    def gripper(self, pandaUid, movement, force):
        for i in [9, 10]:
            p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, movement, force=force)