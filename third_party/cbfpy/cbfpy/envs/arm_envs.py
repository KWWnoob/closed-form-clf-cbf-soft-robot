"""
# Simulation environents for robot arms

This currently includes a very simple 3-DOF environment which helps demonstrate joint limit avoidance,
but more will be added in the future
"""

import numpy as np
import pybullet
from pybullet_utils.bullet_client import BulletClient

from cbfpy.utils.general_utils import find_assets_dir, stdout_redirected
from cbfpy.envs.base_env import BaseEnv

URDF = find_assets_dir() + "three_dof_arm.urdf"


class JointLimitsEnv(BaseEnv):
    """Simulation environment for the 3-DOF arm joint-limit-avoidance demo

    This includes a desired reference trajectory which is unsafe: it will command sinusoidal
    joint motions (with different frequencies per link) that will exceed the joint limits of the robot
    """

    def __init__(self):
        with stdout_redirected(restore=False):
            self.client: pybullet = BulletClient(pybullet.GUI)
        self.robot = pybullet.loadURDF(URDF, useFixedBase=True)
        self.num_joints = self.client.getNumJoints(self.robot)
        self.q_min = np.array(
            [self.client.getJointInfo(self.robot, i)[8] for i in range(self.num_joints)]
        )
        self.q_max = np.array(
            [self.client.getJointInfo(self.robot, i)[9] for i in range(self.num_joints)]
        )
        self.timestep = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.t = 0

        # Sinusoids for the desired joint positions
        # Setting the amplitude to be the full joint range means we will command DOUBLE
        # the joint range, exceeding our limits
        self.omegas = 0.1 * np.array([1.0, 2.0, 3.0])
        self.amps = self.q_max - self.q_min
        self.offsets = np.zeros(3)

    def step(self):
        self.client.stepSimulation()
        self.t += self.timestep

    def get_state(self):
        states = self.client.getJointStates(self.robot, range(self.num_joints))
        return np.array([states[i][0] for i in range(self.num_joints)])

    def get_desired_state(self):
        # Evaluate our unsafe sinusoidal trajectory
        return self.amps * np.sin(self.omegas * self.t) + self.offsets

    def apply_control(self, u):
        self.client.setJointMotorControlArray(
            self.robot,
            list(range(self.num_joints)),
            self.client.VELOCITY_CONTROL,
            targetVelocities=u,
        )


def _test_env():
    """Test the environment behavior under an **unsafe** nominal controller"""

    def nominal_controller(z, z_des):
        k = 1.0
        return k * (z_des - z)

    env = JointLimitsEnv()
    while True:
        z = env.get_state()
        z_des = env.get_desired_state()
        u = nominal_controller(z, z_des)
        env.apply_control(u)
        env.step()


if __name__ == "__main__":
    _test_env()
