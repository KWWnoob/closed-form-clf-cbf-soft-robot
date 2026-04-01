"""
# Simulation environments for point robots
"""

import numpy as np
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import pybullet
from pybullet_utils.bullet_client import BulletClient

from cbfpy.utils.visualization import visualize_3D_box
from cbfpy.envs.base_env import BaseEnv
from cbfpy.utils.general_utils import find_assets_dir, stdout_redirected


class PointRobotEnv(BaseEnv):
    """Simulation environment for a point robot trying to approach a target position in 3D,
    while remaining in a safe set defined as a box

    Args:
        xyz_min (ArrayLike, optional): Minimum bounds of the safe region, shape (3,). Defaults to (-1.0, -1.0, -1.0)
        xyz_max (ArrayLike, optional): Maximum bounds of the safe region, shape (3,). Defaults to (1.0, 1.0, 1.0)
    """

    # Constants
    # Based on the values in the point robot URDF
    URDF = "point_robot.urdf"
    RADIUS = 0.25
    MASS = 1.0

    def __init__(
        self,
        xyz_min: ArrayLike = (-1.0, -1.0, -1.0),
        xyz_max: ArrayLike = (1.0, 1.0, 1.0),
    ):
        self.xyz_min = np.array(xyz_min)
        self.xyz_max = np.array(xyz_max)
        assert len(self.xyz_min) == len(self.xyz_max) == 3
        with stdout_redirected(restore=False):
            self.client: pybullet = BulletClient(pybullet.GUI)
        self.client.setAdditionalSearchPath(find_assets_dir())
        self.client.configureDebugVisualizer(self.client.COV_ENABLE_GUI, 0)
        self.robot = self.client.loadURDF(self.URDF)
        self.target = self.client.loadURDF(self.URDF, basePosition=(3, 1, 1))
        self.client.changeVisualShape(self.target, -1, rgbaColor=[1, 0, 0, 1])
        self.client.changeDynamics(self.target, -1, linearDamping=10, angularDamping=10)
        self.box = visualize_3D_box(
            [self.xyz_min - self.RADIUS, self.xyz_max + self.RADIUS],
            rgba=(0, 1, 0, 0.5),
        )

        # For color determination
        self.is_in_box = True
        self.tol = 1e-3

    def _update_color(self, robot_pos: ArrayLike) -> None:
        """Update the color of the box depending on if the robot is inside or not (Green inside, red outside)"""
        robot_pos = np.array(robot_pos)
        if np.any(robot_pos < self.xyz_min - self.tol) or np.any(
            robot_pos > self.xyz_max + self.tol
        ):
            if self.is_in_box:
                self.client.changeVisualShape(self.box, -1, rgbaColor=[1, 0, 0, 0.5])
            self.is_in_box = False
        else:
            if not self.is_in_box:
                self.client.changeVisualShape(self.box, -1, rgbaColor=[0, 1, 0, 0.5])
            self.is_in_box = True

    def get_state(self) -> Array:
        robot_pos = self.client.getBasePositionAndOrientation(self.robot)[0]
        robot_vel = self.client.getBaseVelocity(self.robot)[0]
        self._update_color(robot_pos)
        return np.array([*robot_pos, *robot_vel])

    def get_desired_state(self) -> Array:
        target_pos = self.client.getBasePositionAndOrientation(self.target)[0]
        target_vel = self.client.getBaseVelocity(self.target)[0]
        return np.array([*target_pos, *target_vel])

    def apply_control(self, u: Array) -> None:
        robot_pos = self.client.getBasePositionAndOrientation(self.robot)[0]
        self.client.applyExternalForce(
            self.robot, -1, u, robot_pos, self.client.WORLD_FRAME
        )

    def step(self):
        self.client.stepSimulation()


class PointRobotObstacleEnv(BaseEnv):
    """Simulation environment for a point robot avoiding a movable obstacle,
    while remaining in a safe set defined as a box

    Args:
        robot_pos (ArrayLike, optional): Initial position of the robot. Defaults to (0, 0, 0).
        robot_vel (ArrayLike, optional): Initial velocity of the robot. Defaults to (0, 0, 0).
        obstacle_pos (ArrayLike, optional): Initial position of the obstacle. Defaults to (1, 1, 1).
        obstacle_vel (ArrayLike, optional): Initial velocity of the obstacle. Defaults to (-0.5, -0.6, -0.7).
        xyz_min (ArrayLike, optional): Minimum bounds of the safe region, shape (3,). Defaults to (-1.0, -1.0, -1.0)
        xyz_max (ArrayLike, optional): Maximum bounds of the safe region, shape (3,). Defaults to (1.0, 1.0, 1.0)
    """

    # Constants
    # Based on the values in the point robot URDF
    URDF = "point_robot.urdf"
    RADIUS = 0.25
    MASS = 1.0

    def __init__(
        self,
        robot_pos: ArrayLike = (0, 0, 0),
        robot_vel: ArrayLike = (0, 0, 0),
        obstacle_pos: ArrayLike = (1, 1, 1),
        obstacle_vel: ArrayLike = (-0.5, -0.6, -0.7),
        xyz_min: ArrayLike = (-1.0, -1.0, -1.0),
        xyz_max: ArrayLike = (1.0, 1.0, 1.0),
    ):
        with stdout_redirected(restore=False):
            self.client: pybullet = BulletClient(pybullet.GUI)
        self.client.setAdditionalSearchPath(find_assets_dir())
        self.client.configureDebugVisualizer(self.client.COV_ENABLE_GUI, 0)
        self.robot = self.client.loadURDF(self.URDF, basePosition=robot_pos)
        self.target = self.client.loadURDF(self.URDF, basePosition=obstacle_pos)
        self.client.changeVisualShape(self.target, -1, rgbaColor=[1, 0, 0, 1])
        self.client.changeDynamics(self.target, -1, angularDamping=10)
        self.client.resetBaseVelocity(self.target, obstacle_vel, (0, 0, 0))
        self.client.resetBaseVelocity(self.robot, robot_vel, (0, 0, 0))
        self.client.addUserDebugPoints([[0, 0, 0]], [[1, 0, 0]], 10, 0)
        self.xyz_min = np.array(xyz_min)
        self.xyz_max = np.array(xyz_max)
        self.box = visualize_3D_box(
            [self.xyz_min - self.RADIUS, self.xyz_max + self.RADIUS],
            rgba=(0, 1, 0, 0.5),
        )

    def get_state(self):
        robot_pos = self.client.getBasePositionAndOrientation(self.robot)[0]
        robot_vel = self.client.getBaseVelocity(self.robot)[0]
        obstacle_pos = self.client.getBasePositionAndOrientation(self.target)[0]
        obstacle_vel = self.client.getBaseVelocity(self.target)[0]
        return np.array([*robot_pos, *robot_vel]), np.array(
            [*obstacle_pos, *obstacle_vel]
        )

    def get_desired_state(self):
        return np.zeros(6)

    def apply_control(self, u):
        robot_pos = self.client.getBasePositionAndOrientation(self.robot)[0]
        self.client.applyExternalForce(
            self.robot, -1, u, robot_pos, self.client.WORLD_FRAME
        )

    def step(self):
        self.client.stepSimulation()


## Tests ##


def _test_standard_env():
    def nominal_controller(z, z_des):
        # Use a PD controller to try to "touch" the target robot
        Kp = 1.0
        Kd = 1.0
        pos_diff = z_des[:3] - z[:3]
        des_pos = z_des[:3] - 2 * PointRobotEnv.RADIUS * pos_diff / jnp.linalg.norm(
            pos_diff
        )
        return -Kp * (z[:3] - des_pos) - Kd * (z[3:] - z_des[3:])

    env = PointRobotEnv()
    while True:
        z = env.get_state()
        z_des = env.get_desired_state()
        u = nominal_controller(z, z_des)
        env.apply_control(u)
        env.step()


def _test_obstacle_env():
    def nominal_controller(z, z_des):
        Kp = 1.0
        Kd = 1.0
        return -Kp * (z[:3] - z_des[:3]) - Kd * (z[3:] - z_des[3:])

    env = PointRobotObstacleEnv()
    while True:
        z, z_obs = env.get_state()
        z_des = env.get_desired_state()
        u = nominal_controller(z, z_des)
        env.apply_control(u)
        env.step()


def _test_env(env_class: BaseEnv):
    print("\nDemoing the point robot environment with an unsafe controller")
    if env_class == PointRobotEnv:
        return _test_standard_env()
    elif env_class == PointRobotObstacleEnv:
        return _test_obstacle_env()
    else:
        raise ValueError("Invalid env class")


if __name__ == "__main__":
    _test_env(PointRobotEnv)
