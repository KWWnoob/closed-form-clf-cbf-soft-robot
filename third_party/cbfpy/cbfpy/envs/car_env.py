"""
# Adaptive Cruise Control Simulation Environment

Simple simulation of a leader-follower vehicle system for adaptive cruise control testing
"""

from typing import Tuple

import numpy as np
import pygame

from cbfpy.envs.base_env import BaseEnv

CAR_ASCII = """
        _______
       //  ||\\ \\
 _____//___||_\\ \\___
 )  _          _    \\
 |_/ \\________/ \\___|
___\\_/________\\_/______
"""


class VehicleEnv(BaseEnv):
    """Leader/follower vehicle simulation environment for adaptive cruise control testing

    This will bring up an interactive Pygame window where you can control the speed of the leader car,
    and the follower car will be controlled by a simple adaptive cruise control algorithm.

    Args:
        controller_name (str): Name of the controller being tested
        mass (float, optional): Mass of the follower vehicle. Defaults to 1650.0 (kg)
        drag_coeffs (Tuple[float, float, float]): Coefficients of a simple polynomial friction model, as
            described in the Ames CBF paper. Defaults to (0.1, 5.0, 0.25)
        v_des (float, optional): Desired velocity of the follower car. Defaults to 24.0 (m/s)
        init_leader_pos (float, optional): Initial position of the leader car. Defaults to 0.0 (meters)
        init_leader_vel (float, optional): Initial velocity of the leader car. Defaults to 14.0 (m/s)
        init_follower_pos (float, optional): Initial position of the follower car. Defaults to -20.0 (meters)
        init_follower_vel (float, optional): Initial velocity of the follower car. Defaults to 10.0 (m/s)
        u_min (float, optional): Minimum control input, in Newtons. Defaults to -np.inf (unconstrained)
        u_max (float, optional): Maximum control input, in Newtons. Defaults to np.inf (unconstrained)
    """

    # Constants
    PIXELS_PER_METER = 15
    MIN_FOLLOW_DIST_M = 1
    MIN_FOLLOW_DIST_PX = PIXELS_PER_METER * MIN_FOLLOW_DIST_M

    # Screen dimensions
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 800

    # Colors
    WHITE = (255, 255, 255)
    GRAY = (100, 100, 100)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)

    # Road properties
    DASH_HEIGHT_M = 2
    DASH_WIDTH_M = 0.3
    DASH_GAP_M = 1
    DASH_HEIGHT_PX = PIXELS_PER_METER * DASH_HEIGHT_M
    DASH_WIDTH_PX = PIXELS_PER_METER * DASH_WIDTH_M
    DASH_GAP_PX = PIXELS_PER_METER * DASH_GAP_M

    # Car properties
    CAR_WIDTH_M = 2
    CAR_HEIGHT_M = 4
    CAR_WIDTH_PX = PIXELS_PER_METER * CAR_WIDTH_M
    CAR_HEIGHT_PX = PIXELS_PER_METER * CAR_HEIGHT_M

    def __init__(
        self,
        controller_name: str,
        mass: float = 1650.0,
        drag_coeffs: Tuple[float, float, float] = (0.1, 5.0, 0.25),
        v_des: float = 24.0,
        init_leader_pos: float = 0.0,
        init_leader_vel: float = 14.0,
        init_follower_pos: float = -20.0,
        init_follower_vel: float = 10.0,
        u_min: float = -np.inf,
        u_max: float = np.inf,
    ):
        self.mass = mass
        self.v_des = v_des  # Desired velocity of the follower car
        assert len(drag_coeffs) == 3
        self.f0, self.f1, self.f2 = drag_coeffs
        # Initialize Pygame
        pygame.init()
        # Set up the display
        self.screen = pygame.display.set_mode(
            (VehicleEnv.SCREEN_WIDTH, VehicleEnv.SCREEN_HEIGHT)
        )
        pygame.display.set_caption(f"Adaptive Cruise Control: {controller_name}")
        # Create car sprites
        self.leader_sprite = pygame.Surface(
            (VehicleEnv.CAR_WIDTH_PX, VehicleEnv.CAR_HEIGHT_PX)
        )
        self.leader_sprite.fill(VehicleEnv.RED)
        self.follower_sprite = pygame.Surface(
            (VehicleEnv.CAR_WIDTH_PX, VehicleEnv.CAR_HEIGHT_PX)
        )
        self.follower_sprite.fill(VehicleEnv.BLUE)
        # Positioning the cars in the display
        self.leader_x = int(VehicleEnv.SCREEN_WIDTH / 4 - VehicleEnv.CAR_WIDTH_PX / 2)
        self.leader_y = VehicleEnv.SCREEN_HEIGHT // 5
        self.follower_x = self.leader_x
        self.follower_y = self.leader_y + VehicleEnv.MIN_FOLLOW_DIST_PX

        # Init dynamics
        self.leader_pos = init_leader_pos
        self.leader_vel = init_leader_vel
        self.follower_pos = init_follower_pos
        self.follower_vel = init_follower_vel

        self.leader_vel_des = init_leader_vel

        # Initial position of dashed lines
        self.dash_offset = 0

        self.font = pygame.font.SysFont("Arial", 20)

        self.fps = 60
        self.dt = 1 / self.fps
        self.running = True

        self.last_control = 0.0

        self.u_min = u_min
        self.u_max = u_max

        # Print instructions
        print(CAR_ASCII)
        print("Beginning Vehicle Simulation")
        print("Press UP and DOWN to control the speed of the leader car")
        print("Press ESC to quit")

    def pixels_to_meters(self, n_pixels: int) -> float:
        """Helper function: Converts pixels to meters

        Args:
            n_pixels (int): Number of pixels

        Returns:
            float: Distance in meters
        """
        return n_pixels / self.PIXELS_PER_METER

    def meters_to_pixels(self, n_meters: float) -> float:
        """Helper function: Converts meters to pixels

        Args:
            n_meters (float): Distance in meters

        Returns:
            float: Number of pixels (Note: Not rounded to an integer value)
        """
        return n_meters * self.PIXELS_PER_METER

    def friction(self, v: float) -> float:
        """Computes the drag force on the vehicle with the simple model presented in the Ames CBF paper

        Args:
            v (float): Car speed, in m/s

        Returns:
            float: Drag force on the car, in Newtons
        """
        return self.f0 + self.f1 * v + self.f2 * v**2

    @property
    def follow_distance(self):
        """Distance between the leader and follower cars, in meters"""
        return self.leader_pos - self.follower_pos - self.CAR_HEIGHT_M

    def get_state(self):
        return np.array(
            [
                self.follower_vel,
                self.leader_vel,
                self.follow_distance,
            ]
        )

    def get_desired_state(self):
        leader_speed_kmh = 3.6 * self.leader_vel
        safe_follow_distance = leader_speed_kmh / 2 + self.MIN_FOLLOW_DIST_M
        # Note: This desired state is primarily for the simple nominal controller and NOT the CLF-CBF
        return np.array(
            [
                self.v_des,
                self.leader_vel,
                safe_follow_distance,
            ]
        )

    def apply_control(self, u) -> None:
        u = np.clip(u, self.u_min, self.u_max).item()
        if np.isnan(u):
            print("Infeasible. Using last safe control")
            u = self.last_control
        force = u - self.friction(self.follower_vel)
        follower_accel = force / self.mass
        self.follower_vel += follower_accel * self.dt
        self.follower_pos += self.follower_vel * self.dt
        self.last_control = u

    def leader_controller(self) -> float:
        """Simple controller for the leader car

        The desired leader velocity will be set by the user, and this controller will try to track that velocity

        Returns:
            float: Control input to the leader car: Acceleration force, in Newtons
        """
        kv = 1000
        return kv * (self.leader_vel_des - self.leader_vel)

    def step(self):
        # Handle events
        # This includes where the speed of the main controlled by the user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.leader_vel_des += 1
                elif event.key == pygame.K_DOWN:
                    self.leader_vel_des -= 1
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                    return

        leader_u = self.leader_controller()
        leader_force = leader_u - self.friction(self.leader_vel)
        # Assume both vehicles have the same mass
        leader_accel = leader_force / self.mass
        self.leader_vel += leader_accel * self.dt
        self.leader_pos += self.leader_vel * self.dt
        follow_dist = self.follow_distance

        # Update locations of the cars in pixel frame
        self.follower_y = (
            self.leader_y + self.meters_to_pixels(follow_dist) + self.CAR_HEIGHT_PX
        )

        # Clear the screen
        self.screen.fill(VehicleEnv.WHITE)

        # Draw the road
        pygame.draw.rect(
            self.screen,
            VehicleEnv.GRAY,
            (
                0,
                0,
                VehicleEnv.SCREEN_WIDTH // 2,
                VehicleEnv.SCREEN_HEIGHT,
            ),
        )

        # Draw dashed lines on the road
        dash_y = self.dash_offset
        while dash_y < VehicleEnv.SCREEN_HEIGHT:
            pygame.draw.rect(
                self.screen,
                VehicleEnv.YELLOW,
                (
                    VehicleEnv.SCREEN_WIDTH // 4 - VehicleEnv.DASH_WIDTH_PX // 2,
                    dash_y,
                    VehicleEnv.DASH_WIDTH_PX,
                    VehicleEnv.DASH_HEIGHT_PX,
                ),
            )
            dash_y += VehicleEnv.DASH_HEIGHT_PX + VehicleEnv.DASH_GAP_PX

        # Move the dashed lines
        self.dash_offset += self.leader_vel
        if self.dash_offset >= VehicleEnv.DASH_HEIGHT_PX + VehicleEnv.DASH_GAP_PX:
            self.dash_offset = 0

        # Draw the cars
        self.screen.blit(self.leader_sprite, (self.leader_x, self.leader_y))
        self.screen.blit(self.follower_sprite, (self.follower_x, self.follower_y))

        # Print info to the pygame window
        info_1 = f"Leader desired speed: {self.leader_vel_des:.2f}"
        info_2 = f"Leader speed: {self.leader_vel:.2f}"
        info_3 = f"Follower speed: {self.follower_vel:.2f}"
        info_4 = f"Follow distance: {follow_dist:.2f}"
        info_5 = f"Control: {self.last_control:.2f}"
        text = [info_1, info_2, info_3, info_4, info_5]
        for i, line in enumerate(text):
            self.screen.blit(
                self.font.render(line, True, VehicleEnv.BLACK),
                (VehicleEnv.SCREEN_WIDTH // 2 + 10, 10 + 30 * i),
            )

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(self.fps)


def _test_env():
    """Test the environment behavior under an **unsafe** nominal controller"""

    def nominal_controller(z, z_des):
        vf, vl, D = z
        vf_des, vl_des, D_des = z_des
        kp = 1000
        kv = 1000
        return kp * (D - D_des) + kv * (vf_des - vf)

    env = VehicleEnv("Nominal Controller")
    print("\nDemoing the vehicle environment with an unsafe controller")
    while env.running:
        z = env.get_state()
        z_des = env.get_desired_state()
        u = nominal_controller(z, z_des)
        env.apply_control(u)
        env.step()


if __name__ == "__main__":
    _test_env()
