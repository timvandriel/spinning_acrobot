from gymnasium.envs.classic_control.acrobot import AcrobotEnv
import numpy as np


class SpinningAcrobotEnv(AcrobotEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        # Spin tracking variables
        self.prev_tip_angle = None  # Previous tip angle, to compute delta
        self.total_spin_angle = 0.0  # Accumulated angle in one direction
        self.spin_direction = None  # 'cw' or 'ccw', decided after first movement
        self.target_spins = 20  # Number of spins required
        self.spin_complete = False  # Flag for whether goal is met

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # Reset tip rotation tracking
        self.prev_tip_angle = None
        self.total_spin_angle = 0.0
        self.spin_direction = None  # 'cw' or 'ccw'
        self.spin_complete = False

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Get current joint angles
        theta1, theta2 = self.state[0], self.state[1]

        # Compute the tip position (radians)
        x_tip = -np.cos(theta1) - np.cos(theta1 + theta2)
        y_tip = -np.sin(theta1) - np.sin(theta1 + theta2)

        # Compute current tip angle
        tip_angle = np.arctan2(y_tip, x_tip)

        if self.prev_tip_angle is not None:
            # Handle wrap-around using np.unwrap-like logic
            delta = tip_angle - self.prev_tip_angle
            delta = (delta + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

            # Determine direction if not already known
            if self.spin_direction is None and abs(delta) > 0.01:
                self.spin_direction = "cw" if delta < 0 else "ccw"

            # Only accumulate angle if it's in the same direction
            if self.spin_direction == "cw" and delta < 0:
                self.total_spin_angle += abs(delta)
            elif self.spin_direction == "ccw" and delta > 0:
                self.total_spin_angle += abs(delta)
            else:
                # Wrong direction: reset
                self.total_spin_angle = 0.0
                self.spin_direction = None

        self.prev_tip_angle = tip_angle

        # Check for success
        if self.total_spin_angle >= self.target_spins * 2 * np.pi:
            reward = 1000.0
            terminated = True
            self.spin_complete = True
        else:
            reward = 0.0
            terminated = False

        return obs, reward, terminated, truncated, info
