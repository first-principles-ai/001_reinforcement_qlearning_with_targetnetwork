"""
Inverted Pendulum Environment — Custom Physics Simulation

Simulates a cart-pole system using Lagrangian-derived equations of motion
with semi-implicit Euler integration. No dependency on OpenAI Gym.
"""

import numpy as np


class InvertedPendulumEnv:
    FORCE_TABLE = [-10.0, -5.0, 0.0, 5.0, 10.0]

    def __init__(self, config=None):
        cfg_phys = config.get("physics", {}) if config else {}
        cfg_env = config.get("environment", {}) if config else {}
        cfg_reward = config.get("reward", {}) if config else {}

        self.mc = cfg_phys.get("cart_mass", 1.0)
        self.mp = cfg_phys.get("pole_mass", 0.1)
        self.l = cfg_phys.get("pole_half_length", 0.5)
        self.g = cfg_phys.get("gravity", 9.81)
        self.force_max = cfg_phys.get("force_max", 10.0)
        self.dt = cfg_phys.get("timestep", 0.02)

        self.max_steps = cfg_env.get("max_steps", 500)
        self.angle_threshold = np.radians(cfg_env.get("angle_threshold_deg", 12.0))
        self.position_threshold = cfg_env.get("position_threshold", 2.4)
        self.init_perturbation = cfg_env.get("initial_perturbation", 0.05)

        self.alive_bonus = cfg_reward.get("alive_bonus", 1.0)
        self.angle_weight = cfg_reward.get("angle_weight", -10.0)
        self.position_weight = cfg_reward.get("position_weight", -1.0)
        self.velocity_weight = cfg_reward.get("velocity_weight", -0.1)
        self.death_penalty = cfg_reward.get("death_penalty", -100.0)

        self.state_size = 4
        self.action_size = len(self.FORCE_TABLE)
        self.total_mass = self.mc + self.mp
        self.state = None
        self.step_count = 0

    def reset(self):
        self.state = np.random.uniform(
            low=-self.init_perturbation,
            high=self.init_perturbation,
            size=(4,),
        ).astype(np.float32)
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        assert 0 <= action < self.action_size
        if self.state is None:
            self.reset()

        force = self.FORCE_TABLE[action]
        x_ddot, theta_ddot = self._compute_accelerations(self.state, force)
        x, x_dot, theta, theta_dot = self.state

        theta_dot_new = theta_dot + theta_ddot * self.dt
        theta_new = theta + theta_dot_new * self.dt
        x_dot_new = x_dot + x_ddot * self.dt
        x_new = x + x_dot_new * self.dt

        self.state = np.array([x_new, x_dot_new, theta_new, theta_dot_new], dtype=np.float32)
        self.step_count += 1

        timeout = self.step_count >= self.max_steps
        failed = abs(theta_new) > self.angle_threshold or abs(x_new) > self.position_threshold
        done = bool(failed or timeout)
        reward = self._compute_reward(x_new, x_dot_new, theta_new, theta_dot_new, failed)
        info = {"step_count": self.step_count, "force_applied": force, "timeout": timeout}
        return self.state.copy(), float(reward), done, info

    def _compute_accelerations(self, state, force):
        _, _, theta, theta_dot = state
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        numerator = (
            self.g * sin_theta
            + cos_theta * (-force - self.mp * self.l * theta_dot ** 2 * sin_theta) / self.total_mass
        )
        denominator = self.l * (4.0 / 3.0 - self.mp * cos_theta ** 2 / self.total_mass)
        theta_ddot = numerator / denominator

        x_ddot = (
            force + self.mp * self.l * (theta_dot ** 2 * sin_theta - theta_ddot * cos_theta)
        ) / self.total_mass
        return x_ddot, theta_ddot

    def _compute_reward(self, x, x_dot, theta, theta_dot, failed):
        if failed:
            return self.death_penalty
        angle_penalty = self.angle_weight * theta ** 2
        position_penalty = self.position_weight * x ** 2
        velocity_penalty = self.velocity_weight * (x_dot ** 2 + theta_dot ** 2)
        return self.alive_bonus + angle_penalty + position_penalty + velocity_penalty
