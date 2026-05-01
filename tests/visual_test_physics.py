"""
Visual test of the physics simulation.

Runs the inverted pendulum environment with random actions
and displays a real-time Matplotlib animation of the cart-pole system.
"""

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.q_network import InvertedPendulumEnv


def run_visual_test():
    env = InvertedPendulumEnv()
    state = env.reset()
    fig, (ax_main, ax_state) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

    cart_w = 0.4
    cart_h = 0.2
    pole_len = 2 * env.l
    rail_limit = env.position_threshold + 0.5

    ax_main.set_xlim(-rail_limit, rail_limit)
    ax_main.set_ylim(-0.5, pole_len + 0.8)
    ax_main.axhline(0, color="black", linewidth=1.5)
    ax_main.axvline(-env.position_threshold, color="red", linestyle="--", alpha=0.5)
    ax_main.axvline(env.position_threshold, color="red", linestyle="--", alpha=0.5)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.grid(alpha=0.3)

    cart = patches.FancyBboxPatch((-cart_w / 2, -cart_h / 2), cart_w, cart_h,
                                  boxstyle="round,pad=0.02", facecolor="steelblue")
    ax_main.add_patch(cart)
    pole_line, = ax_main.plot([], [], color="firebrick", linewidth=4, marker="o", markersize=8)
    info_text = ax_main.text(0.02, 0.95, "", transform=ax_main.transAxes, va="top",
                             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                             family="monospace")
    force_arrow = [None]

    ax_state.set_xlim(0, 300)
    ax_state.set_ylim(-2.5, 2.5)
    ax_state.grid(alpha=0.3)
    line_x, = ax_state.plot([], [], color="royalblue", label="x")
    line_theta, = ax_state.plot([], [], color="firebrick", label="θ×10")
    line_reward, = ax_state.plot([], [], color="seagreen", label="reward")
    ax_state.legend(loc="upper right")

    episode_num = [1]
    step_in_episode = [0]
    total_reward = [0.0]
    t_hist, x_hist, theta_hist, reward_hist = [], [], [], []

    def update(frame):
        nonlocal state
        action = np.random.randint(0, env.action_size)
        state, reward, done, info = env.step(action)
        x, x_dot, theta, theta_dot = state
        total_reward[0] += reward
        step_in_episode[0] += 1

        cart.set_x(x - cart_w / 2)
        pole_line.set_data([x, x + pole_len * np.sin(theta)], [cart_h / 2, cart_h / 2 + pole_len * np.cos(theta)])

        if force_arrow[0] is not None:
            force_arrow[0].remove()
        force = info["force_applied"]
        force_arrow[0] = ax_main.annotate("", xy=(x + 0.05 * force, -0.3), xytext=(x, -0.3),
                                          arrowprops=dict(arrowstyle="->", color="green", lw=2))
        info_text.set_text(
            f"Episode: {episode_num[0]}   Step: {step_in_episode[0]:3d}/500\n"
            f"x: {x:+6.3f} m   ẋ: {x_dot:+6.3f} m/s\n"
            f"θ: {np.degrees(theta):+6.2f}°  θ̇: {theta_dot:+6.3f} rad/s\n"
            f"Force: {force:+6.1f} N   Reward: {reward:+6.2f}\n"
            f"Total: {total_reward[0]:+8.1f}"
        )

        t_hist.append(frame)
        x_hist.append(float(x))
        theta_hist.append(float(theta * 10.0))
        reward_hist.append(float(reward))
        if len(t_hist) > 300:
            del t_hist[:-300], x_hist[:-300], theta_hist[:-300], reward_hist[:-300]
        line_x.set_data(range(len(x_hist)), x_hist)
        line_theta.set_data(range(len(theta_hist)), theta_hist)
        line_reward.set_data(range(len(reward_hist)), reward_hist)

        if done:
            state = env.reset()
            episode_num[0] += 1
            step_in_episode[0] = 0
            total_reward[0] = 0.0
        return cart, pole_line, info_text, line_x, line_theta, line_reward

    FuncAnimation(fig, update, frames=2000, interval=20, blit=False)
    plt.show()


if __name__ == "__main__":
    run_visual_test()
