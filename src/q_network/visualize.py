"""
Visualization — Real-time pendulum animation and training metrics plots

Provides:
- animate_episode(): Matplotlib FuncAnimation of the trained agent
- plot_training_metrics(): Reward, loss, and episode length plots
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import yaml

from .environment import InvertedPendulumEnv
from .agent import DQNAgent


def _load_weights_from_json(model, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    weight_arrays = [np.array(v, dtype=np.float32) for v in data.values()]
    model.set_weights(weight_arrays)


def moving_average(data, window):
    data = np.asarray(data, dtype=np.float32)
    if len(data) < window:
        return data
    cumsum = np.cumsum(data)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    result = np.empty(len(data))
    for i in range(min(window, len(data))):
        result[i] = cumsum[i] / (i + 1)
    result[window - 1:] = cumsum[window - 1:] / window
    return result


def plot_training_metrics(rewards, losses, lengths, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    episodes = np.arange(1, len(rewards) + 1)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    losses_arr = np.asarray(losses, dtype=np.float32)
    lengths_arr = np.asarray(lengths, dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(episodes, rewards_arr, color="royalblue", alpha=0.3, label="Reward")
    axes[0].plot(episodes, moving_average(rewards_arr, 100), color="crimson", label="MA-100")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if len(losses_arr):
        axes[1].plot(episodes[: len(losses_arr)], np.maximum(losses_arr, 1e-12), color="darkorange")
    axes[1].set_yscale("log")
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)

    axes[2].plot(episodes, lengths_arr, color="seagreen", alpha=0.3, label="Length")
    axes[2].plot(episodes, moving_average(lengths_arr, 100), color="darkgreen", label="MA-100")
    if len(lengths_arr):
        axes[2].axhline(np.max(lengths_arr), color="red", linestyle="--", alpha=0.4, label="Max seen")
    axes[2].set_title("Episode Length")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Steps")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_metrics.png"), dpi=150)
    plt.close(fig)

    _save_single_plot(episodes, rewards_arr, "Episode Reward", "Reward", os.path.join(save_dir, "reward_plot.png"), ma=True)
    _save_single_plot(episodes[: len(losses_arr)], np.maximum(losses_arr, 1e-12), "Training Loss", "Loss", os.path.join(save_dir, "loss_plot.png"), log=True)
    _save_single_plot(episodes, lengths_arr, "Episode Length", "Steps", os.path.join(save_dir, "episode_length_plot.png"), ma=True)


def _save_single_plot(x, y, title, ylabel, path, ma=False, log=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(y):
        ax.plot(x, y, alpha=0.35 if ma else 1.0)
        if ma:
            ax.plot(x, moving_average(y, 100))
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def animate_episode(env, agent, max_steps=500, save_path=None, fps=50, record_path=None, record_interval=1):
    if save_path:
        matplotlib.use("Agg")

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    state = env.reset()
    total_reward = [0.0]
    episode = [1]
    step_in_episode = [0]
    recorded = []

    fig, ax = plt.subplots(figsize=(10, 6))
    pole_len = 2 * env.l
    rail_limit = env.position_threshold + 0.5
    ax.set_xlim(-rail_limit, rail_limit)
    ax.set_ylim(-0.5, pole_len + 0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=1.5)

    cart_w = 0.4
    cart_h = 0.2
    cart = patches.FancyBboxPatch((-cart_w / 2, -cart_h / 2), cart_w, cart_h,
                                  boxstyle="round,pad=0.02", facecolor="steelblue")
    ax.add_patch(cart)
    pole_line, = ax.plot([], [], color="firebrick", linewidth=4, marker="o", markersize=8)
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                        family="monospace")
    force_arrow = [None]

    def init():
        return cart, pole_line, info_text

    def update(frame):
        nonlocal state
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        x, x_dot, theta, theta_dot = next_state
        total_reward[0] += reward
        step_in_episode[0] += 1

        cart.set_x(x - cart_w / 2)
        cart.set_y(-cart_h / 2)
        pivot_y = cart_h / 2
        tip_x = x + pole_len * np.sin(theta)
        tip_y = pivot_y + pole_len * np.cos(theta)
        pole_line.set_data([x, tip_x], [pivot_y, tip_y])

        if force_arrow[0] is not None:
            force_arrow[0].remove()
        force = info["force_applied"]
        if abs(force) > 1e-12:
            force_arrow[0] = ax.annotate("", xy=(x + 0.05 * force, -0.3), xytext=(x, -0.3),
                                         arrowprops=dict(arrowstyle="->", color="green", lw=2))
        else:
            force_arrow[0] = None

        info_text.set_text(
            f"Episode: {episode[0]}   Step: {step_in_episode[0]:3d}/{env.max_steps}\n"
            f"x: {x:+6.3f} m   ẋ: {x_dot:+6.3f} m/s\n"
            f"θ: {np.degrees(theta):+6.2f}°  θ̇: {theta_dot:+6.3f} rad/s\n"
            f"Force: {force:+6.1f} N   Reward: {reward:+6.2f}\n"
            f"Total: {total_reward[0]:+8.1f}"
        )

        if record_path and frame % max(1, record_interval) == 0:
            recorded.append({
                "frame": int(frame),
                "time_s": float(frame * env.dt),
                "episode": int(episode[0]),
                "step": int(step_in_episode[0]),
                "x": float(x),
                "x_dot": float(x_dot),
                "theta_rad": float(theta),
                "theta_deg": float(np.degrees(theta)),
                "theta_dot": float(theta_dot),
                "force_N": float(force),
                "reward": float(reward),
                "total_reward": float(total_reward[0]),
            })

        state = next_state
        if done:
            state = env.reset()
            episode[0] += 1
            step_in_episode[0] = 0
            total_reward[0] = 0.0
        artists = [cart, pole_line, info_text]
        if force_arrow[0] is not None:
            artists.append(force_arrow[0])
        return artists

    anim = FuncAnimation(fig, update, init_func=init, frames=max_steps, interval=1000 / fps, blit=False)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        anim.save(save_path, writer="pillow", fps=fps)
    else:
        plt.show()
    plt.close(fig)

    if record_path:
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        payload = {
            "metadata": {
                "model": getattr(agent, "_loaded_model_path", None),
                "dt": env.dt,
                "fps": fps,
                "record_interval": record_interval,
                "total_frames": max_steps,
                "episodes": int(episode[0]),
            },
            "data": recorded,
        }
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    agent.epsilon = original_epsilon
    return anim


def _make_sim_folder_name(config, model_path):
    mode = "target" if config.get("training", {}).get("use_target_network", True) else "basic"
    weights_stem = os.path.splitext(os.path.basename(model_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"sim_{mode}_{weights_stem}_{timestamp}"


def _default_config_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "sim_config.yaml"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config", type=str, default=_default_config_path())
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--record-interval", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sim_cfg = config.get("simulation", {})
    model_path = args.model or sim_cfg.get("weights_file")
    episodes = args.episodes or sim_cfg.get("episodes", 3)
    record_interval = args.record_interval or sim_cfg.get("record_interval", 1)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_base = os.path.join(project_root, "results")
    sim_dir = os.path.join(results_base, _make_sim_folder_name(config, model_path or "model"))
    os.makedirs(sim_dir, exist_ok=True)

    record_path = args.record if args.record is not None else sim_cfg.get("record_path")
    if record_path is None:
        record_path = os.path.join(sim_dir, "simulation_data.json")
    save_gif = args.save_gif if args.save_gif is not None else sim_cfg.get("save_gif")
    if save_gif:
        save_gif = save_gif if os.path.isabs(save_gif) else os.path.join(sim_dir, save_gif)

    print("[Device] Running on CPU")
    env = InvertedPendulumEnv(config)
    agent = DQNAgent(config)
    agent.model(np.zeros((1, 4), dtype=np.float32))
    if model_path:
        model_path_abs = model_path if os.path.isabs(model_path) else os.path.join(project_root, model_path)
        if model_path_abs.endswith(".json"):
            _load_weights_from_json(agent.model, model_path_abs)
            agent.update_target_network()
        else:
            agent.load(model_path_abs)
        agent._loaded_model_path = model_path_abs
    agent.epsilon = 0.0

    with open(os.path.join(sim_dir, "sim_config_used.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    animate_episode(env, agent, max_steps=episodes * env.max_steps, save_path=save_gif,
                    fps=config.get("visualization", {}).get("fps", 50),
                    record_path=record_path, record_interval=record_interval)


if __name__ == "__main__":
    main()
