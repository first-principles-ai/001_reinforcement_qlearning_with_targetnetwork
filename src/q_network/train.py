"""
Training Loop — Main entry point for DQN training

Runs the full training pipeline:
1. Load configuration from YAML
2. Initialize environment and agent
3. Train for N episodes with experience replay
4. Log metrics, save checkpoints, generate plots
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

from .environment import InvertedPendulumEnv
from .agent import DQNAgent
from .visualize import plot_training_metrics


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_weights_from_json(model, json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    weight_arrays = [np.array(v, dtype=np.float32) for v in data.values()]
    model.set_weights(weight_arrays)


def _save_weights_to_json(model, json_path):
    weights = {}
    for i, layer in enumerate(model.layers):
        for j, w in enumerate(layer.get_weights()):
            key = f"layer_{i}_{'kernel' if j == 0 else 'bias'}_{list(w.shape)}"
            weights[key] = w.tolist()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)


def _make_run_folder_name(config, num_episodes, pretrained_path):
    mode = "target" if config["training"].get("use_target_network", True) else "basic"
    weights_tag = ""
    if pretrained_path:
        weights_tag = f"_from_{os.path.splitext(os.path.basename(pretrained_path))[0]}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"train_{mode}_{num_episodes}ep{weights_tag}_{timestamp}"


def _make_weights_filename(config, num_episodes, pretrained_path):
    mode = "target" if config["training"].get("use_target_network", True) else "basic"
    suffix = "_finetuned" if pretrained_path else ""
    return f"weights_{mode}_{num_episodes}ep{suffix}.json"


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_project_path(path):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(_project_root(), path)


def train(config, args):
    cfg_train = config.setdefault("training", {})
    if getattr(args, "episodes", None) is not None:
        cfg_train["num_episodes"] = args.episodes
    if getattr(args, "lr", None) is not None:
        cfg_train["learning_rate"] = args.lr
    if getattr(args, "gamma", None) is not None:
        cfg_train["gamma"] = args.gamma
    if getattr(args, "batch_size", None) is not None:
        cfg_train["batch_size"] = args.batch_size
    if getattr(args, "no_target", False):
        cfg_train["use_target_network"] = False
    if getattr(args, "weights", None) is not None:
        cfg_train["pretrained_weights"] = args.weights

    num_episodes = cfg_train.get("num_episodes", 2000)
    pretrained_cfg = cfg_train.get("pretrained_weights")
    pretrained_path = _resolve_project_path(pretrained_cfg) if pretrained_cfg else None

    results_base = os.path.join(_project_root(), "results")
    os.makedirs(results_base, exist_ok=True)
    results_dir = os.path.join(results_base, _make_run_folder_name(config, num_episodes, pretrained_path))
    os.makedirs(results_dir, exist_ok=True)

    print("[Device] Running on CPU")
    env = InvertedPendulumEnv(config)
    agent = DQNAgent(config)
    agent.model(np.zeros((1, 4), dtype=np.float32))

    if pretrained_path:
        if os.path.exists(pretrained_path):
            if pretrained_path.endswith(".json"):
                _load_weights_from_json(agent.model, pretrained_path)
            else:
                agent.load(pretrained_path)
            agent.update_target_network()
        else:
            print(f"[Warning] Pretrained weights not found: {pretrained_path}. Starting fresh.")

    print("=" * 72)
    print("Inverted Pendulum DQN Training")
    print(f"Mode: {'target' if cfg_train.get('use_target_network', True) else 'basic'}")
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: {cfg_train.get('learning_rate')}")
    print(f"Gamma: {cfg_train.get('gamma')}")
    print(f"Batch size: {cfg_train.get('batch_size')}")
    print(f"Parameters: {agent.model.count_params()}")
    print("=" * 72)

    all_rewards = []
    all_losses = []
    all_lengths = []
    best_avg_reward = -np.inf
    start_time = time.time()
    log_interval = cfg_train.get("log_interval", 50)
    save_interval = cfg_train.get("save_interval", 100)

    for episode_idx in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        losses = []
        steps = 0

        for _ in range(env.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            episode_reward += reward
            steps += 1
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        if episode_idx % agent.target_update_freq == 0:
            agent.update_target_network()

        all_rewards.append(float(episode_reward))
        all_losses.append(float(np.mean(losses)) if losses else 0.0)
        all_lengths.append(int(steps))

        avg100 = float(np.mean(all_rewards[-100:]))
        if avg100 > best_avg_reward:
            best_avg_reward = avg100
            agent.save(os.path.join(results_dir, "trained_model.keras"))

        if episode_idx % save_interval == 0:
            agent.save(os.path.join(results_dir, f"checkpoint_ep{episode_idx}.keras"))

        if episode_idx % log_interval == 0 or episode_idx == 1:
            print(
                f"Episode {episode_idx:5d}/{num_episodes} | "
                f"reward={episode_reward:9.3f} | avg100={avg100:9.3f} | "
                f"len={steps:4d} | eps={agent.epsilon:.4f} | "
                f"loss={all_losses[-1]:.6f}"
            )

    agent.save(os.path.join(results_dir, "trained_model_final.keras"))
    weights_filename = _make_weights_filename(config, num_episodes, pretrained_path)
    run_weights_path = os.path.join(results_dir, weights_filename)
    root_weights_path = os.path.join(results_base, weights_filename)
    _save_weights_to_json(agent.model, run_weights_path)
    _save_weights_to_json(agent.model, root_weights_path)

    total_time = time.time() - start_time
    metadata = {
        "run_folder": os.path.basename(results_dir),
        "mode": "target" if cfg_train.get("use_target_network", True) else "basic",
        "pretrained_weights": pretrained_path,
        "num_episodes": num_episodes,
        "total_time_s": total_time,
        "best_avg_reward": float(best_avg_reward),
        "final_epsilon": float(agent.epsilon),
        "device": "CPU",
        "parameters": int(agent.model.count_params()),
    }
    with open(os.path.join(results_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "config": config,
            "rewards": all_rewards,
            "losses": all_losses,
            "episode_lengths": all_lengths,
        }, f, indent=2)

    plot_training_metrics(all_rewards, all_losses, all_lengths, results_dir)
    print(f"Training complete in {total_time:.2f}s")
    print(f"Results written to: {results_dir}")
    return agent, all_rewards, all_losses, all_lengths


def main():
    default_config = os.path.join(_project_root(), "configs", "default.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no-target", action="store_true")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
