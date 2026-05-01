"""
DQN Agent with Target Network — Section 3.4 / Listing 3.7

Implements the improved Deep Q-Learning from "Einstieg in Deep Reinforcement Learning":

Section 3.3 (Listing 3.5): Basic DQN with experience replay
Section 3.4 (Listing 3.7): + Target network for stable training

Algorithm (Section 3.4):
  - Online network Q(s,a; θ) → used for action selection and prediction X
  - Target network Q(s,a; θ⁻) → used for Bellman target Y (frozen copy)
  - Hard update θ⁻ ← θ every N episodes
  - Loss = MSE(X, Y) = (1/B) · Σ (X_i - Y_i)²
  - X_i = Q(s_i, a_i; θ)
  - Y_i = r_i + γ·(1 - d_i)·max_a' Q(s'_i, a'; θ⁻)
"""

import random
from collections import deque

import numpy as np
import tensorflow as tf

from .model import build_q_network


class ReplayBuffer:
    def __init__(self, max_size=5000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, config=None):
        cfg_net = config.get("network", {}) if config else {}
        cfg_train = config.get("training", {}) if config else {}

        self.state_size = 4
        self.action_size = 5
        net_kwargs = dict(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=cfg_net.get("hidden_layers", [128, 64]),
            activation=cfg_net.get("activation", "relu"),
            initializer=cfg_net.get("initializer", "he_normal"),
        )

        self.model = build_q_network(**net_kwargs)
        self.use_target_network = cfg_train.get("use_target_network", True)
        if self.use_target_network:
            self.target_model = build_q_network(**net_kwargs)
            self.update_target_network()
        else:
            self.target_model = self.model

        self.target_update_freq = cfg_train.get("target_update_freq", 10)
        self.learning_rate = cfg_train.get("learning_rate", 1e-3)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.gamma = cfg_train.get("gamma", 0.99)
        self.batch_size = cfg_train.get("batch_size", 200)
        self.epsilon = cfg_train.get("epsilon_start", 1.0)
        self.epsilon_min = cfg_train.get("epsilon_end", 0.01)
        self.epsilon_decay = cfg_train.get("epsilon_decay", 0.995)
        buffer_size = cfg_train.get("replay_buffer_size", 5000)
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)

    def update_target_network(self):
        if not self.use_target_network:
            return
        self.target_model.set_weights(self.model.get_weights())

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_size))
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        q_values = self.model(state_tensor, training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        state1_batch = tf.convert_to_tensor(states)
        action_batch = tf.convert_to_tensor(actions)
        reward_batch = tf.convert_to_tensor(rewards)
        state2_batch = tf.convert_to_tensor(next_states)
        done_batch = tf.convert_to_tensor(dones)

        with tf.GradientTape() as tape:
            Q1 = self.model(state1_batch, training=True)
            indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), action_batch], axis=1)
            X = tf.gather_nd(Q1, indices)
            Q2 = self.target_model(state2_batch, training=False)
            V_hat = tf.reduce_max(Q2, axis=1)
            Y = reward_batch + self.gamma * (1.0 - done_batch) * V_hat
            loss = tf.reduce_mean(tf.square(X - tf.stop_gradient(Y)))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return float(loss.numpy())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        import logging
        logger = logging.getLogger("tensorflow")
        level = logger.level
        logger.setLevel(logging.ERROR)
        self.model.save(path)
        logger.setLevel(level)

    def load(self, path):
        import logging
        logger = logging.getLogger("tensorflow")
        level = logger.level
        logger.setLevel(logging.ERROR)
        self.model = tf.keras.models.load_model(path)
        logger.setLevel(level)
        if self.use_target_network:
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
        else:
            self.target_model = self.model
