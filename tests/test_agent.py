import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from src.q_network import DQNAgent, ReplayBuffer


class TestReplayBuffer:
    def test_add_and_length(self):
        buf = ReplayBuffer(max_size=10)
        assert len(buf) == 0
        buf.add(np.zeros(4), 1, 1.0, np.ones(4), False)
        assert len(buf) == 1

    def test_sample_shapes(self):
        buf = ReplayBuffer(max_size=100)
        for i in range(50):
            buf.add(np.ones(4) * i, i % 5, float(i), np.ones(4) * (i + 1), False)
        states, actions, rewards, next_states, dones = buf.sample(10)
        assert states.shape == (10, 4)
        assert actions.shape == (10,)
        assert rewards.shape == (10,)
        assert next_states.shape == (10, 4)
        assert dones.shape == (10,)
        assert states.dtype == np.float32
        assert actions.dtype == np.int32
        assert rewards.dtype == np.float32
        assert next_states.dtype == np.float32
        assert dones.dtype == np.float32

    def test_max_size_not_exceeded(self):
        buf = ReplayBuffer(max_size=20)
        for i in range(50):
            buf.add(np.zeros(4), 0, 0.0, np.zeros(4), False)
        assert len(buf) == 20

    def test_sample_raises_if_not_enough(self):
        buf = ReplayBuffer(max_size=10)
        buf.add(np.zeros(4), 0, 0.0, np.zeros(4), False)
        with pytest.raises(ValueError):
            buf.sample(10)


class TestDQNAgent:
    @pytest.fixture
    def agent(self):
        config = {
            "network": {"hidden_layers": [32, 16]},
            "training": {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "batch_size": 10,
                "replay_buffer_size": 100,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.99,
            },
        }
        a = DQNAgent(config)
        a.model(tf.zeros((1, 4)))
        return a

    def test_select_action_returns_valid(self, agent):
        state = np.zeros(4, dtype=np.float32)
        for _ in range(20):
            action = agent.select_action(state)
            assert 0 <= action < 5

    def test_select_action_exploit(self, agent):
        agent.epsilon = 0.0
        state = np.zeros(4, dtype=np.float32)
        actions = [agent.select_action(state) for _ in range(10)]
        assert len(set(actions)) == 1

    def test_epsilon_decay(self, agent):
        agent.epsilon = 1.0
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.99)

    def test_epsilon_minimum(self, agent):
        agent.epsilon = 0.02
        agent.epsilon_min = 0.01
        agent.epsilon_decay = 0.5
        agent.decay_epsilon()
        assert agent.epsilon >= 0.01

    def test_train_step_returns_none_if_buffer_small(self, agent):
        assert agent.train_step() is None

    def _fill_buffer(self, agent, n=20):
        for _ in range(n):
            agent.replay_buffer.add(
                np.zeros(4, dtype=np.float32),
                2,
                1.0,
                np.zeros(4, dtype=np.float32),
                False,
            )

    def test_train_step_runs(self, agent):
        self._fill_buffer(agent, 20)
        loss = agent.train_step()
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_train_step_updates_weights(self, agent):
        self._fill_buffer(agent, 20)
        before = [w.copy() for w in agent.model.get_weights()]
        agent.train_step()
        after = agent.model.get_weights()
        assert any(not np.allclose(b, a) for b, a in zip(before, after))

    def test_train_step_reduces_loss(self, agent):
        for _ in range(50):
            agent.replay_buffer.add(
                np.array([0, 0, 0.01, 0], dtype=np.float32),
                2,
                1.0,
                np.array([0, 0, 0.01, 0], dtype=np.float32),
                False,
            )
        losses = [agent.train_step() for _ in range(50)]
        assert np.mean(losses[-10:]) < np.mean(losses[:10])
