import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.q_network.environment import InvertedPendulumEnv


@pytest.fixture
def env():
    return InvertedPendulumEnv()


class TestReset:
    def test_reset_returns_valid_state(self, env):
        state = env.reset()
        assert state.shape == (4,)
        assert state.dtype == np.float32

    def test_reset_values_in_range(self, env):
        for _ in range(50):
            state = env.reset()
            assert np.all(state <= 0.05 + 1e-7)
            assert np.all(state >= -0.05 - 1e-7)

    def test_reset_resets_step_count(self, env):
        env.reset()
        env.step(2)
        assert env.step_count == 1
        env.reset()
        assert env.step_count == 0


class TestStep:
    def test_step_returns_tuple(self, env):
        env.reset()
        state, reward, done, info = env.step(2)
        assert state.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_count(self, env):
        env.reset()
        env.step(2)
        assert env.step_count == 1
        env.step(2)
        assert env.step_count == 2

    def test_invalid_action_raises(self, env):
        env.reset()
        with pytest.raises(AssertionError):
            env.step(5)
        with pytest.raises(AssertionError):
            env.step(-1)

    def test_all_actions_valid(self, env):
        for action in range(env.action_size):
            env.reset()
            state, reward, done, info = env.step(action)
            assert np.isfinite(state).all()
            assert np.isfinite(reward)


class TestPhysics:
    def test_no_force_pole_falls(self, env):
        env.state = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float32)
        initial = abs(env.state[2])
        for _ in range(10):
            state, *_ = env.step(2)
        assert abs(state[2]) > initial

    def test_symmetry(self):
        env1 = InvertedPendulumEnv()
        env2 = InvertedPendulumEnv()
        env1.state = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float32)
        env2.state = np.array([0.0, 0.0, -0.05, 0.0], dtype=np.float32)
        s1, *_ = env1.step(4)
        s2, *_ = env2.step(0)
        assert np.allclose(s1[[0, 2]], -s2[[0, 2]], atol=1e-6)

    def test_force_table_mapping(self, env):
        assert env.FORCE_TABLE == [-10.0, -5.0, 0.0, 5.0, 10.0]


class TestTermination:
    def test_terminal_angle(self, env):
        env.state = np.array([0.0, 0.0, 0.21, 5.0], dtype=np.float32)
        _, _, done, _ = env.step(2)
        assert done is True

    def test_terminal_position(self, env):
        env.state = np.array([2.5, 0.0, 0.0, 0.0], dtype=np.float32)
        _, _, done, _ = env.step(2)
        assert done is True

    def test_max_steps_terminates(self, env):
        env.reset()
        done = False
        for _ in range(500):
            _, _, done, _ = env.step(2)
            if done:
                break
        assert done is True

    def test_death_penalty(self, env):
        env.state = np.array([2.5, 0.0, 0.0, 0.0], dtype=np.float32)
        _, reward, done, _ = env.step(2)
        assert done is True
        assert reward == -100.0


class TestReward:
    def test_alive_reward_positive(self, env):
        reward = env._compute_reward(0.0, 0.0, 0.0, 0.0, False)
        assert reward > 0.5

    def test_angle_penalty_dominates(self, env):
        reward0 = env._compute_reward(0.0, 0.0, 0.0, 0.0, False)
        reward1 = env._compute_reward(0.0, 0.0, 0.1, 0.0, False)
        assert reward1 < reward0
