import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from src.q_network import build_q_network


class TestModelArchitecture:
    def test_model_output_shape(self):
        model = build_q_network()
        y = model(tf.zeros((1, 4)))
        assert y.shape == (1, 5)

    def test_model_batch_output_shape(self):
        model = build_q_network()
        y = model(tf.zeros((32, 4)))
        assert y.shape == (32, 5)

    def test_model_parameter_count(self):
        model = build_q_network(hidden_layers=[128, 64])
        model(tf.zeros((1, 4)))
        assert model.count_params() == 9221

    def test_model_custom_layers(self):
        model = build_q_network(action_size=3, hidden_layers=[64, 32, 16])
        y = model(tf.zeros((1, 4)))
        assert y.shape == (1, 3)

    def test_forward_pass_no_nan(self):
        model = build_q_network()
        for _ in range(10):
            y = model(tf.random.normal((16, 4)))
            assert not np.isnan(y.numpy()).any()
            assert np.isfinite(y.numpy()).all()

    def test_output_unbounded(self):
        model = build_q_network()
        y = model(tf.constant([[100.0, 100.0, 100.0, 100.0]]))
        assert y.shape == (1, 5)

    def test_model_is_differentiable(self):
        model = build_q_network()
        x = tf.random.normal((8, 4))
        with tf.GradientTape() as tape:
            y = model(x)
            loss = tf.reduce_mean(y ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in grads)
