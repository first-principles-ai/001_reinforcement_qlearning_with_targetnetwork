"""
DQN Q-Network — TensorFlow/Keras Implementation (CPU-only)

Small fully-connected network: 4 → 128 → 64 → 5
Approximates Q(s, a) for the inverted pendulum control task.
No GPU required — model has only 9,221 parameters.
"""

import tensorflow as tf


def build_q_network(state_size=4, action_size=5, hidden_layers=None,
                    activation="relu", initializer="he_normal"):
    if hidden_layers is None:
        hidden_layers = [128, 64]

    layers = [tf.keras.layers.Input(shape=(state_size,))]
    for units in hidden_layers:
        layers.append(
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_initializer=initializer,
            )
        )
    layers.append(
        tf.keras.layers.Dense(
            action_size,
            activation=None,
            kernel_initializer=initializer,
        )
    )
    return tf.keras.Sequential(layers)
