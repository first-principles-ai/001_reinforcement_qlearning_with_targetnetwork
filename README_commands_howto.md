# inverted-pendulum-dqn

Deep Q-Network for inverted pendulum control with a self-contained cart-pole physics simulation, TensorFlow/Keras model, training loop, visualization, comparison plots, and tests.

## Quick start

```bash
poetry install
poetry run python -m pytest tests/ -v
poetry run python -m src.q_network.train
poetry run python -m src.q_network.visualize
poetry run python -m src.q_network.compare
```
