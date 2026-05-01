# Inverted Pendulum DQN

A self-contained Deep Q-Network implementation for balancing an inverted pendulum / cart-pole system using TensorFlow 2.x.

This project implements the full reinforcement learning pipeline from scratch:

- Custom cart-pole physics simulation
- Discrete-action inverted pendulum environment
- Deep Q-Network agent with replay buffer
- Optional target network for stable DQN training
- YAML-based configuration
- Training metrics and plots
- Trained-agent visualization
- Simulation recording to JSON
- Multi-run comparison plots
- Unit tests for the environment, model, and agent

The project does **not** depend on OpenAI Gym. The cart-pole dynamics are implemented directly from Lagrangian-derived equations of motion.

---

## Project Structure

```text
inverted-pendulum-dqn/
├── pyproject.toml
├── requirements.txt
├── README.md
├── readme_handling.md
├── configs/
│   ├── default.yaml
│   ├── sim_config.yaml
│   └── compare_config.yaml
├── src/
│   ├── __init__.py
│   └── q_network/
│       ├── __init__.py
│       ├── environment.py
│       ├── model.py
│       ├── agent.py
│       ├── train.py
│       ├── visualize.py
│       └── compare.py
├── tests/
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_model.py
│   ├── test_agent.py
│   └── visual_test_physics.py
├── results/
└── sim_results/
```

---

## What This Project Does

The goal is to train a DQN agent to keep an inverted pendulum upright by applying discrete horizontal forces to a cart.

The state is a 4-dimensional continuous vector:

```text
[x, x_dot, theta, theta_dot]
```

where:

- `x` is the cart position
- `x_dot` is the cart velocity
- `theta` is the pole angle from vertical
- `theta_dot` is the angular velocity

The action space is discrete:

```python
[-10.0, -5.0, 0.0, 5.0, 10.0]
```

Each action applies a horizontal force in Newtons to the cart.

---

## Main Features

### Custom Physics Environment

The environment is implemented in `src/q_network/environment.py`.

It includes:

- Semi-implicit Euler integration
- Configurable cart mass, pole mass, gravity, timestep, and force limits
- Configurable terminal conditions
- Reward shaping for angle, position, and velocity
- No Gym dependency

### Deep Q-Network Agent

The agent is implemented in `src/q_network/agent.py`.

It includes:

- Experience replay buffer
- Epsilon-greedy action selection
- Bellman target calculation
- Optional target network
- JSON and Keras model saving support

### Neural Network

The default Q-network architecture is:

```text
Input(4)
  → Dense(128, relu)
  → Dense(64, relu)
  → Dense(5)
```

The output contains one Q-value per discrete action.

Default parameter count:

```text
9,221 parameters
```

---

## Requirements

Recommended platform:

- Python `>=3.10,<3.13`
- Poetry
- TensorFlow 2.x CPU
- NumPy
- Matplotlib
- PyYAML
- Pytest

The project is designed to run on CPU. No CUDA or GPU setup is required.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/inverted-pendulum-dqn.git
cd inverted-pendulum-dqn
```

### 2. Install with Poetry

```bash
poetry install
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
poetry shell
```

or run commands through Poetry:

```powershell
poetry run python --version
```

---

## Running Tests

Run all tests:

```bash
python -m pytest tests/ -v
```

Run only environment tests:

```bash
python -m pytest tests/test_environment.py -v
```

Run only model tests:

```bash
python -m pytest tests/test_model.py -v
```

Run only agent tests:

```bash
python -m pytest tests/test_agent.py -v
```

---

## Training

Start training with the default configuration:

```bash
python -m src.q_network.train
```

Use a custom config file:

```bash
python -m src.q_network.train --config configs/default.yaml
```

Override training parameters from the command line:

```bash
python -m src.q_network.train --episodes 1000 --lr 0.0005 --gamma 0.99 --batch-size 64
```

Train without a target network:

```bash
python -m src.q_network.train --no-target
```

Training outputs are saved under:

```text
results/
```

A typical training run creates files like:

```text
results/train_target_2000ep_YYYYMMDD_HHMMSS/
├── trained_model.keras
├── trained_model_final.keras
├── checkpoint_ep100.keras
├── weights_target_2000ep.json
├── training_metrics.json
├── training_metrics.png
├── reward_plot.png
├── loss_plot.png
└── episode_length_plot.png
```

The final JSON weights are also copied to the root `results/` directory.

Example:

```text
results/weights_target_2000ep.json
```

---

## Important Note About Visualization

Visualization requires trained weights.

If you run:

```bash
python -m src.q_network.visualize
```

before training, you may see an error similar to:

```text
FileNotFoundError: results/weights_target_network_1000ep.json
```

This means the default simulation config points to a weights file that does not exist yet.

You have two options.

### Option 1: Train first

```bash
python -m src.q_network.train
```

Then run visualization with the generated weights file:

```bash
python -m src.q_network.visualize --model results/weights_target_2000ep.json
```

### Option 2: Edit the simulation config

Open:

```text
configs/sim_config.yaml
```

and set:

```yaml
simulation:
  weights_file: results/YOUR_WEIGHTS_FILE.json
```

---

## Visualizing a Trained Agent

Run the default visualization:

```bash
python -m src.q_network.visualize
```

Use a specific trained model:

```bash
python -m src.q_network.visualize --model results/weights_target_2000ep.json
```

Run more simulation episodes:

```bash
python -m src.q_network.visualize --model results/weights_target_2000ep.json --episodes 5
```

Save a GIF:

```bash
python -m src.q_network.visualize --model results/weights_target_2000ep.json --save-gif results/pendulum.gif
```

Record simulation data to JSON:

```bash
python -m src.q_network.visualize --model results/weights_target_2000ep.json --record sim_results/simulation.json
```

Record every step:

```bash
python -m src.q_network.visualize --model results/weights_target_2000ep.json --record sim_results/simulation.json --record-interval 1
```

---

## Drawing Training Plots

Training automatically creates plots in the corresponding run folder:

```text
training_metrics.png
reward_plot.png
loss_plot.png
episode_length_plot.png
```

Example:

```text
results/train_target_2000ep_YYYYMMDD_HHMMSS/training_metrics.png
```

These plots show:

- Episode reward
- Training loss
- Episode length
- Moving averages

---

## Comparing Two Simulation Runs

The comparison tool reads two recorded simulation JSON files and draws a 4-panel comparison plot:

- Pole angle
- Cart position
- Cart velocity
- Angular velocity

First, generate two simulation recordings:

```bash
python -m src.q_network.visualize --model results/weights_target_1000ep.json --record sim_results/run_1000ep.json
python -m src.q_network.visualize --model results/weights_target_2000ep.json --record sim_results/run_2000ep.json
```

Then edit:

```text
configs/compare_config.yaml
```

Example:

```yaml
compare:
  run_1:
    path: sim_results/run_1000ep.json
    label: "1000 episodes"

  run_2:
    path: sim_results/run_2000ep.json
    label: "2000 episodes"

  time_start: 0.0
  time_end: 10.0

  save_path: results/comparison_plot.png
  dpi: 150
  figure_size: [14, 10]
```

Run the comparison:

```bash
python -m src.q_network.compare
```

The output image is saved to:

```text
results/comparison_plot.png
```

---

## Visual Physics Test

A real-time random-action physics visualization is available:

```bash
python tests/visual_test_physics.py
```

This is not a pytest test. It opens a Matplotlib window showing the cart-pole physics and state history.

---

## Configuration

Most behavior is controlled through YAML files.

### Training Config

```text
configs/default.yaml
```

Controls:

- Physics parameters
- Environment limits
- Reward shaping
- Network architecture
- Training hyperparameters
- Target network behavior
- Save intervals

### Simulation Config

```text
configs/sim_config.yaml
```

Controls:

- Which weights file to load
- Number of simulation episodes
- GIF output
- JSON recording path
- Visualization settings

### Comparison Config

```text
configs/compare_config.yaml
```

Controls:

- Simulation files to compare
- Labels
- Time window
- Output image path

---

## DQN Algorithm

The project implements Deep Q-Learning with experience replay.

For each sampled transition:

```text
(state, action, reward, next_state, done)
```

the Bellman target is:

```text
Y = reward + gamma * (1 - done) * max_a Q_target(next_state, a)
```

The online network predicts:

```text
X = Q_online(state, action)
```

The loss is mean squared error:

```text
loss = mean((X - Y)^2)
```

When the target network is enabled, it is periodically synchronized with the online network:

```text
target_network ← online_network
```

This improves training stability by keeping the Bellman target fixed for several episodes.

---

## Common Commands

### Install

```bash
poetry install
```

### Run tests

```bash
python -m pytest tests/ -v
```

### Train

```bash
python -m src.q_network.train
```

### Train for 500 episodes

```bash
python -m src.q_network.train --episodes 500
```

### Visualize trained model

```bash
python -m src.q_network.visualize --model results/weights_target_500ep.json
```

### Save GIF

```bash
python -m src.q_network.visualize --model results/weights_target_500ep.json --save-gif results/demo.gif
```

### Record simulation

```bash
python -m src.q_network.visualize --model results/weights_target_500ep.json --record sim_results/demo.json
```

### Compare simulations

```bash
python -m src.q_network.compare
```

### Run visual physics test

```bash
python tests/visual_test_physics.py
```

---

## Troubleshooting

### TensorFlow oneDNN message

You may see this message:

```text
oneDNN custom operations are on
```

This is normal and can usually be ignored.

To disable it on Windows PowerShell:

```powershell
$env:TF_ENABLE_ONEDNN_OPTS="0"
python -m src.q_network.train
```

### Missing weights file during visualization

If visualization fails with:

```text
FileNotFoundError
```

train the agent first or pass an existing weights file:

```bash
python -m src.q_network.visualize --model results/YOUR_WEIGHTS_FILE.json
```

### Matplotlib window does not open

Use GIF output instead:

```bash
python -m src.q_network.visualize --model results/YOUR_WEIGHTS_FILE.json --save-gif results/demo.gif
```

---

## Development Notes

This project intentionally keeps the code simple and educational:

- No Gym dependency
- No GPU-specific code
- No distributed training
- Small neural network
- YAML-driven experiments
- JSON simulation recordings for easy plotting and comparison

The goal is to make the DQN pipeline easy to inspect, debug, and modify.

---

## License

Add your license here.

Example:

```text
MIT License
```

---

## Acknowledgements

This project is inspired by classic cart-pole reinforcement learning examples and implements a Deep Q-Network approach for inverted pendulum control using TensorFlow/Keras.
