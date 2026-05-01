# Handling Guide — Train, Simulate, and Draw Plots

This guide lists the commands needed to set up the project, train the DQN agent, run a simulation/animation, record simulation data, and draw comparison pictures.

## 1. Open a terminal in the project folder

```powershell
cd inverted-pendulum-dqn
```

On Linux/macOS:

```bash
cd inverted-pendulum-dqn
```

## 2. Install dependencies

Recommended setup with Poetry:

```powershell
poetry config virtualenvs.in-project true
poetry install
```

Activate the Poetry virtual environment on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Alternative pip setup:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS pip activation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run tests

Run all unit tests:

```powershell
python -m pytest tests/ -v
```

Run only one test module:

```powershell
python -m pytest tests/test_environment.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_agent.py -v
```

Run the visual physics test:

```powershell
python tests/visual_test_physics.py
```

This opens a live Matplotlib window with a random-action cart-pole simulation.

## 4. Train the agent

Train with the default configuration:

```powershell
python -m src.q_network.train
```

Train with an explicit config file:

```powershell
python -m src.q_network.train --config configs/default.yaml
```

Train for a custom number of episodes:

```powershell
python -m src.q_network.train --episodes 500
```

Train with custom learning rate, discount factor, and batch size:

```powershell
python -m src.q_network.train --episodes 1000 --lr 0.0005 --gamma 0.99 --batch-size 64
```

Train without the target network, using the basic DQN variant:

```powershell
python -m src.q_network.train --no-target
```

Training writes results to a timestamped folder under:

```text
results/
```

Typical output files are:

```text
results/<run_folder>/trained_model.keras
results/<run_folder>/trained_model_final.keras
results/<run_folder>/weights_target_<episodes>ep.json
results/<run_folder>/training_metrics.json
results/<run_folder>/training_metrics.png
results/<run_folder>/reward_plot.png
results/<run_folder>/loss_plot.png
results/<run_folder>/episode_length_plot.png
```

The JSON weights are also copied to the root `results/` directory for easy simulation loading.

## 5. Start simulation / visualization

Run the trained agent using the default simulation configuration:

```powershell
python -m src.q_network.visualize
```

Run simulation with an explicit config file:

```powershell
python -m src.q_network.visualize --config configs/sim_config.yaml
```

Run simulation with a specific trained weights file:

```powershell
python -m src.q_network.visualize --model results/weights_target_1000ep.json
```

Run a fixed number of simulation episodes:

```powershell
python -m src.q_network.visualize --episodes 3
```

Save a GIF animation:

```powershell
python -m src.q_network.visualize --model results/weights_target_1000ep.json --save-gif results/simulation.gif
```

Record simulation state data to JSON:

```powershell
python -m src.q_network.visualize --model results/weights_target_1000ep.json --record sim_results/sim_weights_target_1000ep.json
```

Record every simulation step:

```powershell
python -m src.q_network.visualize --model results/weights_target_1000ep.json --record sim_results/sim_weights_target_1000ep.json --record-interval 1
```

The simulation JSON can later be used by the comparison plot tool.

## 6. Draw the training pictures

Training automatically creates these pictures in the run folder:

```text
training_metrics.png
reward_plot.png
loss_plot.png
episode_length_plot.png
```

To generate them, simply run training:

```powershell
python -m src.q_network.train
```

After training, open the generated PNG files from the newest folder inside:

```text
results/
```

## 7. Draw a comparison picture between two simulations

First, create two simulation recordings:

```powershell
python -m src.q_network.visualize --model results/weights_target_1000ep.json --record sim_results/sim_weights_target_1000ep.json
python -m src.q_network.visualize --model results/weights_target_2000ep.json --record sim_results/sim_weights_target_2000ep.json
```

Then edit `configs/compare_config.yaml` so that `run_1.path` and `run_2.path` point to those two JSON files.

Run the comparison plot command:

```powershell
python -m src.q_network.compare
```

Or explicitly pass the comparison config:

```powershell
python -m src.q_network.compare --config configs/compare_config.yaml
```

By default, this creates:

```text
results/comparison_plot.png
```

The comparison plot contains four panels:

```text
1. Pole angle θ in degrees
2. Cart position x in meters
3. Cart velocity x_dot in m/s
4. Pole angular velocity theta_dot in rad/s
```

## 8. Typical complete workflow

A common end-to-end workflow is:

```powershell
cd inverted-pendulum-dqn
poetry config virtualenvs.in-project true
poetry install
.venv\Scripts\activate
python -m pytest tests/ -v
python -m src.q_network.train --episodes 1000
python -m src.q_network.visualize --model results/weights_target_1000ep.json --episodes 3 --record sim_results/sim_weights_target_1000ep.json
python -m src.q_network.compare --config configs/compare_config.yaml
```

## 9. Important configuration files

Main training config:

```text
configs/default.yaml
```

Simulation config:

```text
configs/sim_config.yaml
```

Comparison plot config:

```text
configs/compare_config.yaml
```

Adjust these files when you want to change physics parameters, training hyperparameters, network architecture, simulation weights, recording paths, or output picture paths.

## 10. Notes

- Use Python 3.10, 3.11, or 3.12. The project is configured for `>=3.10,<3.13`.
- TensorFlow is used in CPU mode; no CUDA/GPU setup is required.
- If simulation fails because the weights file does not exist, train first or update `configs/sim_config.yaml` to point to an existing weights file.
- If comparison plotting fails because simulation JSON files do not exist, run `visualize` with `--record` first.
