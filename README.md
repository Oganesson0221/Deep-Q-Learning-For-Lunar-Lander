# Deep Q-Learning for Lunar Landing

This project implements a Deep Q-Learning algorithm to train an agent to successfully land a spacecraft in the **Lunar Lander** environment provided by OpenAI Gym. The implementation utilizes **PyTorch** for neural network construction and training.

---

## Requirements

Before running the code, ensure you have the following libraries installed:

- Python 3.7+
- PyTorch
- NumPy
- OpenAI Gym (with `gymnasium[box2d]`)
- Swig (for environment dependencies)

### Installation

Install the required libraries using the following commands:

```bash
pip install torch numpy gymnasium
pip install "gymnasium[box2d]" "gymnasium[atari, accept-rom-license]"
apt-get install -y swig
```

---

## Files and Components

### 1. **Network Architecture**

- The `Network` class defines a neural network with three fully connected layers:
  - Input layer size: `state_size` (environment observation space)
  - Two hidden layers with 64 neurons each
  - Output layer size: `action_size` (number of possible actions)

### 2. **Experience Replay**

- Implemented through the `ReplayMemory` class:
  - Stores past experiences (state, action, reward, next state, done flag) up to a specified capacity.
  - Enables random sampling of experiences for training to break correlation in observation sequences.

### 3. **Deep Q-Network (DQN) Agent**

- The `Agent` class handles:
  - Interaction with the environment
  - Training using Q-learning
  - Action selection via an epsilon-greedy policy
  - Soft updates for the target Q-network to stabilize learning

---

## Key Hyperparameters

- **Learning Rate**: `5e-4`
- **Batch Size**: `100`
- **Discount Factor (γ)**: `0.99`
- **Replay Buffer Size**: `1e5`
- **Interpolation Parameter (τ)**: `1e-3`
- **Epsilon Decay**: `0.995`

---

## Training Process

### Steps

1. Initialize the environment (`LunarLander-v3`) and the DQN agent.
2. Train the agent over a specified number of episodes.
3. Update the agent's policy using sampled experiences from the replay buffer.
4. Evaluate performance based on average scores over the last 100 episodes.

### Key Metrics

- Success is defined as achieving an average score of `200` over 100 consecutive episodes.

---

## Usage

### Run the Code

Execute the script to train the agent in the Lunar Lander environment. Progress will be printed as:

```plaintext
Episode X    Average Score: Y
```

### Termination

The training stops if:

1. The maximum number of episodes (`2000`) is reached.
2. The environment is solved (average score ≥ `200`).

---

## Results

After successful training:

- The trained agent can land the spacecraft with optimal precision.
- Model weights and training scores can be saved for evaluation or further analysis.

---

## Improvements and Extensions

- **Optimization**: Experiment with different architectures and hyperparameters.
- **Evaluation**: Add visualization to assess the agent's performance in the environment.
- **Transfer Learning**: Apply the trained model to similar environments.

Feel free to explore and extend the implementation!
