### README: Deep Q-Learning for Lunar Lander

---

## **Project Overview**

This repository implements a **Deep Q-Learning (DQL)** algorithm to solve the **Lunar Lander** environment from OpenAI Gym. The goal is to control a lunar lander and safely land it on the moon's surface by learning an optimal policy. The agent interacts with the environment and learns from its experiences using neural networks to approximate Q-values.

---

## **Environment: Lunar Lander**

### **Environment Description**

- **Objective**: Safely land the lunar module on the landing pad.
- **State Space**:

  - An 8-dimensional vector representing the lander's state:
    1. X and Y positions.
    2. X and Y velocities.
    3. Angle of the lander.
    4. Angular velocity.
    5. Boolean values indicating whether the left or right leg is in contact with the ground.

- **Action Space**:

  - 4 discrete actions:
    1. Do nothing.
    2. Fire the left orientation engine.
    3. Fire the main engine.
    4. Fire the right orientation engine.

- **Reward**:

  - Positive rewards for landing on the pad.
  - Negative rewards for crashing or moving further away.
  - Small penalty for using fuel to encourage efficiency.

- **Episode Termination**:
  - The agent crashes or lands successfully.

### **Solved Criteria**

- The environment is considered solved when the average score over 100 episodes is **≥ 200**.

---

## **Deep Q-Learning Algorithm**

### **Key Components**

1. **Q-Learning**:

   - An off-policy reinforcement learning algorithm that estimates the Q-value function to learn optimal action policies.
   - Uses the Bellman Equation:  
     \[
     Q(s, a) = r + \gamma \max_a Q(s', a)
     \]
     where \( \gamma \) is the discount factor.

2. **Deep Q-Network (DQN)**:

   - A neural network is used to approximate Q-values for each state-action pair.
   - Architecture:
     - Input layer: State vector (8 dimensions).
     - Hidden layers: Two layers with 64 neurons each and ReLU activation.
     - Output layer: Q-values for the 4 actions.

3. **Experience Replay**:

   - Stores past experiences \((s, a, r, s')\) in a replay buffer.
   - Randomly samples mini-batches from the buffer to break correlations and improve stability.

4. **Target Network**:

   - A separate network is used to compute target Q-values, updated periodically to stabilize training.

5. **Epsilon-Greedy Policy**:
   - Balances exploration and exploitation:
     - With probability \(\epsilon\): Choose a random action.
     - With probability \(1-\epsilon\): Choose the action with the highest Q-value.

---

## **Implementation Steps**

1. **Setup Environment**:

   - Initialize the Lunar Lander environment from OpenAI Gym.

2. **Initialize Networks**:

   - Create a Q-network and a target network with the same architecture.

3. **Training Loop**:

   - Reset the environment at the start of each episode.
   - Use the epsilon-greedy policy to select actions.
   - Store experiences in the replay buffer.
   - Sample mini-batches from the buffer to train the Q-network.
   - Periodically update the target network.

4. **Termination Criteria**:
   - Stop training when the average score over 100 episodes is ≥ 200.

---

## **Dependencies**

Install the following packages:

```bash
pip install numpy tensorflow gym matplotlib
```

---

## **Usage**

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Train the agent:

   ```bash
   python train.py
   ```

3. Test the trained agent:

   ```bash
   python test.py
   ```

4. Visualize training progress:

   - Training rewards will be plotted after the script completes.

---

## **Results**

- **Average Score**: The agent consistently achieves an average score of **≥ 200** over 100 episodes.
- **Training Time**: Depends on hardware but typically requires a few thousand episodes.

---

## **Future Improvements**

1. **Double DQN**:
   - Mitigate overestimation of Q-values by decoupling action selection and evaluation.
2. **Dueling DQN**:

   - Separate state-value estimation and action-advantage estimation for better performance.

3. **Prioritized Experience Replay**:
   - Sample experiences based on their importance (i.e., TD error).

---

## **Acknowledgments**

This project was inspired by:

- OpenAI Gym for providing the Lunar Lander environment.
- Deep reinforcement learning papers and tutorials.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
