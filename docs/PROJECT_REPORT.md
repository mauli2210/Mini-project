# MINI PROJECT REPORT
## INTERACTIVE INTELLIGENT ROUTING USING REINFORCEMENT LEARNING

**Submitted by:** [Your Name]  
**Enrollment No.:** [Your Enrollment Number]  
**Programme:** B.Tech – Computer Science and Engineering  
**Institute:** [Institute Name]  
**Academic Year:** 2026–2027  
**Guide:** [Faculty Guide Name]

## Abstract
This project implements an interactive simulation of intelligent network routing using Reinforcement Learning (RL), specifically Q-learning. The user can enter a source router, destination router, and congestion level (low, medium, or high). The simulated network contains routers A to I connected through links with delay, congestion, and packet-loss values. A Q-learning agent is trained for the selected input and learns a route by receiving rewards and penalties. The learned route is compared with a Dijkstra shortest-cost baseline. The system reports route cost, delay, average congestion, and packet loss, and generates graphs to visualize training and route-cost comparison.

## 1. Introduction
Routing selects a path for data to travel through interconnected routers. Network conditions such as delay, congestion, and packet loss affect the quality of communication. Reinforcement Learning enables an agent to learn decisions by interacting with an environment and receiving feedback. This project applies Q-learning to a simulated network and makes the routing process interactive through user input.

## 2. Problem Statement
To create an interactive routing simulation where a user can choose source, destination, and congestion level, and a Q-learning agent learns a low-cost path through the network.

## 3. Objectives
1. Model a network containing routers A to I.
2. Accept source and destination router inputs.
3. Allow selection of low, medium, or high congestion.
4. Apply Q-learning to learn routing decisions.
5. Compare the learned route with Dijkstra's algorithm.
6. Display route metrics and generate graphs.

## 4. Technologies Used
- Python
- NumPy
- Matplotlib
- Q-learning
- Dijkstra shortest-path algorithm

## 5. System Design and Methodology
### 5.1 User Input
The program asks the user to enter a source router and a different destination router. It then asks the user to select low, medium, or high congestion.

### 5.2 State and Action
The current router is the state. Moving to one of the directly connected neighboring routers is an action.

### 5.3 Reward Function
The link cost is calculated as:

`Cost = Delay + 8 × Congestion + 20 × Packet Loss`

The agent receives the negative link cost as a reward. A positive reward is added when the destination is reached.

### 5.4 Q-learning Update
`Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]`

Here, alpha is the learning rate, gamma is the discount factor, and r is the reward.

### 5.5 Training and Evaluation
The Q-learning agent trains for 5,000 episodes for the selected source and destination. The final route is extracted from the learned Q-table. Dijkstra is also run on the same network and cost values to provide a baseline comparison.

## 6. Implementation
The implementation is contained in `src/main.py`. It includes:
- Network topology and congestion-level configuration.
- Input validation for source and destination.
- Q-learning training.
- Route extraction.
- Dijkstra baseline.
- Metric calculation and graph generation.

Run:
`python -m pip install -r requirements.txt`
`python src/main.py`

## 7. Results and Discussion
The program displays the selected inputs, the Q-learning route, the Dijkstra route, combined cost, total delay, average congestion, and average packet loss. It saves `training_rewards.png` and `routing_comparison.png` in the results folder.

**Insert screenshots of your actual terminal output and generated graphs here after testing different source/destination pairs and congestion levels. Do not claim results for inputs you have not tested.**

## 8. Conclusion
The project demonstrates an interactive application of Q-learning to network routing. Users can vary the source, destination, and congestion level and observe the resulting routing decisions. Dijkstra provides a conventional shortest-cost benchmark. The project helps explain how reinforcement learning can be applied to routing optimization.

## 9. Limitations and Future Scope
This is a small educational simulation with fixed network topology and scaled congestion values. It does not forward real network packets. Future work may include dynamic traffic, larger topologies, multiple simultaneous flows, Deep Q-Networks, and packet-level network simulation.

## References
1. C. J. C. H. Watkins and P. Dayan, “Q-learning,” Machine Learning, vol. 8, pp. 279–292, 1992.
2. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
3. E. W. Dijkstra, “A note on two problems in connexion with graphs,” Numerische Mathematik, vol. 1, pp. 269–271, 1959.
4. Python documentation: https://docs.python.org/3/
5. NumPy documentation: https://numpy.org/doc/
6. Matplotlib documentation: https://matplotlib.org/stable/
