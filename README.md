# Interactive Intelligent Routing Using Reinforcement Learning

A Python mini-project that lets the user enter a source router, destination router, and network congestion level. A Q-learning agent is trained for the selected routing task and its route is compared with Dijkstra's shortest-cost route.

## Features
- User selects any source router from A to I.
- User selects a different destination router from A to I.
- User selects low, medium, or high congestion.
- Q-learning learns a route for the selected input.
- Dijkstra provides a shortest-cost benchmark.
- Console output reports cost, delay, average congestion, and packet loss.
- Training and route-comparison graphs are generated in `results/`.

## Run locally
Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python src/main.py
```

Example input:
```text
Enter source router (A-I): A
Enter destination router (A-I): I
Select network congestion:
1. Low congestion
2. Medium congestion
3. High congestion
Enter choice (1/2/3): 2
```

## Q-learning model
- State: current router.
- Action: move to an adjacent router.
- Reward: negative link cost, with a positive reward for reaching the destination.
- Policy: epsilon-greedy exploration during training.
- Link cost: `delay + 8 * congestion + 20 * packet_loss`.

## Repository layout
```text
intelligent-routing-rl/
├── src/main.py
├── docs/PROJECT_REPORT.md
├── results/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Important limitation
This is a learning-focused network simulation, not a live router or real packet-forwarding implementation. Congestion levels scale the simulated link congestion and packet-loss values. Dijkstra is a deterministic shortest-cost benchmark for the same fixed topology and cost function.
