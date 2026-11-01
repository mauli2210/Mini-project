"""
Interactive Intelligent Routing Using Reinforcement Learning (Q-learning)

Run:
    python src/main.py

The user can choose any valid source and destination router and select
low, medium, or high congestion. The RL agent is trained for that request.
"""
from __future__ import annotations
import random
import heapq
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
ALPHA, GAMMA, EPSILON = 0.15, 0.90, 0.25
EPISODES, MAX_STEPS = 5000, 25
NODES = list("ABCDEFGHI")
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# Base links: neighbor: (delay in ms, congestion 0..1, packet loss 0..1)
BASE_NETWORK = {
    "A": {"B": (2, .15, .01), "C": (4, .05, .00)},
    "B": {"A": (2, .15, .01), "D": (2, .25, .02), "E": (3, .10, .01)},
    "C": {"A": (4, .05, .00), "D": (2, .10, .01), "F": (3, .20, .02)},
    "D": {"B": (2, .25, .02), "C": (2, .10, .01), "E": (2, .65, .08), "G": (3, .10, .01)},
    "E": {"B": (3, .10, .01), "D": (2, .65, .08), "H": (3, .15, .01)},
    "F": {"C": (3, .20, .02), "G": (2, .10, .01), "I": (3, .05, .00)},
    "G": {"D": (3, .10, .01), "F": (2, .10, .01), "H": (2, .20, .02)},
    "H": {"E": (3, .15, .01), "G": (2, .20, .02), "I": (2, .10, .01)},
    "I": {"F": (3, .05, .00), "H": (2, .10, .01)},
}

CONGESTION_LEVELS = {
    "1": ("Low", 0.55),
    "2": ("Medium", 1.0),
    "3": ("High", 1.65),
}

def build_network(congestion_factor: float):
    """Create a symmetric network and scale congestion/loss by selected level."""
    network = {node: {} for node in BASE_NETWORK}
    for node, neighbors in BASE_NETWORK.items():
        for neighbor, (delay, congestion, loss) in neighbors.items():
            scaled_congestion = min(0.99, congestion * congestion_factor)
            scaled_loss = min(0.50, loss * congestion_factor)
            network[node][neighbor] = (delay, scaled_congestion, scaled_loss)
    return network

def edge_cost(attrs):
    delay, congestion, loss = attrs
    return delay + 8.0 * congestion + 20.0 * loss

def dijkstra(network, start, goal):
    queue = [(0.0, start, [start])]
    best = {start: 0.0}
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == goal:
            return path, cost
        if cost > best.get(node, float("inf")):
            continue
        for nxt, attrs in network[node].items():
            new_cost = cost + edge_cost(attrs)
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                heapq.heappush(queue, (new_cost, nxt, path + [nxt]))
    return [], float("inf")

def train_agent(network, start, goal):
    q = {node: {neighbor: 0.0 for neighbor in network[node]} for node in network}
    rewards = []
    for _ in range(EPISODES):
        state = start
        total_reward = 0.0
        visited = {start}
        for _ in range(MAX_STEPS):
            actions = list(network[state])
            if random.random() < EPSILON:
                action = random.choice(actions)
            else:
                max_q = max(q[state][a] for a in actions)
                candidates = [a for a in actions if q[state][a] == max_q]
                action = random.choice(candidates)

            cost = edge_cost(network[state][action])
            reward = -cost
            done = action == goal
            if done:
                reward += 25.0
                future = 0.0
            else:
                future = max(q[action].values()) if q[action] else 0.0
                # Small penalty discourages wandering around loops.
                if action in visited:
                    reward -= 2.0
            q[state][action] += ALPHA * (reward + GAMMA * future - q[state][action])
            total_reward += reward
            state = action
            visited.add(state)
            if done:
                break
        rewards.append(total_reward)
    return q, rewards

def extract_route(network, q, start, goal):
    state = start
    route = [state]
    visited = {state}
    for _ in range(len(network) * 2):
        if state == goal:
            return route
        actions = [a for a in network[state] if a not in visited]
        if not actions:
            return []
        action = max(actions, key=lambda a: q[state][a])
        route.append(action)
        visited.add(action)
        state = action
    return route if route[-1] == goal else []

def metrics(network, route):
    if not route:
        return {"cost": float("inf"), "delay": 0, "congestion": 0, "loss": 0}
    links = [network[a][b] for a, b in zip(route, route[1:])]
    return {
        "cost": sum(edge_cost(x) for x in links),
        "delay": sum(x[0] for x in links),
        "congestion": sum(x[1] for x in links) / len(links),
        "loss": sum(x[2] for x in links) / len(links),
    }

def save_plots(rewards, rl_route, baseline_route, network):
    plt.figure(figsize=(9, 4.8))
    plt.plot(rewards, alpha=.25, label="Episode reward")
    window = 100
    if len(rewards) >= window:
        moving = np.convolve(rewards, np.ones(window)/window, mode="valid")
        plt.plot(range(window-1, len(rewards)), moving, linewidth=2, label="100-episode average")
    plt.title("Q-learning Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (higher is better)")
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "training_rewards.png", dpi=160)
    plt.close()

    costs = [metrics(network, rl_route)["cost"], metrics(network, baseline_route)["cost"]]
    plt.figure(figsize=(7, 4.5))
    plt.bar(["RL learned route", "Dijkstra baseline"], costs)
    plt.ylabel("Total link cost (lower is better)")
    plt.title("Routing Cost Comparison")
    plt.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(RESULTS / "routing_comparison.png", dpi=160)
    plt.close()

def get_user_input():
    print("\nAvailable routers:", ", ".join(NODES))
    while True:
        start = input("Enter source router (A-I): ").strip().upper()
        if start in NODES:
            break
        print("Invalid source. Please enter one letter from A to I.")
    while True:
        goal = input("Enter destination router (A-I): ").strip().upper()
        if goal in NODES and goal != start:
            break
        if goal == start:
            print("Source and destination must be different.")
        else:
            print("Invalid destination. Please enter one letter from A to I.")
    print("\nSelect network congestion:")
    print("1. Low congestion")
    print("2. Medium congestion")
    print("3. High congestion")
    while True:
        choice = input("Enter choice (1/2/3): ").strip()
        if choice in CONGESTION_LEVELS:
            break
        print("Please enter 1, 2, or 3.")
    return start, goal, CONGESTION_LEVELS[choice]

def main():
    print("=" * 65)
    print("     INTERACTIVE INTELLIGENT ROUTING USING Q-LEARNING")
    print("=" * 65)
    start, goal, (level_name, factor) = get_user_input()
    network = build_network(factor)

    print(f"\nTraining agent for {start} -> {goal} under {level_name} congestion...")
    random.seed(SEED)
    np.random.seed(SEED)
    q, rewards = train_agent(network, start, goal)
    rl_route = extract_route(network, q, start, goal)
    baseline_route, _ = dijkstra(network, start, goal)

    print("\n" + "-" * 65)
    print(f"Source: {start} | Destination: {goal} | Congestion: {level_name}")
    print(f"Training episodes: {EPISODES}")
    for label, route in [("RL learned route", rl_route), ("Dijkstra baseline", baseline_route)]:
        result = metrics(network, route)
        print(f"\n{label}: {' -> '.join(route) if route else 'No route found'}")
        print(f"  Combined cost  : {result['cost']:.2f}")
        print(f"  Total delay    : {result['delay']:.2f} ms")
        print(f"  Average congestion: {result['congestion']:.3f}")
        print(f"  Average packet loss: {result['loss']:.3f}")
    save_plots(rewards, rl_route, baseline_route, network)
    print(f"\nUpdated graphs saved in: {RESULTS}")
    print("Simulation completed successfully.")
    print("-" * 65)

if __name__ == "__main__":
    main()
