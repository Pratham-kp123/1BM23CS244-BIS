import random
import numpy as np

class AntColony:
    def __init__(self, graph, num_ants, num_iterations, alpha, beta, evaporation_rate, q0):
        self.graph = graph
        self.num_nodes = len(graph)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q0 = q0
        self.pheromone = np.ones_like(graph) * 0.1

    def run(self, start_node, end_node):
        best_path = None
        best_cost = float('inf')
        for _ in range(self.num_iterations):
            all_paths = []
            all_costs = []
            for _ in range(self.num_ants):
                path, cost = self._simulate_ant(start_node, end_node)
                all_paths.append(path)
                all_costs.append(cost)
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
            self._update_pheromones(all_paths, all_costs)
        return best_path, best_cost

    def _simulate_ant(self, start_node, end_node):
        path = [start_node]
        current_node = start_node
        visited = set([start_node])
        while current_node != end_node:
            next_node = self._choose_next_node(current_node, visited)
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        cost = sum(self.graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
        return path, cost

    def _choose_next_node(self, current_node, visited):
        candidates = [i for i in range(self.num_nodes) if i not in visited and self.graph[current_node][i] > 0]
        if len(candidates) == 1:
            return candidates[0]
        probabilities = []
        total_pheromone = 0
        for node in candidates:
            pheromone = self.pheromone[current_node][node] ** self.alpha
            distance = self.graph[current_node][node] ** self.beta
            total_pheromone += pheromone / distance
        for node in candidates:
            pheromone = self.pheromone[current_node][node] ** self.alpha
            distance = self.graph[current_node][node] ** self.beta
            probability = (pheromone / distance) / total_pheromone
            probabilities.append(probability)
        if random.random() < self.q0:
            next_node = candidates[np.argmax(probabilities)]
        else:
            next_node = random.choices(candidates, probabilities)[0]
        return next_node

    def _update_pheromones(self, all_paths, all_costs):
        self.pheromone *= (1 - self.evaporation_rate)
        for path, cost in zip(all_paths, all_costs):
            pheromone_deposit = 1.0 / cost
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += pheromone_deposit

if __name__ == "__main__":
    graph = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 2, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 2, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 0]
    ])

    num_ants = 10
    num_iterations = 100
    alpha = 0.5
    beta = 2
    evaporation_rate = 0.1
    q0 = 0.9

    ant_colony = AntColony(graph, num_ants, num_iterations, alpha, beta, evaporation_rate, q0)
    start_node = 0
    end_node = 5
    best_path, best_cost = ant_colony.run(start_node, end_node)
    print(f"Best path from node {start_node} to node {end_node}: {best_path}")
    print(f"Path cost: {best_cost}")
Output:
Best path from node 0 to node 5: [0, 2, 5]
Path cost: 2
