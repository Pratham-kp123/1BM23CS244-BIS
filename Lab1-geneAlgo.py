import random
import math


waypoints = [
    (0, 0), (2, 3), (4, 3), (6, 1), (3, 7), (7, 5), (8, 2)
]

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def path_length(path):
    length = 0
    for i in range(len(path) - 1):
        length += distance(waypoints[path[i]], waypoints[path[i+1]])
    return length


POP_SIZE = 20
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
GENERATIONS = 100


def create_chromosome():
    
    return random.sample(range(1, len(waypoints)), len(waypoints) - 1)

def create_population():
    return [create_chromosome() for _ in range(POP_SIZE)]


def fitness(chromosome):
    full_path = [0] + chromosome  
    return 1 / path_length(full_path)


def selection(population):
    total_fitness = sum(fitness(ch) for ch in population)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ch in population:
        current += fitness(ch)
        if current > pick:
            return ch


def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1[:]
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end+1] = parent1[start:end+1]

   
    p2_items = [item for item in parent2 if item not in child]
    pos = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = p2_items[pos]
            pos += 1
    return child


def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]


population = create_population()
best_solution = None
best_distance = float('inf')

for gen in range(GENERATIONS):
    new_population = []
    for _ in range(POP_SIZE):
        parent1 = selection(population)
        parent2 = selection(population)
        child = crossover(parent1, parent2)
        mutate(child)
        new_population.append(child)
    
    population = new_population
    
    for ch in population:
        full_path = [0] + ch
        dist = path_length(full_path)
        if dist < best_distance:
            best_distance = dist
            best_solution = full_path
    
    if gen % 10 == 0:
        print(f"Generation {gen}: Best Distance = {best_distance:.2f}")

print("\nBest Path:", best_solution)
print("Shortest Distance:", best_distance)

#Output:
#Generation 0: Best Distance = 23.70
#Generation 10: Best Distance = 18.30
#Generation 20: Best Distance = 18.30
#Generation 30: Best Distance = 18.30
#Generation 40: Best Distance = 18.30
#Generation 50: Best Distance = 18.30
#Generation 60: Best Distance = 18.30
#Generation 70: Best Distance = 18.30
#Generation 80: Best Distance = 18.30
#Generation 90: Best Distance = 18.30

#Best Path: [0, 1, 2, 3, 6, 5, 4]
#Shortest Distance: 18.30445999287793
