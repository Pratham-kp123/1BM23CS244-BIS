import numpy as np
import random

data = np.array([
    [120, 180, 95, 0],
    [145, 220, 130, 1],
    [160, 240, 150, 1],
    [110, 170, 85, 0],
    [130, 200, 120, 1],
    [115, 160, 90, 0],
    [150, 210, 140, 1],
    [125, 190, 100, 0]
])

X, y = data[:, :-1], data[:, -1]
attributes = ["BP", "CHOL", "GLUC"]

def random_rule():
    attr = random.choice(range(len(attributes)))
    op = random.choice([">", "<="])
    threshold = random.randint(80, 250)
    return (attr, op, threshold)

def rule_to_string(rule):
    attr, op, thresh = rule
    return f"{attributes[attr]} {op} {thresh}"

def evaluate_rule(rule, X, y):
    attr, op, thresh = rule
    if op == ">":
        preds = (X[:, attr] > thresh).astype(int)
    else:
        preds = (X[:, attr] <= thresh).astype(int)
    return np.mean(preds == y)

def crossover(rule1, rule2):
    if random.random() < 0.5:
        return (rule1[0], rule1[1], rule2[2])
    else:
        return (rule2[0], rule2[1], rule1[2])

def mutate(rule):
    attr, op, thresh = rule
    if random.random() < 0.3:
        attr = random.choice(range(len(attributes)))
    if random.random() < 0.3:
        op = random.choice([">", "<="])
    if random.random() < 0.3:
        thresh = random.randint(80, 250)
    return (attr, op, thresh)

def gene_expression_algorithm(X, y, pop_size=20, generations=30, mutation_rate=0.2):
    population = [random_rule() for _ in range(pop_size)]
    best_rule, best_fit = None, -1
    for g in range(generations):
        fitness_vals = [evaluate_rule(r, X, y) for r in population]
        best_idx = np.argmax(fitness_vals)
        if fitness_vals[best_idx] > best_fit:
            best_fit = fitness_vals[best_idx]
            best_rule = population[best_idx]
        print(f"Gen {g+1}: Best Rule = {rule_to_string(best_rule)}, Accuracy = {best_fit:.2f}")
        selected = [population[i] for i in np.argsort(fitness_vals)[-pop_size//2:]]
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_pop.append(child)
        population = new_pop
    return best_rule, best_fit

best_rule, accuracy = gene_expression_algorithm(X, y)
print("\nFinal Best Rule:", rule_to_string(best_rule), f"(Accuracy: {accuracy:.2f})")
