import numpy as np
from multiprocessing import Pool
import inspect
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parallel_cellular_optimization(func, bounds, grid_size=(10, 10), iterations=100,
                                   neighborhood='moore', check_samples=2000, plot=True):
    """
    Optimize a mathematical function using a Parallel Cellular Algorithm (PCA),
    check if extrema exist, and visualize the function surface (for 2D functions).
    """
    try:
        src = inspect.getsource(func)
        expr_line = [line.strip() for line in src.split('\n') if "return" in line][0]
        expr = expr_line.replace("return", "").strip()
        print(f"f(x) = {expr}")
    except Exception:
        print(f"f(x) = {func.__name__}(x)")
    print()
    
    rows, cols = grid_size
    dim = len(bounds)
    
    samples = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(check_samples, dim)
    )
    values = np.array([func(x) for x in samples])
    
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        print("⚠️ No finite values found for the given function in the given range.")
        sys.exit()
    
    approx_min, approx_max = np.min(finite_values), np.max(finite_values)
    if np.isinf(approx_min) or np.isinf(approx_max) or np.isnan(approx_min) or np.isnan(approx_max):
        print("⚠️ The function appears unbounded in the given range — no optimal values exist.")
        sys.exit()
    
    print(f"✅ Function check passed. Possible value range ≈ [{approx_min:.3f}, {approx_max:.3f}]")
    print("Starting optimization...\n")
    
    population = np.array([
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(rows * cols)
    ])
    
    def evaluate_population(pop):
        with Pool() as pool:
            fitness = pool.map(func, pop)
        return np.array(fitness)
    
    fitness = evaluate_population(population)
    
    def get_neighbors(index):
        r, c = divmod(index, cols)
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if neighborhood == 'von_neumann' and abs(dr) + abs(dc) != 1:
                    continue
                nr, nc = (r + dr) % rows, (c + dc) % cols
                neighbors.append(nr * cols + nc)
        return neighbors
    
    for it in range(iterations):
        new_population = population.copy()
        
        for i in range(len(population)):
            neighbors = get_neighbors(i)
            neighbor_best = min(neighbors, key=lambda j: fitness[j])
            
            alpha = np.random.uniform(0, 1, dim)
            new_population[i] = population[i] + alpha * (population[neighbor_best] - population[i])
            
            for d, (low, high) in enumerate(bounds):
                new_population[i, d] = np.clip(new_population[i, d], low, high)
        
        population = new_population
        fitness = evaluate_population(population)
        
        if (it + 1) % (iterations // 10) == 0:
            print(f"Iteration {it+1}/{iterations} | Best Value: {fitness.min():.6f}")
    
    min_idx = np.argmin(fitness)
    max_idx = np.argmax(fitness)
    min_val, max_val = float(fitness[min_idx]), float(fitness[max_idx])
    min_pos, max_pos = population[min_idx], population[max_idx]
    
    print("\n=== Optimization Completed ===")
    print(f"Minimum Value: {min_val:.6f} at {min_pos}")
    print(f"Maximum Value: {max_val:.6f} at {max_pos}")
    
    if abs(max_val - min_val) < 1e-5:
        print("⚠️ Warning: Minimum and maximum values are nearly identical. The function might be flat or search area too small.")
    
    if plot and dim == 2:
        print("\n📈 Plotting the function surface with extrema points...")
        x_range = np.linspace(bounds[0][0], bounds[0][1], 200)
        y_range = np.linspace(bounds[1][0], bounds[1][1], 200)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[func([x, y]) for x in x_range] for y in y_range])
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.scatter(min_pos[0], min_pos[1], min_val, color='red', s=80, label='Minimum')
        ax.scatter(max_pos[0], max_pos[1], max_val, color='blue', s=80, label='Maximum')
        ax.set_title(f"Function Surface: {func.__name__}")
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_zlabel('f(x₁, x₂)')
        ax.legend()
        plt.show()
    
    elif plot and dim != 2:
        print("⚠️ Plotting is only supported for 2D functions.")
    
    return {
        'min_value': min_val,
        'min_position': min_pos,
        'max_value': max_val,
        'max_position': max_pos
    }

if __name__ == "__main__":
    def wave_function(x):
        x1, x2 = x
        return np.sin(x1) * np.cos(x2) + 0.2 * np.sin(2 * x1 + x2)

    bounds = [(-5, 5), (-5, 5)]
    
    result = parallel_cellular_optimization(
        func=wave_function,
        bounds=bounds,
        grid_size=(15, 15),
        iterations=60,
        plot=True
    )
