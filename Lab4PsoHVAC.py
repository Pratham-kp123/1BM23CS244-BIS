import numpy as np

def hvac_energy(position):
    T_set = position[0]
    outside_temp = 35
    comfort_range = (22, 25)
    penalty = 0
    if T_set < comfort_range[0] or T_set > comfort_range[1]:
        penalty = 1000
    energy = abs(outside_temp - T_set) * 1.2
    return energy + penalty

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_value = hvac_energy(self.position)

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])
        value = hvac_energy(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = np.copy(self.position)

def particle_swarm_optimization(
    fitness_func,
    dim=1,
    num_particles=20,
    max_iter=50,
    bounds=(np.array([18]), np.array([30])),
    w=0.5,
    c1=1.5,
    c2=1.5
):
    swarm = [Particle(dim, bounds) for _ in range(num_particles)]
    global_best_position = swarm[0].best_position
    global_best_value = swarm[0].best_value

    for particle in swarm:
        if particle.best_value < global_best_value:
            global_best_value = particle.best_value
            global_best_position = particle.best_position

    for iter in range(max_iter):
        for particle in swarm:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position(bounds)
            if particle.best_value < global_best_value:
                global_best_value = particle.best_value
                global_best_position = particle.best_position

        print(f"Iteration {iter+1}/{max_iter}, Best Energy: {global_best_value:.4f}, Setpoint: {global_best_position[0]:.2f}°C")

    return global_best_position, global_best_value

best_pos, best_energy = particle_swarm_optimization(fitness_func=hvac_energy)

print("\n Optimal Temperature Setpoint Found:")
print(f"Setpoint: {best_pos[0]:.2f}°C")
print(f"Minimum Daily HVAC Energy Consumption: {best_energy:.2f} kWh")
