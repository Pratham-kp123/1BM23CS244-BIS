import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found. Please upload the correct image file.")
else:
    def fitness(threshold, image):
        if threshold is None:
            return -1
        
        foreground = image > threshold
        background = image <= threshold
        foreground_ratio = np.sum(foreground) / image.size
        background_ratio = np.sum(background) / image.size
        return abs(foreground_ratio - background_ratio)

    def init_population(pop_size):
        return [random.randint(0, 255) for _ in range(pop_size)]

    def crossover(parent1, parent2):
        child1 = (parent1 + parent2) // 2
        child2 = (parent1 + parent2) // 2
        return child1, child2

    def mutate(child, mutation_rate):
        if random.random() < mutation_rate:
            return random.randint(0, 255)
        return child

    def genetic_algorithm(image, pop_size=20, generations=50, mutation_rate=0.05):
        population = init_population(pop_size)
        best_threshold = None
        best_fitness = -1

        for generation in range(generations):
            fitness_values = [fitness(individual, image) for individual in population]
            
            if len(fitness_values) == 0 or None in fitness_values:
                break
            
            best_individual = population[np.argmax(fitness_values)]
            current_best_fitness = max(fitness_values)
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_threshold = best_individual

            selected_parents = [population[i] for i in np.argsort(fitness_values)[-pop_size//2:]]
            
            new_population = []
            while len(new_population) < pop_size:
                parent1, parent2 = random.sample(selected_parents, 2)
                child1, child2 = crossover(parent1, parent2)
                new_population.append(mutate(child1, mutation_rate))
                new_population.append(mutate(child2, mutation_rate))
            
            population = new_population[:pop_size]

        if best_threshold is None:
            best_threshold = random.randint(0, 255)
        
        return best_threshold

    best_threshold = genetic_algorithm(image)

    segmented_image = image > best_threshold

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title(f'Segmented Image (Threshold: {best_threshold})')
    plt.show()
